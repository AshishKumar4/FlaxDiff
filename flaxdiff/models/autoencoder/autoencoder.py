import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Callable, Sequence, Any, Union, Optional
import einops
from ..common import kernel_init, ConvLayer, Upsample, Downsample, PixelShuffle
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class AutoEncoder(ABC):
    """Base class for autoencoder models with video support.
    
    This class defines the interface for autoencoders and provides
    video handling functionality, allowing child classes to focus
    on implementing the core encoding/decoding for individual frames.
    """
    @abstractmethod
    def __encode__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Abstract method for encoding a batch of images.
        
        Child classes must implement this method to perform the actual encoding.
        
        Args:
            x: Input tensor of shape [B, H, W, C] (batch of images)
            **kwargs: Additional arguments for the encoding process
            
        Returns:
            Encoded latent representation
        """
        raise NotImplementedError
    
    @abstractmethod
    def __decode__(self, z: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Abstract method for decoding a batch of latents.
        
        Child classes must implement this method to perform the actual decoding.
        
        Args:
            z: Latent tensor of shape [B, h, w, c] (encoded representation)
            **kwargs: Additional arguments for the decoding process
            
        Returns:
            Decoded images
        """
        raise NotImplementedError
    
    def encode(self, x: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None, **kwargs) -> jnp.ndarray:
        """Encode input data, with special handling for video data.
        
        This method handles both standard image batches and video data (5D tensors).
        For videos, it reshapes the input, processes each frame, and then restores
        the temporal dimension.
        
        Args:
            x: Input tensor, either [B, H, W, C] for images or [B, T, H, W, C] for videos
            key: Optional random key for stochastic encoding
            **kwargs: Additional arguments passed to __encode__
            
        Returns:
            Encoded representation with the same batch and temporal dimensions as input
        """
        # Check for video data (5D tensor)
        is_video = len(x.shape) == 5
        
        if is_video:
            # Extract dimensions for reshaping
            batch_size, seq_len, height, width, channels = x.shape
            
            # Reshape to [B*T, H, W, C] to process as regular images
            x_reshaped = x.reshape(-1, height, width, channels)
            
            # Encode all frames
            latent = self.__encode__(x_reshaped, key=key, **kwargs)
            
            # Reshape back to include temporal dimension [B, T, h, w, c]
            latent_shape = latent.shape
            return latent.reshape(batch_size, seq_len, *latent_shape[1:])
        else:
            # Standard image processing
            return self.__encode__(x, key=key, **kwargs)
    
    def decode(self, z: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None, **kwargs) -> jnp.ndarray:
        """Decode latent representations, with special handling for video data.
        
        This method handles both standard image latents and video latents (5D tensors).
        For videos, it reshapes the input, processes each frame, and then restores
        the temporal dimension.
        
        Args:
            z: Latent tensor, either [B, h, w, c] for images or [B, T, h, w, c] for videos
            key: Optional random key for stochastic decoding
            **kwargs: Additional arguments passed to __decode__
            
        Returns:
            Decoded output with the same batch and temporal dimensions as input
        """
        # Check for video data (5D tensor)
        is_video = len(z.shape) == 5
        
        if is_video:
            # Extract dimensions for reshaping
            batch_size, seq_len, height, width, channels = z.shape
            
            # Reshape to [B*T, h, w, c] to process as regular latents
            z_reshaped = z.reshape(-1, height, width, channels)
            
            # Decode all frames
            decoded = self.__decode__(z_reshaped, key=key, **kwargs)
            
            # Reshape back to include temporal dimension [B, T, H, W, C]
            decoded_shape = decoded.shape
            return decoded.reshape(batch_size, seq_len, *decoded_shape[1:])
        else:
            # Standard latent processing
            return self.__decode__(z, key=key, **kwargs)
    
    def __call__(self, x: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None, **kwargs):
        """Encode and then decode the input (autoencoder).
        
        Args:
            x: Input tensor, either [B, H, W, C] for images or [B, T, H, W, C] for videos
            key: Optional random key for stochastic encoding/decoding
            **kwargs: Additional arguments for encoding and decoding
            
        Returns:
            Reconstructed output with the same dimensions as input
        """
        if key is not None:
            encode_key, decode_key = jax.random.split(key)
        else:
            encode_key = decode_key = None
            
        # Encode then decode
        z = self.encode(x, key=encode_key, **kwargs)
        return self.decode(z, key=decode_key, **kwargs)
    
    @property
    def spatial_scale(self) -> int:
        """Get the spatial scale factor between input and latent spaces."""
        return getattr(self, "_spatial_scale", None)
    
    @property
    def name(self) -> str:
        """Get the name of the autoencoder model."""
        raise NotImplementedError
    
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize the model parameters and configuration."""
        raise NotImplementedError