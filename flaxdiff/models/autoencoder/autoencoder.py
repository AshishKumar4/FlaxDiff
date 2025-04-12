import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Callable, Sequence, Any, Union, Optional
import einops
from ..common import kernel_init, ConvLayer, Upsample, Downsample, PixelShuffle
from dataclasses import dataclass

@dataclass
class AutoEncoder():
    """Base class for autoencoder models with video support.
    
    This class defines the interface for autoencoders and provides
    video handling functionality, allowing child classes to focus
    on implementing the core encoding/decoding for individual frames.
    """
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
        # Check if we're dealing with video data (5D tensor)
        if len(x.shape) == 5:  # [B, T, H, W, C]
            batch_size, time_steps, height, width, channels = x.shape
            
            # Reshape to [B*T, H, W, C] for frame-by-frame encoding
            x_reshaped = x.reshape(-1, height, width, channels)
            
            # Encode each frame
            encoded_frames = self.__encode__(x_reshaped, key=key, **kwargs)
            
            # Get the new dimensions
            _, enc_h, enc_w, enc_c = encoded_frames.shape
            
            # Reshape back to [B, T, enc_h, enc_w, enc_c]
            return encoded_frames.reshape(batch_size, time_steps, enc_h, enc_w, enc_c)
        else:
            # Standard image encoding (4D tensor)
            return self.__encode__(x, key=key, **kwargs)
    
    def decode(self, z: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None, **kwargs) -> jnp.ndarray:
        """Decode latent representation, with special handling for video data.
        
        This method handles both standard latent batches and video latents (5D tensors).
        For videos, it reshapes the input, processes each frame, and then restores
        the temporal dimension.
        
        Args:
            z: Latent tensor, either [B, h, w, c] for images or [B, T, h, w, c] for videos
            key: Optional random key for stochastic decoding
            **kwargs: Additional arguments passed to __decode__
            
        Returns:
            Decoded data with the same batch and temporal dimensions as input
        """
        # Check if we're dealing with video data (5D tensor)
        if len(z.shape) == 5:  # [B, T, h, w, c]
            batch_size, time_steps, height, width, channels = z.shape
            
            # Reshape to [B*T, h, w, c] for frame-by-frame decoding
            z_reshaped = z.reshape(-1, height, width, channels)
            
            # Decode each frame
            decoded_frames = self.__decode__(z_reshaped, key=key, **kwargs)
            
            # Get the new dimensions
            _, dec_h, dec_w, dec_c = decoded_frames.shape
            
            # Reshape back to [B, T, dec_h, dec_w, dec_c]
            return decoded_frames.reshape(batch_size, time_steps, dec_h, dec_w, dec_c)
        else:
            # Standard latent decoding (4D tensor)
            return self.__decode__(z, key=key, **kwargs)
    
    def __call__(self, x: jnp.ndarray, key: Optional[jax.random.PRNGKey] = None, **kwargs):
        """Encode and then decode the input.
        
        Args:
            x: Input tensor, either images or videos
            key: Optional random key
            **kwargs: Additional arguments
            
        Returns:
            Reconstructed data
        """
        # Split the key if provided
        if key is not None:
            encode_key, decode_key = jax.random.split(key)
        else:
            encode_key, decode_key = None, None
            
        latents = self.encode(x, key=encode_key, **kwargs)
        reconstructions = self.decode(latents, key=decode_key, **kwargs)
        return reconstructions
    
    @property
    def downscale_factor(self) -> int:
        """Returns the downscale factor for the encoder."""
        raise 8
    
    @property 
    def latent_channels(self) -> int:
        """Returns the number of channels in the latent space."""
        raise 4