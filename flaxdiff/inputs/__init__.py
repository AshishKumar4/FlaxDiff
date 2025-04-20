import jax
import jax.numpy as jnp
import flax.struct as struct
import flax.linen as nn
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import partial
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from abc import ABC, abstractmethod

from flaxdiff.models.autoencoder import AutoEncoder
from .encoders import *

@dataclass
class ConditionalInputConfig:
    """Class representing a conditional input for the model."""
    encoder: ConditioningEncoder
    conditioning_data_key: str = None       # Key in the batch for this conditioning input
    pretokenized: bool = False
    unconditional_input: Any = None
    model_key_override: Optional[str] = None  # Optional key override for the model
    
    __uncond_cache__ = None  # Cache for unconditional input
    
    def __post_init__(self):
        if self.unconditional_input is not None:
            uncond = self.encoder([self.unconditional_input])
        else:
            uncond = self.encoder([""])  # Default empty text
        self.__uncond_cache__ = uncond  # Cache the unconditional input
    
    def __call__(self, batch_data):
        """Process batch data to produce conditioning."""
        key =  self.conditioning_data_key if self.conditioning_data_key else self.encoder.key
        if self.pretokenized:
            return self.encoder.encode_from_tokens(batch_data[key])
        return self.encoder(batch_data[key])
    
    def get_unconditional(self):
        """Get unconditional version of this input."""
        return self.__uncond_cache__
    
    def serialize(self):
        """Serialize the configuration."""
        serialized_config = {
            "encoder": self.encoder.serialize(),
            "encoder_key": self.encoder.key,
            "conditioning_data_key": self.conditioning_data_key,
            "unconditional_input": self.unconditional_input,
            "model_key_override": self.model_key_override,
        }
        return serialized_config
    
    @staticmethod
    def deserialize(serialized_config):
        """Deserialize the configuration."""
        encoder_key = serialized_config["encoder_key"]
        encoder_class = CONDITIONAL_ENCODERS_REGISTRY.get(encoder_key)
        if encoder_class is None:
            raise ValueError(f"Unknown encoder type: {encoder_key}")
        
        # Create the encoder instance
        encoder = encoder_class.deserialize(serialized_config["encoder"])
        # Deserialize the rest of the configuration
        conditioning_data_key = serialized_config.get("conditioning_data_key")
        unconditional_input = serialized_config.get("unconditional_input")
        model_key_override = serialized_config.get("model_key_override")
        return ConditionalInputConfig(
            encoder=encoder,
            conditioning_data_key=conditioning_data_key,
            unconditional_input=unconditional_input,
            model_key_override=model_key_override,
        )
    
@dataclass
class DiffusionInputConfig:
    """Configuration for the input data."""
    sample_data_key: str         # Key in the batch for the sample data
    sample_data_shape: Tuple[int, ...]
    conditions: List[ConditionalInputConfig]
    
    def get_input_shapes(
        self, 
        autoencoder: AutoEncoder = None, 
        sample_model_key:str = 'x',
        time_embeddings_model_key:str = 'temb',
    ) -> Dict[str, Tuple[int, ...]]:
        """Get the shapes of the input data."""
        if len(self.sample_data_shape) == 3:
            H, W, C = self.sample_data_shape
        elif len(self.sample_data_shape) == 4:
            T, H, W, C = self.sample_data_shape
        else:
            raise ValueError(f"Unsupported shape for sample data {self.sample_data_shape}")
        if autoencoder is not None:
            downscale_factor = autoencoder.downscale_factor
            H = H // downscale_factor
            W = W // downscale_factor
            C = autoencoder.latent_channels
        
        input_shapes = {
            sample_model_key: (H, W, C),
            time_embeddings_model_key: (),
        }
        for cond in self.conditions:
            # Get the shape of the conditioning data by calling the get_unconditional method
            unconditional = cond.get_unconditional()
            key = cond.model_key_override if cond.model_key_override else cond.encoder.key
            input_shapes[key] = unconditional[0].shape
            
        print(f"Calculated input shapes: {input_shapes}")
        return input_shapes
    
    def get_unconditionals(self):
        """Get unconditional inputs for all conditions."""
        unconditionals = []
        for cond in self.conditions:
            uncond = cond.get_unconditional()
            unconditionals.append(uncond)
        return unconditionals
    
    def process_conditioning(self, batch_data, uncond_mask: Optional[jnp.ndarray] = None):
        """Process the conditioning data."""
        results = []
            
        for cond in self.conditions:
            cond_embeddings = cond(batch_data)
            if uncond_mask is not None:
                assert len(uncond_mask) == len(cond_embeddings), "Unconditional mask length must match the batch size."
                uncond_embedding = cond.get_unconditional()
                    
                # Reshape uncond_mask to be broadcastable with the conditioning embeddings
                # If cond_embeddings has shape (B, T, D), reshape uncond_mask to (B, 1, 1)
                broadcast_shape = [len(uncond_mask)] + [1] * (cond_embeddings.ndim - 1)
                reshaped_mask = jnp.reshape(uncond_mask, broadcast_shape)
                    
                # Repeat uncond_embedding to match batch size
                batch_size = len(cond_embeddings)
                repeated_uncond = jnp.repeat(uncond_embedding, batch_size, axis=0)
                    
                # Apply unconditional embedding based on the mask
                cond_embeddings = jnp.where(reshaped_mask, repeated_uncond, cond_embeddings)
                
            results.append(cond_embeddings)
        return results
    
    def serialize(self):
        """Serialize the configuration."""
        serialized_config = {
            "sample_data_key": self.sample_data_key,
            "sample_data_shape": self.sample_data_shape,
            "conditions": [cond.serialize() for cond in self.conditions],
        }
        return serialized_config
    
    @staticmethod
    def deserialize(serialized_config):
        """Deserialize the configuration."""
        sample_data_key = serialized_config["sample_data_key"]
        sample_data_shape = tuple(serialized_config["sample_data_shape"])
        conditions = serialized_config["conditions"]
        
        # Deserialize each condition
        deserialized_conditions = []
        for cond in conditions:
            deserialized_conditions.append(ConditionalInputConfig.deserialize(cond))
        
        return DiffusionInputConfig(
            sample_data_key=sample_data_key,
            sample_data_shape=sample_data_shape,
            conditions=deserialized_conditions,
        )