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

@dataclass
class ConditioningEncoder(ABC):
    model: nn.Module
    tokenizer: Callable
    
    @property
    def key(self):
        name = self.tokenizer.__name__
        # Remove the 'Encoder' suffix from the name and lowercase it
        if name.endswith("Encoder"):
            name = name[:-7].lower()
        return name

    def __call__(self, data):
        tokens = self.tokenize(data)
        outputs = self.encode_from_tokens(tokens)
        return outputs
        
    def encode_from_tokens(self, tokens):
        outputs = self.model(input_ids=tokens['input_ids'],
                        attention_mask=tokens['attention_mask'])
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state
    
    def tokenize(self, data):
        tokens = self.tokenizer(data, padding="max_length",
                        max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="np")
        return tokens
    
@dataclass
class TextEncoder(ConditioningEncoder):
    @property
    def key(self):
        return "text"

@dataclass
class ConditionalInputConfig:
    """Class representing a conditional input for the model."""
    encoder: ConditioningEncoder
    conditioning_data_key: str = None       # Key in the batch for this conditioning input
    pretokenized: bool = False
    unconditional_input: Any = None
    model_key_override: Optional[str] = None  # Optional key override for the model
    
    def __call__(self, batch_data):
        """Process batch data to produce conditioning."""
        key =  self.conditioning_data_key if self.conditioning_data_key else self.encoder.key
        if self.pretokenized:
            return self.encoder.encode_from_tokens(batch_data[key])
        return self.encoder(batch_data[key])
    
    def get_unconditional(self):
        """Get unconditional version of this input."""
        if self.unconditional_input is not None:
            return self.encoder([self.unconditional_input])
        return self.encoder([""])  # Default empty text
    
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
        