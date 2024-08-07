from typing import Any, List, Optional, Callable
import jax
import flax.linen as nn
from jax import numpy as jnp
from flax.typing import Dtype, PrecisionLike
from .autoencoder import AutoEncoder

class SimpleAutoEncoder(AutoEncoder):
    latent_channels: int
    feature_depths: List[int]=[64, 128, 256, 512]
    attention_configs:list=[{"heads":8}, {"heads":8}, {"heads":8}, {"heads":8}],
    num_res_blocks: int=2
    num_middle_res_blocks:int=1,
    activation:Callable = jax.nn.swish
    norm_groups:int=8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    
    # def encode(self, x: jnp.ndarray):
        
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        latents = self.encode(x)
        reconstructions = self.decode(latents)
        return reconstructions