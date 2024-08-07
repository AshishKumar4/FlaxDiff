import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Callable, Sequence, Any, Union
import einops
from ..common import kernel_init, ConvLayer, Upsample, Downsample, PixelShuffle


class AutoEncoder():
    def encode(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def decode(self, z: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def __call__(self, x: jnp.ndarray):
        latents = self.encode(x)
        reconstructions = self.decode(latents)
        return reconstructions