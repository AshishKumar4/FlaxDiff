import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Callable, Sequence, Any, Union
import einops
from ..common import kernel_init, ConvLayer, Upsample, Downsample, PixelShuffle


class AutoEncoder(nn.Module):
    def encode(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    def decode(self, z: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError
    
    @nn.compact
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError