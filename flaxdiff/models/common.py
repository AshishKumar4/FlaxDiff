import jax.numpy as jnp
from flax import linen as nn

# Kernel initializer to use
def kernel_init(scale, dtype=jnp.float32):
    scale = max(scale, 1e-10)
    return nn.initializers.variance_scaling(scale=scale, mode="fan_avg", distribution="truncated_normal", dtype=dtype)
