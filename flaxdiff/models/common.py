import jax.numpy as jnp
from flax import linen as nn

# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return nn.initializers.variance_scaling(scale=scale, mode="fan_in", distribution="truncated_normal")
