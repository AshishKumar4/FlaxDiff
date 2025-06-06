import numpy as np
import jax.numpy as jnp
from .discrete import DiscreteNoiseScheduler
from .continuous import ContinuousNoiseScheduler
from .common import reshape_rates

class SqrtContinuousNoiseScheduler(ContinuousNoiseScheduler):
    def get_rates(self, steps, shape=(-1, 1, 1, 1)) -> tuple[jnp.ndarray, jnp.ndarray]:
        signal_rates = jnp.sqrt(1 - steps)
        noise_rates = jnp.sqrt(steps)
        return reshape_rates((signal_rates, noise_rates), shape=shape)
