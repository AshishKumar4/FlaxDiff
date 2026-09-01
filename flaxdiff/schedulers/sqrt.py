import jax.numpy as jnp
from .continuous import ContinuousNoiseScheduler
from .common import reshape_rates


class SqrtContinuousNoiseScheduler(ContinuousNoiseScheduler):
    """Square-root schedule from Diffusion-LM (Li et al. 2022).

    alpha(t) = sqrt(1 - t) and sigma(t) = sqrt(t) for t in [0, 1], so it is
    variance preserving (alpha^2 + sigma^2 = 1) with SNR(t) = (1 - t) / t.
    Noise ramps up much faster near t = 0 than the cosine schedule, which is
    what the paper wanted for discrete/embedding data: the low-noise end
    carries little signal about the token identity, so spending fewer steps
    there is a win.
    """
    def get_rates(self, steps, shape=(-1, 1, 1, 1)) -> tuple[jnp.ndarray, jnp.ndarray]:
        steps = jnp.asarray(steps, dtype=self.dtype)
        signal_rates = jnp.sqrt(1 - steps)
        noise_rates = jnp.sqrt(steps)
        return reshape_rates((signal_rates, noise_rates), shape=shape)
