import jax
import jax.numpy as jnp
from .common import DiffusionSampler
from ..schedulers import get_coeff_shapes_tuple
from ..utils import MarkovState, RandomMarkovState

class DDPMSampler(DiffusionSampler):
    """Exact ancestral sampler for the reverse diffusion SDE.

    This is the simplified (but algebraically exact) form of the DDPM
    posterior, phrased purely in terms of signal/noise rates so it works for
    any schedule and any step stride, not just t -> t-1.
    """
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, sample_model_fn, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        noise = jax.random.normal(rng, reconstructed_samples.shape, dtype=jnp.float32)

        shape = get_coeff_shapes_tuple(current_samples)
        current_signal_rate, current_noise_rate = self.noise_schedule.get_rates(current_step, shape)
        next_signal_rate, next_noise_rate = self.noise_schedule.get_rates(next_step, shape)

        pred_noise_coeff = ((next_noise_rate ** 2) * current_signal_rate) / (current_noise_rate * next_signal_rate)

        noise_ratio_squared = (next_noise_rate ** 2) / (current_noise_rate ** 2)
        signal_ratio_squared = (current_signal_rate ** 2) / (next_signal_rate ** 2)
        gamma = jnp.sqrt(noise_ratio_squared * (1 - signal_ratio_squared))

        next_samples = next_signal_rate * reconstructed_samples + pred_noise_coeff * pred_noise + noise * gamma
        return next_samples, state

# The two used to be separate implementations; the posterior-table variant
# crashed for batched steps and could not handle strided sampling
SimpleDDPMSampler = DDPMSampler
