import jax.numpy as jnp
from .common import DiffusionSampler
from ..utils import MarkovState

class DDIMSampler(DiffusionSampler):
    def _renoise(self, 
                 current_samples, reconstructed_samples, 
                 pred_noise, current_step, state:MarkovState, next_step=None) -> tuple[jnp.ndarray, MarkovState]:
        # state, key = state.get_random_key()
        # newnoise = jax.random.normal(key, reconstructed_samples.shape, dtype=jnp.float32)
        return self.noise_schedule.add_noise(reconstructed_samples, pred_noise, next_step), state