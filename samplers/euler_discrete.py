import jax
import jax.numpy as jnp
from .common import DiffusionSampler
from ..utils import MarkovState
class EulerDiscreteSampler(DiffusionSampler):
    # Basically a DDIM Sampler but parameterized as an ODE
    def _renoise(self, 
                 current_samples, reconstructed_samples, 
                 pred_noise, current_step, state:MarkovState, next_step=None) -> tuple[jnp.ndarray, MarkovState]:
        current_signal_rate, current_noise_rate = self.noise_schedule.get_rates(current_step)
        next_signal_rate, next_noise_rate = self.noise_schedule.get_rates(next_step)
        current_sigma = current_noise_rate
        next_sigma = next_noise_rate

        dt = next_sigma - current_sigma
        
        x_0_coeff = (current_signal_rate * next_noise_rate - next_signal_rate * current_noise_rate) / (dt)
        dx = (current_samples - x_0_coeff * reconstructed_samples) / current_noise_rate
        next_samples = current_samples + dx * dt
        return next_samples, state