import jax
import jax.numpy as jnp
from .common import DiffusionSampler
from ..utils import MarkovState

class EulerDiscreteSampler(DiffusionSampler):
    def _renoise(self, 
                 current_samples, reconstructed_samples, 
                 pred_noise, current_step, state:MarkovState, next_step=None) -> tuple[jnp.ndarray, MarkovState]:
        current_signal_rate, current_noise_rate = self.noise_schedule.get_rates(current_step)
        next_signal_rate, next_noise_rate = self.noise_schedule.get_rates(next_step)
        # current_noise_rate = current_noise_rate / current_signal_rate
        # next_noise_rate = next_noise_rate / next_signal_rate
        dt = next_noise_rate - current_noise_rate
        coeff = (current_signal_rate * next_noise_rate - next_signal_rate * current_noise_rate)
        x_0_coeff = coeff / dt
        dx = (current_samples - x_0_coeff * reconstructed_samples) / current_noise_rate
        # dx = (current_samples - reconstructed_samples) / current_noise_rate
        # dx =  pred_noise
        next_samples = current_samples + dx * dt
        return next_samples, state
