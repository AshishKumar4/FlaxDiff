import jax
import jax.numpy as jnp
from .common import DiffusionSampler
from ..schedulers import get_coeff_shapes_tuple
from ..utils import RandomMarkovState

class EulerSampler(DiffusionSampler):
    # Basically a DDIM Sampler but parameterized as an ODE
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, sample_model_fn, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        shape = get_coeff_shapes_tuple(current_samples)
        current_alpha, current_sigma = self.noise_schedule.get_rates(current_step, shape)
        next_alpha, next_sigma = self.noise_schedule.get_rates(next_step, shape)

        dt = next_sigma - current_sigma
        
        x_0_coeff = (current_alpha * next_sigma - next_alpha * current_sigma) / (dt)
        dx = (current_samples - x_0_coeff * reconstructed_samples) / current_sigma
        next_samples = current_samples + dx * dt
        return next_samples, state

class SimplifiedEulerSampler(DiffusionSampler):
    """
    This is for networks with forward diffusion of the form x_{t+1} = x_t + sigma_t * epsilon_t
    """
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, sample_model_fn, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        shape = get_coeff_shapes_tuple(current_samples)
        _, current_sigma = self.noise_schedule.get_rates(current_step, shape)
        _, next_sigma = self.noise_schedule.get_rates(next_step, shape)

        dt = next_sigma - current_sigma
        
        dx = (current_samples - reconstructed_samples) / current_sigma
        next_samples = current_samples + dx * dt
        return next_samples, state
    
class EulerAncestralSampler(DiffusionSampler):
    """
    Similar to EulerSampler but with ancestral sampling
    """
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, sample_model_fn, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        shape = get_coeff_shapes_tuple(current_samples)
        current_alpha, current_sigma = self.noise_schedule.get_rates(current_step, shape)
        next_alpha, next_sigma = self.noise_schedule.get_rates(next_step, shape)

        sigma_up = (next_sigma**2 * (current_sigma**2 - next_sigma**2) / current_sigma**2) ** 0.5
        sigma_down = (next_sigma**2 - sigma_up**2) ** 0.5
        
        dt = sigma_down - current_sigma
        
        x_0_coeff = (current_alpha * next_sigma - next_alpha * current_sigma) / (next_sigma - current_sigma)
        dx = (current_samples - x_0_coeff * reconstructed_samples) / current_sigma
        
        state, subkey = state.get_random_key()
        dW = jax.random.normal(subkey, current_samples.shape) * sigma_up
        
        next_samples = current_samples + dx * dt + dW
        return next_samples, state