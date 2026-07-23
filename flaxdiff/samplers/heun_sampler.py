import jax
import jax.numpy as jnp
from .common import DiffusionSampler
from ..schedulers import get_coeff_shapes_tuple
from ..utils import RandomMarkovState

class HeunSampler(DiffusionSampler):
    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, sample_model_fn, next_step=1) -> tuple[jnp.ndarray, RandomMarkovState]:
        # Get the noise and signal rates for the current and next steps
        shape = get_coeff_shapes_tuple(current_samples)
        current_alpha, current_sigma = self.noise_schedule.get_rates(current_step, shape)
        next_alpha, next_sigma = self.noise_schedule.get_rates(next_step, shape)

        dt = next_sigma - current_sigma
        x_0_coeff = (current_alpha * next_sigma - next_alpha * current_sigma) / dt

        dx_0 = (current_samples - x_0_coeff * reconstructed_samples) / current_sigma
        next_samples_0 = current_samples + dx_0 * dt
        
        # Recompute x_0 and eps at the first estimate to refine the derivative
        estimated_x_0, _, _ = sample_model_fn(next_samples_0, next_step, *model_conditioning_inputs)
        
        # Estimate the refined derivative using the midpoint (Heun's method)
        safe_next_sigma = jnp.where(next_sigma > 0, next_sigma, 1.0)
        dx_1 = (next_samples_0 - x_0_coeff * estimated_x_0) / safe_next_sigma
        # Compute the final next samples by averaging the initial and refined derivatives.
        # When sigma reaches 0 there is no derivative there - fall back to the euler step
        # (Karras et al. 2022, Algorithm 2)
        final_next_samples = jnp.where(
            next_sigma > 0,
            current_samples + 0.5 * (dx_0 + dx_1) * dt,
            next_samples_0,
        )
        
        return final_next_samples, state
