import jax.numpy as jnp
from .common import DiffusionSampler
from ..utils import MarkovState, RandomMarkovState
import jax
from flaxdiff.schedulers import get_coeff_shapes_tuple

class DDIMSampler(DiffusionSampler):
    def __init__(self, *args, eta=0.0, **kwargs):
        """Initialize DDIM sampler with customizable noise level.
        
        Args:
            eta: Controls the stochasticity of the sampler. 
                 0.0 = deterministic (DDIM), 1.0 = DDPM-like.
        """
        super().__init__(*args, **kwargs)
        self.eta = eta
        
    def take_next_step(
        self, 
        current_samples, 
        reconstructed_samples, 
        model_conditioning_inputs,
        pred_noise, 
        current_step, 
        state: RandomMarkovState, 
        sample_model_fn, 
        next_step=1
    ) -> tuple[jnp.ndarray, RandomMarkovState]:
        # Get diffusion coefficients for current and next timesteps
        alpha_t, sigma_t = self.noise_schedule.get_rates(current_step, get_coeff_shapes_tuple(current_samples))
        alpha_next, sigma_next = self.noise_schedule.get_rates(next_step, get_coeff_shapes_tuple(current_samples))
        
        # Extract random noise if needed for stochastic sampling
        if self.eta > 0:
            # For DDIM, we need to compute the variance coefficient
            # This is based on the original DDIM paper's formula
            # When eta=0, it's deterministic DDIM, when eta=1.0 it approaches DDPM
            sigma_tilde = self.eta * sigma_next * (1 - alpha_t**2 / alpha_next**2).sqrt() / (1 - alpha_t**2).sqrt()
            state, noise_key = state.get_random_key()
            noise = jax.random.normal(noise_key, current_samples.shape)
            # Add the stochastic component
            stochastic_term = sigma_tilde * noise
        else:
            stochastic_term = 0
            
        # Direct DDIM update formula
        new_samples = alpha_next * reconstructed_samples + sigma_next * pred_noise + stochastic_term
        
        return new_samples, state
