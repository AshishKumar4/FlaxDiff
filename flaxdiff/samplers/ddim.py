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
            # DDIM paper eq. 16: eta=0 is deterministic DDIM, eta=1.0 approaches DDPM.
            # The direction term must shrink to keep the marginal variance right.
            sigma_tilde = self.eta * (sigma_next / sigma_t) * jnp.sqrt(
                jnp.maximum(1 - alpha_t**2 / alpha_next**2, 0.0))
            state, noise_key = state.get_random_key()
            noise = jax.random.normal(noise_key, current_samples.shape)
            direction_coeff = jnp.sqrt(jnp.maximum(sigma_next**2 - sigma_tilde**2, 0.0))
            new_samples = (alpha_next * reconstructed_samples
                           + direction_coeff * pred_noise
                           + sigma_tilde * noise)
        else:
            # Direct DDIM update formula
            new_samples = alpha_next * reconstructed_samples + sigma_next * pred_noise
        
        return new_samples, state
