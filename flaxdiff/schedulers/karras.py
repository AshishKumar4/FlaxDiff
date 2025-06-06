import jax.numpy as jnp
from .common import GeneralizedNoiseScheduler
import math
import jax
from ..utils import RandomMarkovState

class KarrasVENoiseScheduler(GeneralizedNoiseScheduler):
    def __init__(self, timesteps=1.0, sigma_min=0.002, sigma_max=80, rho=7., sigma_data=0.5, *args, **kwargs):
        super().__init__(timesteps=timesteps, sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data, *args, **kwargs)
        self.min_inv_rho = sigma_min ** (1 / rho)
        self.max_inv_rho = sigma_max ** (1 / rho)
        self.rho = rho
        
    def get_sigmas(self, steps) -> jnp.ndarray:
        # Ensure steps are properly normalized and clamped to avoid edge cases
        ramp = jnp.clip(1 - steps / self.max_timesteps, 0.0, 1.0)
        sigmas = (self.max_inv_rho + ramp * (self.min_inv_rho - self.max_inv_rho)) ** self.rho
        return sigmas
        
    def get_weights(self, steps, shape=(-1, 1, 1, 1)) -> jnp.ndarray:
        sigma = self.get_sigmas(steps)
        # Add epsilon for numerical stability
        epsilon = 1e-6
        weights = ((sigma ** 2 + self.sigma_data ** 2) / ((sigma * self.sigma_data) ** 2 + epsilon))
        return weights.reshape(shape)
    
    def transform_inputs(self, x, steps, num_discrete_chunks=1000) -> tuple[jnp.ndarray, jnp.ndarray]:
        sigmas = self.get_sigmas(steps)
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-12
        sigmas = jnp.log(sigmas + epsilon) / 4
        return x, sigmas
    
    def get_timesteps(self, sigmas:jnp.ndarray) -> jnp.ndarray:
        sigmas = sigmas.reshape(-1)
        # Add epsilon for numerical stability
        epsilon = 1e-12
        inv_rho = (sigmas + epsilon) ** (1 / self.rho)
        # Ensure proper clamping to avoid numerical issues
        denominator = (self.min_inv_rho - self.max_inv_rho)
        if abs(denominator) < 1e-7:
            denominator = jnp.sign(denominator) * 1e-7
        ramp = jnp.clip((inv_rho - self.max_inv_rho) / denominator, 0.0, 1.0)
        steps = jnp.clip(1 - ramp, 0.0, 1.0) * self.max_timesteps
        return steps
    
    def generate_timesteps(self, batch_size, state:RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        timesteps, state = super().generate_timesteps(batch_size, state)
        timesteps = timesteps.astype(jnp.float32)
        return timesteps, state
    
class SimpleExpNoiseScheduler(KarrasVENoiseScheduler):
    def __init__(self, timesteps, sigma_min=0.002, sigma_max=80, rho=7., sigma_data=0.5, *args, **kwargs):
        super().__init__(timesteps=timesteps, sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data, *args, **kwargs)
        if type(timesteps) == int and timesteps > 1:
            n = timesteps
        else:
            n = 1000
        self.sigmas = jnp.exp(jnp.linspace(math.log(sigma_min), math.log(sigma_max), n))

    def get_sigmas(self, steps) -> jnp.ndarray:
        steps = jnp.int16(steps)
        return self.sigmas[steps]

class EDMNoiseScheduler(KarrasVENoiseScheduler):
    def __init__(self, timesteps, sigma_min=0.002, sigma_max=80, rho=7., sigma_data=0.5, *args, **kwargs):
        super().__init__(timesteps=timesteps, sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data, *args, **kwargs)

    def get_sigmas(self, steps, std=1.2, mean=-1.2) -> jnp.ndarray:
        space = steps / self.max_timesteps
        # space = jax.scipy.special.erfinv(self.erf_sigma_min + steps * (self.erf_sigma_max - self.erf_sigma_min))
        return jnp.exp(space * std + mean)
    
    def generate_timesteps(self, batch_size, state:RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        timesteps = jax.random.normal(rng, (batch_size,), dtype=jnp.float32)
        return timesteps, state