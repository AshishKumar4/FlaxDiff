import jax
import jax.numpy as jnp
from typing import Union
from ..utils import RandomMarkovState  

class NoiseScheduler():
    def __init__(self, timesteps,
                    dtype=jnp.float32,
                    clip_min=-1.0,
                    clip_max=1.0,
                    *args, **kwargs):
        self.max_timesteps = timesteps
        self.dtype = dtype
        self.clip_min = clip_min
        self.clip_max = clip_max
        if type(timesteps) == int and timesteps > 1:
            timestep_generator = lambda rng, batch_size, max_timesteps = timesteps: jax.random.randint(rng, (batch_size,), 0, max_timesteps)
        else:
            timestep_generator = lambda rng, batch_size, max_timesteps = timesteps: jax.random.uniform(rng, (batch_size,), minval=0, maxval=max_timesteps)
        self.timestep_generator = timestep_generator

    def generate_timesteps(self, batch_size, state:RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        timesteps = self.timestep_generator(rng, batch_size, self.max_timesteps)
        return timesteps, state
    
    def get_weights(self, steps):
        raise NotImplementedError
    
    def reshape_rates(self, rates:tuple[jnp.ndarray, jnp.ndarray], shape=(-1, 1, 1, 1)) -> tuple[jnp.ndarray, jnp.ndarray]:
        signal_rates, noise_rates = rates
        signal_rates = jnp.reshape(signal_rates, shape)
        noise_rates = jnp.reshape(noise_rates, shape)
        return signal_rates, noise_rates
    
    def get_rates(self, steps, shape=(-1, 1, 1, 1)) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError
    
    def add_noise(self, images, noise, steps) -> jnp.ndarray:
        signal_rates, noise_rates = self.get_rates(steps)
        return signal_rates * images + noise_rates * noise
    
    def remove_all_noise(self, noisy_images, noise, steps, clip_denoised=True, rates=None):
        signal_rates, noise_rates = self.get_rates(steps)
        x_0 = (noisy_images - noise * noise_rates) / signal_rates
        return x_0
    
    def transform_inputs(self, x, steps):
        return x, steps
    
    def get_posterior_mean(self, x_0, x_t, steps):
        raise NotImplementedError
    
    def get_posterior_variance(self, steps, shape=(-1, 1, 1, 1)):
        raise NotImplementedError

    def get_max_variance(self):
        alpha_n, sigma_n = self.get_rates(self.max_timesteps)
        variance = jnp.sqrt(alpha_n ** 2 + sigma_n ** 2) 
        return variance

class GeneralizedNoiseScheduler(NoiseScheduler):
    """
    As per the generalization presented in the paper
    "Elucidating the Design Space of Diffusion-Based
    Generative Models" by Tero Karras et al.
    Basically the signal rate shall always be 1, and the model
    input itself shall be scaled to match the noise rate
    """
    def __init__(self, timesteps, sigma_min=0.002, sigma_max=80.0, sigma_data=1, *args, **kwargs):
        super().__init__(timesteps, *args, **kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
    
    def get_weights(self, steps, shape=(-1, 1, 1, 1)):
        sigma = self.get_sigmas(steps)
        return (1 + (1 / (1 + ((1 - sigma ** 2)/(sigma ** 2)))) / (self.sigma_max ** 2)).reshape(shape)
    
    def get_sigmas(self, steps) -> jnp.ndarray:
        raise NotImplementedError("This method should be implemented in the subclass")
    
    def get_rates(self, steps, shape=(-1, 1, 1, 1)) -> tuple[jnp.ndarray, jnp.ndarray]:
        sigmas = self.get_sigmas(steps)
        signal_rates = 1
        noise_rates = sigmas
        return self.reshape_rates((signal_rates, noise_rates), shape=shape)
    
    def transform_inputs(self, x, steps, num_discrete_chunks=1000):
        sigmas_discrete = (steps / self.max_timesteps) * num_discrete_chunks
        sigmas_discrete = sigmas_discrete.astype(jnp.int32)
        return x, sigmas_discrete
    
    def get_timesteps(self, sigmas):
        """
        Inverse of the get_sigmas method
        """
        raise NotImplementedError("This method should be implemented in the subclass")