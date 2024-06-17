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

    def generate_timesteps(self, batch_size, state:RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        raise NotImplementedError
    
    def get_p2_weights(self, k, gamma):
        raise NotImplementedError
    
    def clip_images(self, images) -> jnp.ndarray:
        raise NotImplementedError
    
    def reshape_rates(self, rates:tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError
    
    def get_rates(self, steps):
        raise NotImplementedError
    
    def add_noise(self, images, noise, steps, rates=None) -> jnp.ndarray:
        raise NotImplementedError
    
    def remove_all_noise(self, noisy_images, noise, steps, clip_denoised=True, rates=None):
        raise NotImplementedError
    
    def transform_steps(self, steps):
        raise NotImplementedError
    
    def get_posterior_mean(self, x_0, x_t, steps):
        raise NotImplementedError
    
    def get_posterior_variance(self, steps):
        raise NotImplementedError
    

class DiscreteNoiseScheduler(NoiseScheduler):
    def __init__(self, timesteps,
                    beta_start=0.0001,
                    beta_end=0.02,
                    schedule_fn=None, 
                    *args, **kwargs):
        super().__init__(timesteps, *args, **kwargs)
        betas = schedule_fn(timesteps, beta_start, beta_end)
        alphas = 1 - betas
        alpha_cumprod = jnp.cumprod(alphas, axis=0)
        alpha_cumprod_prev = jnp.append(1.0, alpha_cumprod[:-1])
        
        self.betas = jnp.array(betas, dtype=jnp.float32)
        self.alphas = alphas.astype(jnp.float32)
        self.alpha_cumprod = alpha_cumprod.astype(jnp.float32)
        self.alpha_cumprod_prev = alpha_cumprod_prev.astype(jnp.float32)

        self.sqrt_alpha_cumprod = jnp.sqrt(alpha_cumprod).astype(jnp.float32)
        self.sqrt_one_minus_alpha_cumprod = jnp.sqrt(1 - alpha_cumprod).astype(jnp.float32)

        self.sqrt_recip_alpha_cumprod = jnp.sqrt(1 / alpha_cumprod).astype(jnp.float32)
        self.sqrt_recip_one_minus_alpha_cumprod = jnp.sqrt(1 / alpha_cumprod - 1).astype(jnp.float32)

        posterior_variance = (betas * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))
        self.posterior_variance = posterior_variance.astype(jnp.float32)
        self.posterior_log_variance_clipped = (jnp.log(jnp.maximum(posterior_variance, 1e-20))).astype(jnp.float32)
        
        self.posterior_mean_coef1 = (betas * jnp.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod)).astype(jnp.float32)
        self.posterior_mean_coef2 = ((1 - alpha_cumprod_prev) * jnp.sqrt(alphas) / (1 - alpha_cumprod)).astype(jnp.float32)

    def generate_timesteps(self, batch_size, state:RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        timesteps = jax.random.randint(rng, (batch_size,), 0, self.max_timesteps)
        return timesteps, state
    
    def get_p2_weights(self, k, gamma):
        return (k + self.alpha_cumprod / (1 - self.alpha_cumprod)) ** -gamma
    
    def clip_images(self, images) -> jnp.ndarray:
        return jnp.clip(images, self.clip_min, self.clip_max)
    
    def reshape_rates(self, rates:tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        signal_rates, noise_rates = rates
        signal_rates = jnp.reshape(signal_rates, (-1, 1, 1, 1))
        noise_rates = jnp.reshape(noise_rates, (-1, 1, 1, 1))
        return signal_rates, noise_rates

    def get_rates(self, steps):
        signal_rate = self.sqrt_alpha_cumprod[steps]
        noise_rate = self.sqrt_one_minus_alpha_cumprod[steps]
        signal_rate = jnp.reshape(signal_rate, (-1, 1, 1, 1))
        noise_rate = jnp.reshape(noise_rate, (-1, 1, 1, 1))
        return signal_rate, noise_rate
    
    # Used while training
    def add_noise(self, images, noise, steps) -> jnp.ndarray:
        signal_rates, noise_rates = self.get_rates(steps)
        return signal_rates * images + noise_rates * noise

    def remove_all_noise(self, noisy_images, noise, steps):
        # Scale 't' to the range [0, 1]
        signal_coeff = self.sqrt_recip_alpha_cumprod[steps]
        noise_coeff = self.sqrt_recip_one_minus_alpha_cumprod[steps]
        signal_coeff, noise_coeff = self.reshape_rates((signal_coeff, noise_coeff))
        pred_images = signal_coeff * noisy_images - noise_coeff * noise
        return pred_images
    
    # Used while training
    def transform_steps(self, steps):
        return steps #/ self.max_timesteps
    
    def get_posterior_mean(self, x_0, x_t, steps):
        x_0_coeff = self.posterior_mean_coef1[steps]
        x_t_coeff = self.posterior_mean_coef2[steps]
        x_0_coeff, x_t_coeff = self.reshape_rates((x_0_coeff, x_t_coeff))
        mean = x_0_coeff * x_0 + x_t_coeff * x_t
        return mean
    
    def get_posterior_variance(self, steps):
        return jnp.exp(0.5 * self.posterior_log_variance_clipped[steps]).reshape(-1, 1, 1, 1)
