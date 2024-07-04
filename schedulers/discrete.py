import jax
import jax.numpy as jnp
from typing import Union
from ..utils import RandomMarkovState  
from .common import NoiseScheduler

class DiscreteNoiseScheduler(NoiseScheduler):
    """
    Variance Preserving Noise Scheduler
    signal_rate**2 + noise_rate**2 = 1
    """
    def __init__(self, timesteps,
                    beta_start=0.0001,
                    beta_end=0.02,
                    schedule_fn=None, 
                    p2_loss_weight_k:float=1,
                    p2_loss_weight_gamma:float=1,
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

        posterior_variance = (betas * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))
        self.posterior_variance = posterior_variance.astype(jnp.float32)
        self.posterior_log_variance_clipped = (jnp.log(jnp.maximum(posterior_variance, 1e-20))).astype(jnp.float32)
        
        self.posterior_mean_coef1 = (betas * jnp.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod)).astype(jnp.float32)
        self.posterior_mean_coef2 = ((1 - alpha_cumprod_prev) * jnp.sqrt(alphas) / (1 - alpha_cumprod)).astype(jnp.float32)

        self.p2_loss_weights = self.get_p2_weights(p2_loss_weight_k, p2_loss_weight_gamma)
    
    def generate_timesteps(self, batch_size, state:RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        timesteps = jax.random.randint(rng, (batch_size,), 0, self.max_timesteps)
        return timesteps, state
    
    def get_p2_weights(self, k, gamma):
        return (k + self.alpha_cumprod / (1 - self.alpha_cumprod)) ** -gamma
    
    def get_weights(self, steps, shape=(-1, 1, 1, 1)):
        steps = jnp.int16(steps)
        return self.p2_loss_weights[steps].reshape(shape)

    def get_rates(self, steps, shape=(-1, 1, 1, 1)) -> tuple[jnp.ndarray, jnp.ndarray]:
        steps = jnp.int16(steps)
        signal_rate = self.sqrt_alpha_cumprod[steps]
        noise_rate = self.sqrt_one_minus_alpha_cumprod[steps]
        signal_rate = jnp.reshape(signal_rate, shape)
        noise_rate = jnp.reshape(noise_rate, shape)
        return signal_rate, noise_rate
    
    def get_posterior_mean(self, x_0, x_t, steps):
        steps = jnp.int16(steps)
        x_0_coeff = self.posterior_mean_coef1[steps]
        x_t_coeff = self.posterior_mean_coef2[steps]
        x_0_coeff, x_t_coeff = self.reshape_rates((x_0_coeff, x_t_coeff))
        mean = x_0_coeff * x_0 + x_t_coeff * x_t
        return mean
    
    def get_posterior_variance(self, steps, shape=(-1, 1, 1, 1)):
        steps = int(steps)
        return jnp.exp(0.5 * self.posterior_log_variance_clipped[steps]).reshape(shape)

        