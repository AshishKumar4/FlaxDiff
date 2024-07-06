import math
import numpy as np
import jax.numpy as jnp
from .discrete import DiscreteNoiseScheduler
from .continuous import ContinuousNoiseScheduler
from .common import GeneralizedNoiseScheduler

def cosine_beta_schedule(timesteps, start_angle=0.008, end_angle=0.999):
    ts = np.linspace(0, 1, timesteps + 1, dtype=np.float64)
    alphas_bar = np.cos((ts + start_angle) / (1 + start_angle) * np.pi /2) ** 2
    alphas_bar = alphas_bar/alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0, end_angle)

class CosineNoiseSchedule(DiscreteNoiseScheduler):
    def __init__(self, timesteps, beta_start=0.008, beta_end=0.999, *args, **kwargs):
        super().__init__(timesteps, beta_start, beta_end, schedule_fn=cosine_beta_schedule, *args, **kwargs)

class CosineGeneralNoiseScheduler(GeneralizedNoiseScheduler):
    def __init__(self, sigma_min=0.02, sigma_max=80.0, kappa=1.0, *args, **kwargs):
        super().__init__(timesteps=1, sigma_min=sigma_min, sigma_max=sigma_max, *args, **kwargs)
        self.kappa = kappa
        logsnr_max = 2 * (math.log(self.kappa) - math.log(self.sigma_max))
        self.theta_max = math.atan(math.exp(-0.5 * logsnr_max))
        logsnr_min = 2 * (math.log(self.kappa) - math.log(self.sigma_min))
        self.theta_min = math.atan(math.exp(-0.5 * logsnr_min))
    
    def get_sigmas(self, steps):
        return jnp.tan(self.theta_min + steps * (self.theta_max - self.theta_min)) / self.kappa
    
class CosineContinuousNoiseScheduler(ContinuousNoiseScheduler):
    def get_rates(self, steps, shape=(-1, 1, 1, 1)) -> tuple[jnp.ndarray, jnp.ndarray]:
        signal_rates = jnp.cos((jnp.pi * steps) / (2 * self.max_timesteps))
        noise_rates = jnp.sin((jnp.pi * steps) / (2 * self.max_timesteps))
        return self.reshape_rates((signal_rates, noise_rates), shape=shape)
    
    def get_weights(self, steps):
        alpha, sigma = self.get_rates(steps, shape=())
        return 1 / (1 + (alpha ** 2 / sigma ** 2))
    