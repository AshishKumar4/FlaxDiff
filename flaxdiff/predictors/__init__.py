from typing import Union
import jax.numpy as jnp
from ..schedulers import (
    NoiseScheduler,
    GeneralizedNoiseScheduler,
    get_coeff_shapes_tuple,
    CosineNoiseScheduler,
    KarrasVENoiseScheduler,
    EDMNoiseScheduler,
)

############################################################################################################
# Prediction Transforms
############################################################################################################

class DiffusionPredictionTransform():
    def pred_transform(self, x_t, preds, rates) -> jnp.ndarray:
        return preds
    
    def __call__(self, x_t, preds, current_step, noise_schedule:NoiseScheduler) -> Union[jnp.ndarray, jnp.ndarray]:
        rates = noise_schedule.get_rates(current_step, shape=get_coeff_shapes_tuple(x_t))
        preds = self.pred_transform(x_t, preds, rates)
        x_0, epsilon = self.backward_diffusion(x_t, preds, rates)
        return x_0, epsilon
    
    def forward_diffusion(self, x_0, epsilon, rates: tuple[jnp.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        signal_rate, noise_rate = rates
        x_t = signal_rate * x_0 + noise_rate * epsilon
        expected_output = self.get_target(x_0, epsilon, (signal_rate, noise_rate))
        c_in = self.get_input_scale((signal_rate, noise_rate))
        return x_t, c_in, expected_output
    
    def backward_diffusion(self, x_t, preds, rates: tuple[jnp.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError
    
    def get_target(self, x_0, epsilon, rates) ->jnp.ndarray:
        return x_0
    
    def get_input_scale(self, rates: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        return 1

class EpsilonPredictionTransform(DiffusionPredictionTransform):
    def backward_diffusion(self, x_t, preds, rates: tuple[jnp.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, jnp.ndarray]:
        # preds is the predicted noise
        epsilon = preds
        signal_rates, noise_rates = rates
        x_0 = (x_t - epsilon * noise_rates) / signal_rates
        return x_0, epsilon
    
    def get_target(self, x_0, epsilon, rates) ->jnp.ndarray:
        return epsilon

class DirectPredictionTransform(DiffusionPredictionTransform):
    def backward_diffusion(self, x_t, preds, rates: tuple[jnp.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, jnp.ndarray]:
        # Here the model predicts x_0 directly
        x_0 = preds
        signal_rate, noise_rate = rates
        epsilon = (x_t - x_0 * signal_rate) / noise_rate
        return x_0, epsilon

class VPredictionTransform(DiffusionPredictionTransform):
    def backward_diffusion(self, x_t, preds, rates: tuple[jnp.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, jnp.ndarray]:
        # here the model output's V = sqrt_alpha_t * epsilon - sqrt_one_minus_alpha_t * x_0
        # where epsilon is the noise
        # x_0 is the current sample
        v = preds
        signal_rate, noise_rate = rates
        variance = signal_rate ** 2 + noise_rate ** 2
        v = v * jnp.sqrt(variance)
        x_0 = signal_rate * x_t - noise_rate * v
        eps_0 = signal_rate * v + noise_rate * x_t
        return x_0 / variance, eps_0 / variance
    
    def get_target(self, x_0, epsilon, rates) ->jnp.ndarray:
        signal_rate, noise_rate = rates
        v = signal_rate * epsilon - noise_rate * x_0
        variance = signal_rate**2 + noise_rate**2
        return v / jnp.sqrt(variance)
    
class KarrasPredictionTransform(DiffusionPredictionTransform):
    def __init__(self, sigma_data=0.5) -> None:
        super().__init__()
        self.sigma_data = sigma_data

    def backward_diffusion(self, x_t, preds, rates: tuple[jnp.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, jnp.ndarray]:
        x_0 = preds
        signal_rate, noise_rate = rates
        epsilon = (x_t - x_0 * signal_rate) / noise_rate
        return x_0, epsilon
    
    def pred_transform(self, x_t, preds, rates: tuple[jnp.ndarray, jnp.ndarray], epsilon=1e-8) -> jnp.ndarray:
        _, sigma = rates
        c_out = sigma * self.sigma_data / (jnp.sqrt(self.sigma_data ** 2 + sigma ** 2) + epsilon)
        c_skip = self.sigma_data ** 2 / (self.sigma_data ** 2 + sigma ** 2 + epsilon)
        c_out = c_out.reshape(get_coeff_shapes_tuple(preds))
        c_skip = c_skip.reshape(get_coeff_shapes_tuple(x_t))
        x_0 = c_out * preds + c_skip * x_t
        return x_0
    
    def get_input_scale(self, rates: tuple[jnp.ndarray, jnp.ndarray], epsilon=1e-8) -> jnp.ndarray:
        _, sigma = rates
        c_in = 1 / (jnp.sqrt(self.sigma_data ** 2 + sigma ** 2) + epsilon)
        return c_in

############################################################################################################
# Noise schedule presets
############################################################################################################

def get_diffusion_preset(
    name: str,
    timesteps: int = 1000,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    sigma_data: float = 0.5,
) -> tuple[NoiseScheduler, NoiseScheduler, DiffusionPredictionTransform]:
    """Named (train schedule, sampling schedule, prediction transform) presets.

    The single source of truth for which schedule pairs with which
    parameterization. Both training and inference build from here, so a model
    is always sampled with the same convention it was trained with.
    """
    if name == 'edm':
        train = EDMNoiseScheduler(1, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data)
        sample = KarrasVENoiseScheduler(1, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data)
        transform = KarrasPredictionTransform(sigma_data=sigma_data)
    elif name == 'karras':
        train = KarrasVENoiseScheduler(1, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data)
        sample = KarrasVENoiseScheduler(1, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data)
        transform = KarrasPredictionTransform(sigma_data=sigma_data)
    elif name == 'cosine':
        train = CosineNoiseScheduler(timesteps, beta_end=1)
        sample = CosineNoiseScheduler(timesteps, beta_end=1)
        transform = VPredictionTransform()
    else:
        raise ValueError(f"Unknown noise schedule preset: {name}")
    return train, sample, transform
