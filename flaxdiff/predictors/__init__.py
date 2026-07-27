from typing import Union
import jax.numpy as jnp
from ..schedulers import (
    NoiseScheduler,
    GeneralizedNoiseScheduler,
    get_coeff_shapes_tuple,
    CosineNoiseScheduler,
    KarrasVENoiseScheduler,
    EDMNoiseScheduler,
    FlowMatchingScheduler,
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

    def target_error_scale(self, snr: jnp.ndarray) -> jnp.ndarray:
        """||target error||^2 / ||x_0 error||^2 at the given SNR.

        Loss weights (min-SNR-gamma and friends) are defined on the x_0 loss;
        dividing by this converts them into the space the model is trained in.
        """
        return 1.0

class EpsilonPredictionTransform(DiffusionPredictionTransform):
    def backward_diffusion(self, x_t, preds, rates: tuple[jnp.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, jnp.ndarray]:
        # preds is the predicted noise
        epsilon = preds
        signal_rates, noise_rates = rates
        x_0 = (x_t - epsilon * noise_rates) / signal_rates
        return x_0, epsilon

    def get_target(self, x_0, epsilon, rates) ->jnp.ndarray:
        return epsilon

    def target_error_scale(self, snr: jnp.ndarray) -> jnp.ndarray:
        return snr

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

    def target_error_scale(self, snr: jnp.ndarray) -> jnp.ndarray:
        return snr + 1

class FlowMatchPredictionTransform(DiffusionPredictionTransform):
    """Rectified flow velocity: the model predicts u = epsilon - x_0, the
    constant velocity of the linear path, so both endpoints are one step away.
    """
    def backward_diffusion(self, x_t, preds, rates: tuple[jnp.ndarray, jnp.ndarray]) -> Union[jnp.ndarray, jnp.ndarray]:
        signal_rate, noise_rate = rates
        x_0 = x_t - noise_rate * preds
        epsilon = x_t + signal_rate * preds
        return x_0, epsilon

    def get_target(self, x_0, epsilon, rates) ->jnp.ndarray:
        return epsilon - x_0

    def target_error_scale(self, snr: jnp.ndarray) -> jnp.ndarray:
        # x_0 error is t times the velocity error, and t = 1 / (1 + sqrt(SNR))
        return (1 + jnp.sqrt(snr)) ** 2

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

    def target_error_scale(self, snr: jnp.ndarray) -> jnp.ndarray:
        # x_0 error is c_out times the raw error, and alpha = 1 here so
        # sigma^2 = 1 / SNR
        return 1 / self.sigma_data ** 2 + snr

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
    P_mean: float = -0.4,
    P_std: float = 1.0,
    shift: float = 1.0,
    min_snr_gamma: float = None,
) -> tuple[NoiseScheduler, NoiseScheduler, DiffusionPredictionTransform]:
    """Named (train schedule, sampling schedule, prediction transform) presets.

    The single source of truth for which schedule pairs with which
    parameterization. Both training and inference build from here, so a model
    is always sampled with the same convention it was trained with.
    """
    # Only the training schedule ever produces loss weights, so only it needs
    # to know the parameterization
    if name == 'edm':
        transform = KarrasPredictionTransform(sigma_data=sigma_data)
        train = EDMNoiseScheduler(1, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data,
                                  P_mean=P_mean, P_std=P_std,
                                  prediction_transform=transform, min_snr_gamma=min_snr_gamma)
        sample = KarrasVENoiseScheduler(1, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data)
    elif name == 'karras':
        transform = KarrasPredictionTransform(sigma_data=sigma_data)
        train = KarrasVENoiseScheduler(1, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data,
                                       prediction_transform=transform, min_snr_gamma=min_snr_gamma)
        sample = KarrasVENoiseScheduler(1, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, sigma_data=sigma_data)
    elif name == 'cosine':
        transform = VPredictionTransform()
        train = CosineNoiseScheduler(timesteps, beta_end=1,
                                     prediction_transform=transform, min_snr_gamma=min_snr_gamma)
        sample = CosineNoiseScheduler(timesteps, beta_end=1)
    elif name in ('flow', 'flow_matching'):
        transform = FlowMatchPredictionTransform()
        train = FlowMatchingScheduler(shift=shift,
                                      prediction_transform=transform, min_snr_gamma=min_snr_gamma)
        sample = FlowMatchingScheduler(shift=shift)
    else:
        raise ValueError(f"Unknown noise schedule preset: {name}")
    return train, sample, transform
