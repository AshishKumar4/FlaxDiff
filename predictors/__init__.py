from typing import Union
import jax.numpy as jnp
from ..schedulers import NoiseScheduler

############################################################################################################
# Prediction Transforms
############################################################################################################

class DiffusionPredictionTransform():
    def __call__(self, x_t, preds, current_step, noise_schedule:NoiseScheduler) -> Union[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError
    
    def get_target(self, x_0, epsilon, current_step, rates) ->jnp.ndarray:
        raise NotImplementedError
    
    def train_step(self, x_0, epsilon, current_step, noise_schedule:NoiseScheduler) -> Union[jnp.ndarray, jnp.ndarray]:
        signal_rate, noise_rate = noise_schedule.get_rates(current_step)
        x_t = signal_rate * x_0 + noise_rate * epsilon
        expected_output = self.get_target(x_0, epsilon, current_step, (signal_rate, noise_rate))
        return x_t, expected_output
    
class EpsilonPredictionTransform(DiffusionPredictionTransform):
    def __call__(self, x_t, preds, current_step, noise_schedule:NoiseScheduler) -> Union[jnp.ndarray, jnp.ndarray]:
        # preds is the predicted noise
        epsilon = preds
        x_0 = noise_schedule.remove_all_noise(x_t, epsilon, current_step)
        return x_0, epsilon
    
    def get_target(self, x_0, epsilon, current_step, rates) ->jnp.ndarray:
        return epsilon
    
class VPredictionTransform(DiffusionPredictionTransform):
    def __call__(self, x_t, preds, current_step, noise_schedule:NoiseScheduler) -> Union[jnp.ndarray, jnp.ndarray]:
        # here the model output's V = sqrt_alpha_t * epsilon - sqrt_one_minus_alpha_t * x_0
        # where epsilon is the noise
        # x_0 is the current sample
        v = preds
        signal_rate, noise_rate = noise_schedule.get_rates(current_step)
        x_0 = signal_rate * x_t - noise_rate * v
        eps_0 = signal_rate * v + noise_rate * x_t
        return x_0, eps_0
    
    def get_target(self, x_0, epsilon, current_step, rates) ->jnp.ndarray:
        signal_rate, noise_rate = rates
        return signal_rate * epsilon - noise_rate * x_0