import jax
import jax.numpy as jnp
from .common import DiffusionSampler
from ..utils import RandomMarkovState, MarkovState
from ..schedulers import GeneralizedNoiseScheduler, get_coeff_shapes_tuple

class RK4Sampler(DiffusionSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert issubclass(type(self.noise_schedule), GeneralizedNoiseScheduler), "Noise schedule must be a GeneralizedNoiseScheduler"
        # Not jitted here: sample_model_fn is a python callable, the model call
        # inside it is already jitted
        def get_derivative(sample_model_fn, x_t, sigma, state:RandomMarkovState, model_conditioning_inputs) -> tuple[jnp.ndarray, RandomMarkovState]:
            t = self.noise_schedule.get_timesteps(sigma)
            x_0, eps, _ = sample_model_fn(x_t, t, *model_conditioning_inputs)
            return eps, state
        
        self.get_derivative = get_derivative

    def sample_step(self, sample_model_fn, current_samples:jnp.ndarray, current_step, model_conditioning_inputs, next_step=None, state:MarkovState=None) -> tuple[jnp.ndarray, MarkovState]:
        step_ones = jnp.ones((current_samples.shape[0], ), dtype=jnp.float32)
        current_step = step_ones * current_step
        next_step = step_ones * next_step
        shape = get_coeff_shapes_tuple(current_samples)
        _, current_sigma = self.noise_schedule.get_rates(current_step, shape)
        _, next_sigma = self.noise_schedule.get_rates(next_step, shape)

        dt = next_sigma - current_sigma

        k1, state = self.get_derivative(sample_model_fn, current_samples, current_sigma, state, model_conditioning_inputs)
        k2, state = self.get_derivative(sample_model_fn, current_samples + 0.5 * k1 * dt, current_sigma + 0.5 * dt, state, model_conditioning_inputs)
        k3, state = self.get_derivative(sample_model_fn, current_samples + 0.5 * k2 * dt, current_sigma + 0.5 * dt, state, model_conditioning_inputs)
        k4, state = self.get_derivative(sample_model_fn, current_samples + k3 * dt, current_sigma + dt, state, model_conditioning_inputs)

        next_samples = current_samples + (((k1 + 2 * k2 + 2 * k3 + k4) * dt) / 6)
        return next_samples, state