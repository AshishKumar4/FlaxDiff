import math
import jax
import jax.numpy as jnp
from ..utils import RandomMarkovState
from .continuous import ContinuousNoiseScheduler
from .common import reshape_rates


def compute_resolution_shift(sequence_length, base_seq_len=256, max_seq_len=4096,
                             base_shift=0.5, max_shift=1.15) -> float:
    """Flux-style resolution dependent timestep shift.

    Longer token sequences carry more redundancy, so the trajectory has to
    spend more of its budget at high noise for the global structure to settle.
    mu is interpolated linearly in sequence length and the shift is exp(mu).
    """
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    mu = base_shift + slope * (sequence_length - base_seq_len)
    return math.exp(mu)


class FlowMatchingScheduler(ContinuousNoiseScheduler):
    """Rectified flow / conditional flow matching on the linear path.

    x_t = (1 - t) * x_0 + t * epsilon for t in [0, 1], so alpha + sigma = 1 and
    the model input needs no scaling. Timesteps are drawn logit-normal as in
    SD3, which concentrates training on the middle of the trajectory where the
    velocity is hardest to predict.
    """
    def __init__(self, shift: float = 1.0, logit_mean: float = 0.0, logit_std: float = 1.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift = shift
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def shift_timesteps(self, steps) -> jnp.ndarray:
        return self.shift * steps / (1 + (self.shift - 1) * steps)

    def generate_timesteps(self, batch_size, state: RandomMarkovState) -> tuple[jnp.ndarray, RandomMarkovState]:
        state, rng = state.get_random_key()
        normal = jax.random.normal(rng, (batch_size,), dtype=self.dtype)
        return jax.nn.sigmoid(normal * self.logit_std + self.logit_mean), state

    def get_rates(self, steps, shape=(-1, 1, 1, 1)) -> tuple[jnp.ndarray, jnp.ndarray]:
        t = self.shift_timesteps(jnp.asarray(steps, dtype=self.dtype))
        return reshape_rates((1 - t, t), shape=shape)

    def get_schedule_weights(self, steps, shape=(-1, 1, 1, 1)) -> jnp.ndarray:
        return jnp.ones_like(jnp.asarray(steps, dtype=self.dtype)).reshape(shape)

    def transform_inputs(self, x, steps) -> tuple[jnp.ndarray, jnp.ndarray]:
        # The Fourier time embedding is tuned for discrete-style timesteps, so
        # the [0, 1] flow time is rescaled into that range
        return x, self.shift_timesteps(jnp.asarray(steps, dtype=self.dtype)) * 1000
