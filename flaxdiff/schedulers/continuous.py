import jax
import jax.numpy as jnp
from typing import Union
from ..utils import RandomMarkovState  
from .common import NoiseScheduler

class ContinuousNoiseScheduler(NoiseScheduler):
    """
    General Continuous Noise Scheduler
    """
    def __init__(self, *args, **kwargs):
        super().__init__(timesteps=1, *args, **kwargs)