import flax
from flax import linen as nn
import jax
from typing import Callable
from dataclasses import field
import jax.numpy as jnp
import optax
import functools
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from typing import Dict, Callable, Sequence, Any, Union, Tuple

from ..schedulers import NoiseScheduler
from ..predictors import DiffusionPredictionTransform, EpsilonPredictionTransform

from flaxdiff.utils import RandomMarkovState

from .simple_trainer import SimpleTrainer, SimpleTrainState, Metrics

from flaxdiff.models.autoencoder.autoencoder import AutoEncoder
from flax.training import dynamic_scale as dynamic_scale_lib

class TrainState(SimpleTrainState):
    rngs: jax.random.PRNGKey
    ema_params: dict

    def apply_ema(self, decay: float = 0.999):
        new_ema_params = jax.tree_util.tree_map(
            lambda ema, param: decay * ema + (1 - decay) * param,
            self.ema_params,
            self.params,
        )
        return self.replace(ema_params=new_ema_params)

from flaxdiff.models.autoencoder.autoencoder import AutoEncoder
from flaxdiff.trainer.diffusion_trainer import DiffusionTrainer

class SimpleVideoDiffusionTrainer(DiffusionTrainer):
    def __init__(self,
                 model: nn.Module,
                 input_shapes: Dict[str, Tuple[int]],
                 optimizer: optax.GradientTransformation,
                 noise_schedule: NoiseScheduler,
                 rngs: jax.random.PRNGKey,
                 unconditional_prob: float = 0.12,
                 name: str = "SimpleVideoDiffusion",
                 model_output_transform: DiffusionPredictionTransform = EpsilonPredictionTransform(),
                 autoencoder: AutoEncoder = None,
                 **kwargs
                 ):
        super().__init__(
            model=model,
            input_shapes=input_shapes,
            optimizer=optimizer,
            noise_schedule=noise_schedule,
            unconditional_prob=unconditional_prob,
            autoencoder=autoencoder,
            model_output_transform=model_output_transform,
            rngs=rngs,
            name=name,
            **kwargs
        )
