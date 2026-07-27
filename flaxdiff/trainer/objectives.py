"""What the trainer is optimizing.

The trainer owns training mechanics - sharding, EMA bookkeeping, checkpoints,
logging, the loops. An Objective owns what is being learned: the parameters it
holds, the loss it computes from a batch, the telemetry that loss reports, and
the artifacts validation scores. Swapping the objective swaps the research
question without touching any of the mechanics.

The loss always returns auxiliary metrics alongside the scalar, so an objective
with several loss terms or with per-step diagnostics (JEPA's collapse
telemetry) can surface them without the trainer knowing what they mean.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import optax

from ..schedulers import NoiseScheduler, get_coeff_shapes_tuple
from ..predictors import DiffusionPredictionTransform
from ..samplers.common import DiffusionSampler
from ..samplers.ddim import DDIMSampler
from ..models.autoencoder.autoencoder import AutoEncoder
from ..inputs import DiffusionInputConfig
from ..utils import RandomMarkovState


@dataclass
class EMASpec:
    """Which slice of the parameters the EMA copy tracks, and how fast.

    decay is a step-indexed schedule rather than a constant because momentum
    ramps are load-bearing for some objectives (I-JEPA anneals 0.996 -> 1.0).
    path selects a subtree; the empty path means the whole parameter tree.
    """
    decay: Callable[[Any], Any]
    path: Tuple[str, ...] = ()


class Objective(ABC):
    """The learning problem: parameters, loss, EMA policy, validation artifacts."""

    ema: EMASpec

    @abstractmethod
    def init_params(self, rng: jax.Array) -> Any:
        """Build the full parameter tree, which may hold several sub-modules."""

    @abstractmethod
    def loss(self, params: Any, ema_params: Any, batch: Any, rng: jax.Array,
             step: Any) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Scalar loss plus auxiliary metrics, differentiated w.r.t. params."""

    @abstractmethod
    def make_validation_step(self, **kwargs) -> Callable[[Any, Any], Any]:
        """Build (val_state, batch) -> artifacts, which eval metrics score."""

    def log_validation_artifacts(self, wandb, artifacts, step: int):
        """Visualize the artifacts. Nothing to draw unless an objective says so."""


class DiffusionObjective(Objective):
    """Denoising diffusion: sample a noise level, corrupt, predict, weight."""

    def __init__(
        self,
        model,
        noise_schedule: NoiseScheduler,
        model_output_transform: DiffusionPredictionTransform,
        input_config: DiffusionInputConfig,
        input_shapes: Dict[str, Tuple[int, ...]],
        autoencoder: AutoEncoder = None,
        unconditional_prob: float = 0.12,
        loss_fn: Callable = optax.l2_loss,
        ema_decay: float = 0.999,
        native_resolution: int = None,
        diffusion_steps: int = 200,
    ):
        self.model = model
        self.noise_schedule = noise_schedule
        self.model_output_transform = model_output_transform
        self.input_config = input_config
        self.input_shapes = input_shapes
        self.autoencoder = autoencoder
        self.unconditional_prob = unconditional_prob
        self.loss_fn = loss_fn
        self.native_resolution = native_resolution
        self.diffusion_steps = diffusion_steps
        self.ema = EMASpec(decay=optax.constant_schedule(ema_decay))

    def init_params(self, rng):
        input_ones = {k: jnp.ones((1, *v)) for k, v in self.input_shapes.items()}
        return self.model.init(rng, **input_ones)

    def loss(self, params, ema_params, batch, rng, step):
        local_rng_state = RandomMarkovState(rng)

        # Extract and normalize data (works for both images and videos)
        data = batch[self.input_config.sample_data_key]
        local_batch_size = data.shape[0]
        data = (jnp.asarray(data, dtype=jnp.float32) - 127.5) / 127.5

        if self.autoencoder is not None:
            local_rng_state, enc_key = local_rng_state.get_random_key()
            data = self.autoencoder.encode(data, enc_key)

        local_rng_state, uncond_key = local_rng_state.get_random_key()
        uncond_mask = jax.random.bernoulli(
            uncond_key, shape=(local_batch_size,), p=self.unconditional_prob)
        all_conditional_inputs = self.input_config.process_conditioning(
            batch, uncond_mask=uncond_mask)

        noise_level, local_rng_state = self.noise_schedule.generate_timesteps(
            local_batch_size, local_rng_state)

        local_rng_state, noise_key = local_rng_state.get_random_key()
        noise = jax.random.normal(noise_key, shape=data.shape, dtype=jnp.float32)

        local_rng_state, dropout_key = local_rng_state.get_random_key()

        rates = self.noise_schedule.get_rates(noise_level, get_coeff_shapes_tuple(data))
        noisy_data, c_in, expected_output = self.model_output_transform.forward_diffusion(
            data, noise, rates)

        inputs = self.noise_schedule.transform_inputs(noisy_data * c_in, noise_level)
        preds = self.model.apply(
            params, *inputs, *all_conditional_inputs,
            train=True, rngs={'dropout': dropout_key},
        )

        preds = self.model_output_transform.pred_transform(noisy_data, preds, rates)
        sample_losses = self.loss_fn(preds, expected_output)

        weights = self.noise_schedule.get_weights(
            noise_level, get_coeff_shapes_tuple(sample_losses))
        return jnp.mean(sample_losses * weights), {}

    def make_validation_step(
        self,
        sampler_class: Type[DiffusionSampler] = DDIMSampler,
        sampling_noise_schedule: NoiseScheduler = None,
    ):
        sampler = sampler_class(
            model=self.model,
            noise_schedule=self.noise_schedule if sampling_noise_schedule is None else sampling_noise_schedule,
            model_output_transform=self.model_output_transform,
            input_config=self.input_config,
            autoencoder=self.autoencoder,
            guidance_scale=3.0,
        )
        conditional_inputs = self.input_config.conditions
        image_size = self._image_size()
        sequence_length = self._sequence_length()

        def generate_samples(val_state, batch):
            model_conditioning_inputs = [cond_input(batch) for cond_input in conditional_inputs]
            batch_size = len(model_conditioning_inputs[0]) if model_conditioning_inputs else 4
            return sampler.generate_samples(
                params=val_state.ema_params,
                resolution=image_size,
                num_samples=batch_size,
                sequence_length=sequence_length,  # None for images
                diffusion_steps=self.diffusion_steps,
                end_step=0,
                priors=None,
                model_conditioning_inputs=tuple(model_conditioning_inputs),
            )

        return generate_samples

    def log_validation_artifacts(self, wandb, artifacts, step: int):
        import numpy as np

        is_video = len(artifacts.shape) == 5
        for i in range(artifacts.shape[0]):
            sample = np.clip((np.array(artifacts[i]) + 1) * 127.5, 0, 255).astype(np.uint8)
            if is_video:
                from wandb import Video as wandbVideo
                wandb.log({f"video_sample_{i}": wandbVideo(
                    sample, fps=10, caption=f"Video Sample {i} at step {step}")}, step=step)
            else:
                from wandb import Image as wandbImage
                wandb.log({f"sample_{i}": wandbImage(
                    sample, caption=f"Sample {i} at step {step}")}, step=step)

    def _is_video(self):
        return len(self.input_config.sample_data_shape) == 4

    def _image_size(self):
        if self.native_resolution is not None:
            return self.native_resolution
        return self.input_config.sample_data_shape[-2]

    def _sequence_length(self):
        return self.input_config.sample_data_shape[0] if self._is_video() else None
