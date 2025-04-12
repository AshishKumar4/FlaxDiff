import flax
from flax import linen as nn
import jax
from typing import Callable, List, Dict, Tuple, Union, Any, Sequence, Type, Optional
from dataclasses import field, dataclass
import jax.numpy as jnp
import optax
import functools
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from ..schedulers import NoiseScheduler, get_coeff_shapes_tuple
from ..predictors import DiffusionPredictionTransform, EpsilonPredictionTransform
from ..samplers.common import DiffusionSampler
from ..samplers.ddim import DDIMSampler

from flaxdiff.utils import RandomMarkovState
from flaxdiff.inputs import ConditioningEncoder, ConditionalInputConfig, DiffusionInputConfig

from .simple_trainer import SimpleTrainer, SimpleTrainState, Metrics

from flaxdiff.models.autoencoder.autoencoder import AutoEncoder
from flax.training import dynamic_scale as dynamic_scale_lib

# Reuse the TrainState from the DiffusionTrainer
from flaxdiff.trainer.diffusion_trainer import TrainState, DiffusionTrainer

class GeneralDiffusionTrainer(DiffusionTrainer):
    """
    General trainer for diffusion models supporting both images and videos.
    
    Extends DiffusionTrainer to support:
    1. Both image data (4D tensors: B,H,W,C) and video data (5D tensors: B,T,H,W,C)
    2. Multiple conditioning inputs
    3. Various model architectures
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: optax.GradientTransformation,
                 noise_schedule: NoiseScheduler,
                 input_config: DiffusionInputConfig,
                 rngs: jax.random.PRNGKey,
                 unconditional_prob: float = 0.12,
                 name: str = "GeneralDiffusion",
                 model_output_transform: DiffusionPredictionTransform = EpsilonPredictionTransform(),
                 autoencoder: AutoEncoder = None,
                 native_resolution: int = None,
                 frames_per_sample: int = None,
                 **kwargs
                 ):
        """
        Initialize the general diffusion trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimization algorithm
            noise_schedule: Noise scheduler for diffusion process
            input_config: Configuration for input data, inluding keys, shapes and conditioning inputs
            rngs: Random number generator keys
            unconditional_prob: Probability of training with unconditional samples
            name: Name of this trainer
            model_output_transform: Transform for model predictions
            autoencoder: Optional autoencoder for latent diffusion
            native_resolution: Native resolution of the data
            frames_per_sample: Number of frames per video sample (for video only)
            **kwargs: Additional arguments for parent class
        """
        # Initialize with parent DiffusionTrainer but without encoder parameter
        input_shapes = input_config.get_input_shapes(
            autoencoder=autoencoder,
        )
        self.input_config = input_config
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
            native_resolution=native_resolution,
            encoder=None,  # Don't use the default encoder from the parent class
            **kwargs
        )
        
        # Store video-specific parameters
        self.frames_per_sample = frames_per_sample
        
        # List of conditional inputs
        self.conditional_inputs = input_config.conditions
        
        # Cache for unconditional inputs
        self.unconditional_cache = {}
        
        # Precompute unconditional inputs for efficiency
        for cond_input in self.conditional_inputs:
            self.unconditional_cache[cond_input.conditioning_data_key] = cond_input.get_unconditional()
            
        # Determine if we're working with video or images
        self.is_video = self._is_video_data()
    
    def _is_video_data(self):
        sample_data_shape = self.input_config.sample_data_shape
        return len(sample_data_shape) == 5
        
    def _define_train_step(self, batch_size):
        """
        Define the training step function for both image and video diffusion.
        Optimized for efficient sharding and JIT compilation.
        """
        # Access class variables once for JIT optimization
        noise_schedule = self.noise_schedule
        model = self.model
        model_output_transform = self.model_output_transform
        loss_fn = self.loss_fn
        unconditional_prob = self.unconditional_prob
        num_unconditional = int(batch_size * unconditional_prob)
        conditional_inputs = self.conditional_inputs
        unconditional_cache = self.unconditional_cache
        distributed_training = self.distributed_training
        autoencoder = self.autoencoder
        sample_data_key = self.input_config.sample_data_key
        
        # JIT-optimized function for processing conditional inputs
        # @functools.partial(jax.jit, static_argnums=(2,))
        def process_conditioning(batch, uncond_cache, num_uncond):
            """Process conditioning inputs in a JIT-compatible way."""
            results = []
            
            for cond_input in conditional_inputs:
                # Apply encoder
                cond_embeddings = cond_input(batch)
                
                # Handle classifier-free guidance with unconditional samples
                if num_uncond > 0:
                    # Get cached unconditional embedding
                    uncond_embedding = uncond_cache[cond_input.encoder.key]
                    
                    # Ensure correct batch size via broadcasting
                    uncond_embedding = jax.lax.broadcast_in_dim(
                        uncond_embedding[:1],
                        (num_uncond,) + uncond_embedding.shape[1:],
                        (0,) + tuple(range(1, len(uncond_embedding.shape)))
                    )
                    
                    # Concatenate unconditional and conditional embeddings
                    cond_embeddings = jnp.concatenate(
                        [uncond_embedding, cond_embeddings[num_uncond:]], 
                        axis=0
                    )
                
                results.append(cond_embeddings)
                
            return results

        # Main training step function - optimized for JIT compilation and sharding
        def train_step(train_state: TrainState, rng_state: RandomMarkovState, batch, local_device_index):
            """Training step optimized for distributed execution."""
            # Random key handling
            rng_state, key_fold = rng_state.get_random_key()
            folded_key = jax.random.fold_in(key_fold, local_device_index.reshape())
            local_rng_state = RandomMarkovState(folded_key)
            
            # Extract and normalize data (works for both images and videos)
            data = batch[sample_data_key]
            local_batch_size = data.shape[0]
            data = (jnp.asarray(data, dtype=jnp.float32) - 127.5) / 127.5
            
            # Autoencoder step (handles both image and video data)
            if autoencoder is not None:
                local_rng_state, enc_key = local_rng_state.get_random_key()
                data = autoencoder.encode(data, enc_key)
            
            # Process conditioning
            all_conditional_inputs = process_conditioning(batch, unconditional_cache, num_unconditional)
            
            # Generate diffusion timesteps
            noise_level, local_rng_state = noise_schedule.generate_timesteps(local_batch_size, local_rng_state)
            
            # Generate noise
            local_rng_state, noise_key = local_rng_state.get_random_key()
            noise = jax.random.normal(noise_key, shape=data.shape, dtype=jnp.float32)
            
            # Forward diffusion process
            rates = noise_schedule.get_rates(noise_level, get_coeff_shapes_tuple(data))
            noisy_data, c_in, expected_output = model_output_transform.forward_diffusion(data, noise, rates)

            # Loss function
            def model_loss(params):
                # Apply model
                inputs = noise_schedule.transform_inputs(noisy_data * c_in, noise_level)
                preds = model.apply(params, *inputs, *all_conditional_inputs)
                
                # Transform predictions and calculate loss
                preds = model_output_transform.pred_transform(noisy_data, preds, rates)
                sample_losses = loss_fn(preds, expected_output)
                
                # Apply loss weighting
                weights = noise_schedule.get_weights(noise_level, get_coeff_shapes_tuple(sample_losses))
                weighted_loss = sample_losses * weights
                
                return jnp.mean(weighted_loss)
            
            # Compute gradients and apply updates
            if train_state.dynamic_scale is not None:
                # Mixed precision training with dynamic scale
                grad_fn = train_state.dynamic_scale.value_and_grad(model_loss, axis_name="data")
                dynamic_scale, is_finite, loss, grads = grad_fn(train_state.params)
                
                train_state = train_state.replace(dynamic_scale=dynamic_scale)
                new_state = train_state.apply_gradients(grads=grads)
                
                # Handle NaN/Inf gradients
                select_fn = functools.partial(jnp.where, is_finite)
                new_state = new_state.replace(
                    opt_state=jax.tree_map(select_fn, new_state.opt_state, train_state.opt_state),
                    params=jax.tree_map(select_fn, new_state.params, train_state.params)
                )
            else:
                # Standard gradient computation
                grad_fn = jax.value_and_grad(model_loss)
                loss, grads = grad_fn(train_state.params)
                
                if distributed_training:
                    grads = jax.lax.pmean(grads, axis_name="data")
                
                new_state = train_state.apply_gradients(grads=grads)
            
            # Apply EMA update
            new_state = new_state.apply_ema(self.ema_decay)
            
            # Average loss across devices if distributed
            if distributed_training:
                loss = jax.lax.pmean(loss, axis_name="data")
                
            return new_state, loss, rng_state

        # Apply sharding for distributed training
        if distributed_training:
            train_step = shard_map(
                train_step, 
                mesh=self.mesh, 
                in_specs=(P(), P(), P('data'), P('data')), 
                out_specs=(P(), P(), P()),
            )
            
        # Apply JIT compilation
        return jax.jit(train_step, donate_argnums=(2))

    def _define_validation_step(self, sampler_class: Type[DiffusionSampler]=DDIMSampler, sampling_noise_schedule: NoiseScheduler=None):
        """
        Define the validation step for both image and video diffusion models.
        """
        # Setup for validation
        model = self.model
        autoencoder = self.autoencoder
        conditional_inputs = self.conditional_inputs
        is_video = self.is_video
        
        # Get necessary parameters
        image_size = self._get_image_size()
        null_labels = self._get_null_labels()
        
        # Get sequence length only for video data
        sequence_length = self._get_sequence_length() if is_video else None
        
        # Initialize the sampler
        sampler = sampler_class(
            model=model,
            params=None,
            noise_schedule=self.noise_schedule if sampling_noise_schedule is None else sampling_noise_schedule,
            model_output_transform=self.model_output_transform,
            image_size=image_size,
            null_labels_seq=null_labels,
            autoencoder=autoencoder,
            guidance_scale=3.0,
        )
        
        def generate_samples(
            val_state: TrainState,
            batch,
            sampler: DiffusionSampler, 
            diffusion_steps: int,
        ):
            # Process all conditional inputs
            model_conditioning_inputs = [cond_input(batch) for cond_input in conditional_inputs]
            
            # Determine batch size
            batch_size = len(model_conditioning_inputs[0]) if model_conditioning_inputs else 4
            
            # Generate samples - works for both images and videos
            return sampler.generate_samples(
                params=val_state.ema_params,
                batch_size=batch_size,
                sequence_length=sequence_length,  # Will be None for images
                diffusion_steps=diffusion_steps,
                start_step=1000,
                end_step=0,
                priors=None,
                model_conditioning_inputs=tuple(model_conditioning_inputs),
            )
        
        return sampler, generate_samples
        
    def _get_image_size(self):
        """Helper to determine image size from available information."""
        if self.native_resolution is not None:
            return self.native_resolution
            
        sample_data_shape = self.input_config.sample_data_shape
        return sample_data_shape[-2] # Assuming [..., H, W, C] format
    
    def _get_sequence_length(self):
        """Helper to determine sequence length for video generation."""
        if not self.is_video:
            return None
            
        sample_data_shape = self.input_config.sample_data_shape
        return sample_data_shape[1]  # Assuming [B,T,H,W,C] format
        
    def _get_null_labels(self):
        """Helper to get unconditional inputs for classifier-free guidance."""
        if self.conditional_inputs:
            return self.conditional_inputs[0].get_unconditional()
        
        # Fallback if no conditional inputs are provided
        return jnp.zeros((1, 1, 1), dtype=jnp.float16)

    def validation_loop(
        self,
        val_state: SimpleTrainState,
        val_step_fn: Callable,
        val_ds,
        val_steps_per_epoch,
        current_step,
        diffusion_steps=200,
    ):
        """
        Run validation and log samples for both image and video diffusion.
        """
        sampler, generate_samples = val_step_fn
        val_ds = iter(val_ds()) if val_ds else None
        
        try:
            # Generate samples
            samples = generate_samples(
                val_state,
                next(val_ds),
                sampler,
                diffusion_steps,
            )
            
            # Log samples to wandb
            if getattr(self, 'wandb', None) is not None and self.wandb:
                import numpy as np
                
                # Process samples differently based on dimensionality
                if len(samples.shape) == 5:  # [B,T,H,W,C] - Video data
                    self._log_video_samples(samples, current_step)
                else:  # [B,H,W,C] - Image data
                    self._log_image_samples(samples, current_step)
                    
        except Exception as e:
            print("Error in validation loop:", e)
            import traceback
            traceback.print_exc()
            
    def _log_video_samples(self, samples, current_step):
        """Helper to log video samples to wandb."""
        import numpy as np
        from wandb import Video as wandbVideo
        
        for i in range(samples.shape[0]):
            # Convert to numpy, denormalize and clip
            sample = np.array(samples[i])
            sample = (sample + 1) * 127.5
            sample = np.clip(sample, 0, 255).astype(np.uint8)
            
            # Log as video
            self.wandb.log({
                f"video_sample_{i}": wandbVideo(
                    sample, 
                    fps=10, 
                    caption=f"Video Sample {i} at step {current_step}"
                )
            }, step=current_step)
            
    def _log_image_samples(self, samples, current_step):
        """Helper to log image samples to wandb."""
        import numpy as np
        from wandb import Image as wandbImage
        
        for i in range(samples.shape[0]):
            # Convert to numpy, denormalize and clip
            sample = np.array(samples[i])
            sample = (sample + 1) * 127.5
            sample = np.clip(sample, 0, 255).astype(np.uint8)
            
            # Log as image
            self.wandb.log({
                f"sample_{i}": wandbImage(
                    sample, 
                    caption=f"Sample {i} at step {current_step}"
                )
            }, step=current_step)