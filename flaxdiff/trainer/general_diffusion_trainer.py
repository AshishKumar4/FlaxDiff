import json
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

from flaxdiff.utils import RandomMarkovState, serialize_model, get_latest_checkpoint
from flaxdiff.inputs import ConditioningEncoder, ConditionalInputConfig, DiffusionInputConfig

from .simple_trainer import SimpleTrainer, SimpleTrainState, Metrics, convert_to_global_tree

from flaxdiff.models.autoencoder.autoencoder import AutoEncoder
from flax.training import dynamic_scale as dynamic_scale_lib

# Reuse the TrainState from the DiffusionTrainer
from .diffusion_trainer import TrainState, DiffusionTrainer
import shutil

from flaxdiff.metrics.common import EvaluationMetric

def generate_modelname(
    dataset_name: str,
    noise_schedule_name: str,
    architecture_name: str,
    model: nn.Module,
    input_config: DiffusionInputConfig,
    autoencoder: AutoEncoder = None,
    frames_per_sample: int = None,
) -> str:
    """
    Generate a model name based on the configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        A string representing the model name.
    """
    import hashlib
    import json
    
    # Extract key components for the name
    
    model_name = f"diffusion-{dataset_name}-res{input_config.sample_data_shape[-2]}"
    
    # model_name = f"diffusion-{dataset_name}-res{input_config.sample_data_shape[-2]}-{noise_schedule_name}-{architecture_name}"
    
    # if autoencoder is not None:
    #     model_name += f"-vae"
        
    # if frames_per_sample is not None:
    #     model_name += f"-frames_{frames_per_sample}"
    
    # model_name += f"-{'.'.join([cond.encoder.key for cond in input_config.conditions])}"
    
    # # Create a sorted representation of model config for consistent hashing
    # def sort_dict_recursively(d):
    #     if isinstance(d, dict):
    #         return {k: sort_dict_recursively(d[k]) for k in sorted(d.keys())}
    #     elif isinstance(d, list):
    #         return [sort_dict_recursively(v) for v in d]
    #     else:
    #         return d
    
    # # Extract model config and sort it
    # model_config = serialize_model(model)
    # sorted_model_config = sort_dict_recursively(model_config)
    
    # # Convert to JSON string with sorted keys for consistent hash
    # try:
    #     config_json = json.dumps(sorted_model_config)
    # except TypeError:
    #     # Handle non-serializable objects
    #     def make_serializable(obj):
    #         if isinstance(obj, dict):
    #             return {k: make_serializable(v) for k, v in obj.items()}
    #         elif isinstance(obj, list):
    #             return [make_serializable(v) for v in obj]
    #         else:
    #             try:
    #                 # Test if object is JSON serializable
    #                 json.dumps(obj)
    #                 return obj
    #             except TypeError:
    #                 return str(obj)
        
    #     serializable_config = make_serializable(sorted_model_config)
    #     config_json = json.dumps(serializable_config)
    
    # # Generate a hash of the configuration
    # config_hash = hashlib.md5(config_json.encode('utf-8')).hexdigest()[:8]
    
    # # Construct the model name
    # model_name = f"{model_name}-{config_hash}"
    return model_name

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
                 wandb_config: Dict[str, Any] = None,
                 eval_metrics: List[EvaluationMetric] = None,
                 **kwargs
                 ):
        """
        Initialize the general diffusion trainer.
        
        Args:
            model: Neural network model
            optimizer: Optimization algorithm
            noise_schedule: Noise scheduler for diffusion process
            input_config: Configuration for input data, including keys, shapes and conditioning inputs
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
        self.eval_metrics = eval_metrics
        
        if wandb_config is not None:
            # If input_config is not in wandb_config, add it
            if 'input_config' not in wandb_config['config']:
                wandb_config['config']['input_config'] = input_config.serialize()
            # If model is not in wandb_config, add it
            if 'model' not in wandb_config['config']:
                wandb_config['config']['model'] = serialize_model(model)
            if 'autoencoder' not in wandb_config['config'] and autoencoder is not None:
                wandb_config['config']['autoencoder'] = autoencoder.name
                wandb_config['config']['autoencoder_opts'] = json.dumps(autoencoder.serialize())
                
            # Generate a model name based on the configuration
            modelname = generate_modelname(
                dataset_name=wandb_config['config']['arguments']['dataset'],
                noise_schedule_name=wandb_config['config']['arguments']['noise_schedule'],
                architecture_name=wandb_config['config']['arguments']['architecture'],
                model=model,
                input_config=input_config,
                autoencoder=autoencoder,
                frames_per_sample=frames_per_sample,
            )
            print("Model name:", modelname)
            self.modelname = modelname
            wandb_config['config']['modelname'] = modelname
        
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
            wandb_config=wandb_config,
            **kwargs
        )
        
        # Store video-specific parameters
        self.frames_per_sample = frames_per_sample
        
        # List of conditional inputs
        self.conditional_inputs = input_config.conditions
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
        distributed_training = self.distributed_training
        autoencoder = self.autoencoder
        unconditional_prob = self.unconditional_prob
        
        input_config = self.input_config
        sample_data_key = input_config.sample_data_key
        
        # JIT-optimized function for processing conditional inputs
        # @functools.partial(jax.jit, static_argnums=(2,))
        def process_conditioning(batch, uncond_mask):
            return input_config.process_conditioning(
                batch,
                uncond_mask=uncond_mask,
            )

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
            
            # Determine number of unconditional samples per mini batch randomly
            local_rng_state, uncond_key = local_rng_state.get_random_key()
            # Determine unconditional samples 
            uncond_mask = jax.random.bernoulli(
                uncond_key,
                shape=(local_batch_size,),
                p=unconditional_prob
            )
            
            # Process conditioning
            all_conditional_inputs = process_conditioning(batch, uncond_mask)
            
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
        train_step = jax.jit(train_step, donate_argnums=(2))
        return train_step

    def _define_validation_step(self, sampler_class: Type[DiffusionSampler]=DDIMSampler, sampling_noise_schedule: NoiseScheduler=None):
        """
        Define the validation step for both image and video diffusion models.
        """
        # Setup for validation
        model = self.model
        autoencoder = self.autoencoder
        input_config = self.input_config
        conditional_inputs = self.conditional_inputs
        is_video = self.is_video
        
        # Get necessary parameters
        image_size = self._get_image_size()
        
        # Get sequence length only for video data
        sequence_length = self._get_sequence_length() if is_video else None
        
        # Initialize the sampler
        sampler = sampler_class(
            model=model,
            noise_schedule=self.noise_schedule if sampling_noise_schedule is None else sampling_noise_schedule,
            model_output_transform=self.model_output_transform,
            input_config=input_config,
            autoencoder=autoencoder,
            guidance_scale=3.0,
        )
        
        def generate_samples(
            val_state: TrainState,
            batch,
            diffusion_steps: int,
        ):
            # Process all conditional inputs
            model_conditioning_inputs = [cond_input(batch) for cond_input in conditional_inputs]
            
            # Determine batch size
            batch_size = len(model_conditioning_inputs[0]) if model_conditioning_inputs else 4
            
            # Generate samples - works for both images and videos
            return sampler.generate_samples(
                params=val_state.ema_params,
                resolution=image_size,
                num_samples=batch_size,
                sequence_length=sequence_length,  # Will be None for images
                diffusion_steps=diffusion_steps,
                start_step=1000,
                end_step=0,
                priors=None,
                model_conditioning_inputs=tuple(model_conditioning_inputs),
            )
        
        return generate_samples
        
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
        global_device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        process_index = jax.process_index()
        generate_samples = val_step_fn
        
        val_ds = iter(val_ds()) if val_ds else None
        # Evaluation step
        try:
            metrics = {metric.name: [] for metric in self.eval_metrics} if self.eval_metrics else {}
            for i in range(val_steps_per_epoch):
                if val_ds is None:
                    batch = None
                else:
                    batch = next(val_ds)
                    if self.distributed_training and global_device_count > 1:
                        batch = convert_to_global_tree(self.mesh, batch)
                # Generate samples
                samples = generate_samples(
                    val_state,
                    batch,
                    diffusion_steps,
                )
                
                if self.eval_metrics is not None:
                    for metric in self.eval_metrics:
                        try:
                            # Evaluate metrics
                            metric_val = metric.function(samples, batch)
                            metrics[metric.name].append(metric_val)
                        except Exception as e:
                            print("Error in evaluation metrics:", e)
                            import traceback
                            traceback.print_exc()
                            pass
                    
                if i == 0:
                    print(f"Evaluation started for process index {process_index}")
                    # Log samples to wandb
                    if getattr(self, 'wandb', None) is not None and self.wandb:
                        import numpy as np
                        
                        # Process samples differently based on dimensionality
                        if len(samples.shape) == 5:  # [B,T,H,W,C] - Video data
                            self._log_video_samples(samples, current_step)
                        else:  # [B,H,W,C] - Image data
                            self._log_image_samples(samples, current_step)
                    
            if getattr(self, 'wandb', None) is not None and self.wandb:
                # metrics is a dict of metrics
                if metrics and type(metrics) == dict:
                    # Flatten the metrics
                    metrics = {k: np.mean(v) for k, v in metrics.items()}
                    # Log the metrics
                    for key, value in metrics.items():
                        if isinstance(value, jnp.ndarray):
                            value = np.array(value)
                        self.wandb.log({
                            f"val/{key}": value,
                        }, step=current_step)
                        
        except StopIteration:
            print(f"Validation dataset exhausted for process index {process_index}")
        except Exception as e:
            print(f"Error during validation for process index {process_index}: {e}")
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
            
    def push_to_registry(
        self,
        registry_name: str = 'wandb-registry-model',
        aliases: List[str] = [],
    ):
        """
        Push the model to wandb registry.
        Args:
            registry_name: Name of the model registry.
            aliases: List of aliases for the model.
        """
        if self.wandb is None:
            raise ValueError("Wandb is not initialized. Cannot push to registry.")
        
        modelname = self.modelname
        if hasattr(self, "wandb_sweep"):
            modelname = f"{modelname}-sweep-{self.wandb_sweep.id}"
        
        latest_checkpoint_path = get_latest_checkpoint(self.checkpoint_path())
        logged_artifact = self.wandb.log_artifact(
            artifact_or_path=latest_checkpoint_path,
            name=modelname,
            type="model",
            aliases=['latest'] + aliases,
        )
        
        target_path = f"{registry_name}/{modelname}"
        
        self.wandb.link_artifact(
            artifact=logged_artifact,
            target_path=target_path,
            aliases=aliases,
        )
        print(f"Model pushed to registry at {target_path}")
        return logged_artifact
    
    def __get_best_sweep_runs__(
        self,
        metric: str = "train/best_loss",
        top_k: int = 5,
    ):
        """
        Get the best runs from a wandb sweep.
        Args:
            metric: Metric to sort by.
            top_k: Number of top runs to return.
        """
        if self.wandb is None:
            raise ValueError("Wandb is not initialized. Cannot get best runs.")
        
        if not hasattr(self, "wandb_sweep"):
            raise ValueError("Wandb sweep is not initialized. Cannot get best runs.")
        
        print(f"Getting best runs from sweep {self.wandb_sweep.id}...")
        # Get the sweep runs
        runs = sorted(self.wandb_sweep.runs, key=lambda x: x.summary.get(metric, float('inf')))
        best_runs = runs[:top_k]
        lower_bound = best_runs[-1].summary.get(metric, float('inf'))
        upper_bound = best_runs[0].summary.get(metric, float('inf'))
        print(f"Best runs from sweep {self.wandb_sweep.id}:")
        for run in best_runs:
            print(f"\t\tRun ID: {run.id}, Metric: {run.summary.get(metric, float('inf'))}")
        return best_runs, (min(lower_bound, upper_bound), max(lower_bound, upper_bound))
    
    def __get_best_general_runs__(
        self,
        metric: str = "train/best_loss",
        top_k: int = 5,
    ):
        """
        Get the best runs from wandb.
        Args:
            metric: Metric to sort by.
            top_k: Number of top runs to return.
        """
        if self.wandb is None:
            raise ValueError("Wandb is not initialized. Cannot get best runs.")
        
        # Get the sweep runs
        runs = sorted(self.wandb.runs, key=lambda x: x.summary.get(metric, float('inf')))
        best_runs = runs[:top_k]
        lower_bound = best_runs[-1].summary.get(metric, float('inf'))
        upper_bound = best_runs[0].summary.get(metric, float('inf'))
        print(f"Best runs from wandb {self.wandb.id}:")
        for run in best_runs:
            print(f"\t\tRun ID: {run.id}, Metric: {run.summary.get(metric, float('inf'))}")
        return best_runs, (min(lower_bound, upper_bound), max(lower_bound, upper_bound))
    
    def __compare_run_against_best__(self, top_k=2, metric="train/best_loss", from_sweeps=False):
        """
        Compare the current run against the best runs from wandb.
        Args:
            top_k: Number of top runs to consider.
            metric: Metric to compare against.
            from_sweeps: Whether to consider runs from sweeps.
        Returns:
            is_good: Whether the current run is among the best.
            is_best: Whether the current run is the best.
        """
        # Get best runs
        if from_sweeps:
            best_runs, bounds = self.__get_best_sweep_runs__(metric=metric, top_k=top_k)
        else:
            best_runs, bounds = self.__get_best_general_runs__(metric=metric, top_k=top_k)
        
        # Determine if lower or higher values are better (for loss, lower is better)
        is_lower_better = "loss" in metric.lower()
        
        # Check if current run is one of the best
        if metric == "train/best_loss":
            current_run_metric = self.best_loss
        else:
            current_run_metric = self.wandb.summary.get(metric, float('inf') if is_lower_better else float('-inf'))
                
        # Check based on bounds
        if (is_lower_better and current_run_metric < bounds[1]) or (not is_lower_better and current_run_metric > bounds[0]):
            print(f"Current run {self.wandb.id} meets performance criteria.")
            is_best = (is_lower_better and current_run_metric < bounds[0]) or (not is_lower_better and current_run_metric > bounds[1])
            return True, is_best
            
        return False, False
            
    def save(self, epoch=0, step=0, state=None, rngstate=None):
        super().save(epoch=epoch, step=step, state=state, rngstate=rngstate)
        
        if self.wandb is not None:
            checkpoint = get_latest_checkpoint(self.checkpoint_path())
            try:
                is_good, is_best = self.__compare_run_against_best__(top_k=5, metric="train/best_loss", from_sweeps=hasattr(self, "wandb_sweep"))
                if is_good:
                    # Push to registry with appropriate aliases
                    aliases = []
                    if is_best:
                        aliases.append("best")
                    self.push_to_registry(aliases=aliases)
                    print("Model pushed to registry successfully with aliases:", aliases)
                else:
                    print("Current run is not one of the best runs. Not saving model.")
                
                # Only delete after successful registry push
                shutil.rmtree(checkpoint, ignore_errors=True)
                print(f"Checkpoint deleted at {checkpoint}")
            except Exception as e:
                print(f"Error during registry operations: {e}")
                print(f"Checkpoint preserved at {checkpoint}")
