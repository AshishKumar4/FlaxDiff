from typing import Union, Type

import jax
import jax.numpy as jnp
import tqdm
from flax import linen as nn
from typing import List, Tuple, Dict, Any, Optional

from ..predictors import DiffusionPredictionTransform, EpsilonPredictionTransform
from ..schedulers import NoiseScheduler, get_coeff_shapes_tuple
from ..utils import RandomMarkovState, MarkovState, clip_images
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from flaxdiff.models.autoencoder import AutoEncoder
from flaxdiff.inputs import DiffusionInputConfig

class DiffusionSampler:
    """Base class for diffusion samplers."""
    
    def __init__(
        self,
        model: nn.Module,
        noise_schedule: NoiseScheduler,
        model_output_transform: DiffusionPredictionTransform,
        input_config: DiffusionInputConfig,
        guidance_scale: float = 0.0,
        autoencoder: AutoEncoder = None,
    ):
        """Initialize the diffusion sampler.
        
        Args:
            model: Neural network model
            params: Model parameters
            noise_schedule: Noise scheduler
            model_output_transform: Transform for model predictions
            guidance_scale: Scale for classifier-free guidance (0.0 means disabled)
            autoencoder: Optional autoencoder for latent diffusion
        """
        self.model = model
        self.noise_schedule = noise_schedule
        self.model_output_transform = model_output_transform
        self.guidance_scale = guidance_scale
        self.autoencoder = autoencoder
        self.input_config = input_config
        
        unconditionals = input_config.get_unconditionals()
        
        if self.guidance_scale > 0:
            # Classifier free guidance
            print("Using classifier-free guidance")

            def sample_model(params, x_t, t, *conditioning_inputs):
                # Concatenate unconditional and conditional inputs
                x_t_cat = jnp.concatenate([x_t] * 2, axis=0)
                t_cat = jnp.concatenate([t] * 2, axis=0)
                rates_cat = self.noise_schedule.get_rates(t_cat, get_coeff_shapes_tuple(x_t_cat))
                c_in_cat = self.model_output_transform.get_input_scale(rates_cat)
                
                final_conditionals = []
                for conditional, unconditional in zip(conditioning_inputs, unconditionals):
                    final = jnp.concatenate([
                        conditional,
                        jnp.broadcast_to(unconditional, conditional.shape)
                    ], axis=0)
                    final_conditionals.append(final)
                final_conditionals = tuple(final_conditionals)
                
                model_output = self.model.apply(
                    params, 
                    *self.noise_schedule.transform_inputs(x_t_cat * c_in_cat, t_cat), 
                    *final_conditionals
                )
                
                # Split model output into unconditional and conditional parts
                model_output_cond, model_output_uncond = jnp.split(model_output, 2, axis=0)
                model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

                x_0, eps = self.model_output_transform(x_t, model_output, t, self.noise_schedule)
                return x_0, eps, model_output
        else:
            # Unconditional sampling
            def sample_model(params, x_t, t, *conditioning_inputs):
                rates = self.noise_schedule.get_rates(t, get_coeff_shapes_tuple(x_t))
                c_in = self.model_output_transform.get_input_scale(rates)
                model_output = self.model.apply(
                    params, 
                    *self.noise_schedule.transform_inputs(x_t * c_in, t), 
                    *conditioning_inputs
                )
                x_0, eps = self.model_output_transform(x_t, model_output, t, self.noise_schedule)
                return x_0, eps, model_output
            
        # JIT compile the sampling function for better performance
        def post_process(samples: jnp.ndarray):
            """Post-process the generated samples."""
            if autoencoder is not None:
                samples = autoencoder.decode(samples)
                
            samples = clip_images(samples)
            return samples
        
        self.sample_model = jax.jit(sample_model)
        self.post_process = jax.jit(post_process)

    def sample_step(
        self, 
        sample_model_fn, 
        current_samples: jnp.ndarray, 
        current_step, 
        model_conditioning_inputs, 
        next_step=None, 
        state: RandomMarkovState = None
    ) -> tuple[jnp.ndarray, RandomMarkovState]:
        """Perform a single sampling step in the diffusion process.
        
        Args:
            sample_model_fn: Function to sample from model
            current_samples: Current noisy samples
            current_step: Current diffusion timestep
            model_conditioning_inputs: Conditioning inputs for the model
            next_step: Next diffusion timestep
            state: Current Markov state
            
        Returns:
            Tuple of (new samples, updated state)
        """
        step_ones = jnp.ones((len(current_samples),), dtype=jnp.float32)
        current_step = step_ones * current_step
        next_step = step_ones * next_step
        
        pred_images, pred_noise, _ = sample_model_fn(
            current_samples, current_step, *model_conditioning_inputs
        )
        
        new_samples, state = self.take_next_step(
            current_samples=current_samples,
            reconstructed_samples=pred_images,
            pred_noise=pred_noise,
            current_step=current_step,
            next_step=next_step,
            state=state,
            model_conditioning_inputs=model_conditioning_inputs,
            sample_model_fn=sample_model_fn,
        )
        return new_samples, state


    def take_next_step(
        self,
        current_samples,
        reconstructed_samples,
        model_conditioning_inputs,
        pred_noise,
        current_step,
        state: RandomMarkovState,
        sample_model_fn,
        next_step=1,
    ) -> tuple[jnp.ndarray, RandomMarkovState]:
        """Take the next step in the diffusion process.
        
        This method needs to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement take_next_step method")


    def get_steps(self, start_step, end_step, diffusion_steps):
        """Get the descending sequence of timesteps for the diffusion process.

        Timesteps are in the noise schedule's own domain: [0, max_timesteps],
        so [0, 1] for the continuous schedules and [0, T] for the discrete ones.
        Note that for a Karras schedule, linear spacing in t is already the
        rho-spacing in sigma from the EDM paper.

        Args:
            start_step: Starting timestep (typically max_timesteps)
            end_step: Ending timestep (typically 0)
            diffusion_steps: Number of steps to use

        Returns:
            Array of timesteps for sampling
        """
        step_range = start_step - end_step
        if diffusion_steps is None or diffusion_steps == 0:
            diffusion_steps = int(step_range)
        if self.noise_schedule.max_timesteps > 1:
            # Discrete schedules cannot take more steps than they have entries
            diffusion_steps = min(diffusion_steps, int(step_range))

        return jnp.linspace(start_step, end_step, diffusion_steps, dtype=jnp.float32)


    def generate_samples(
        self,
        params: dict,
        num_samples: int,
        resolution: int,
        sequence_length: int = None,
        diffusion_steps: int = 1000,
        start_step: int = None,
        end_step: int = 0,
        steps_override=None,
        priors=None,
        rngstate: RandomMarkovState = None,
        conditioning: List[Union[Tuple, Dict]] = None,
        model_conditioning_inputs: Tuple = None,
    ) -> jnp.ndarray:
        """Generate samples using the diffusion model.
        
        Provides a unified interface for generating both images and videos.
        For images, just specify batch_size.
        For videos, specify both batch_size and sequence_length.
        
        Args:
            params: Model parameters (uses self.params if None)
            num_samples: Number of samples to generate (videos or images)
            resolution: Resolution of the generated samples (H, W)
            sequence_length: Length of each sequence (for videos/audio/etc)
                             If None, generates regular images
            diffusion_steps: Number of diffusion steps to perform
            start_step: Starting timestep (defaults to max)
            end_step: Ending timestep
            steps_override: Override default timestep sequence
            priors: Prior samples to start from instead of noise
            rngstate: Random state for reproducibility
            conditioning: (Optional) List of conditioning inputs for the model
            model_conditioning_inputs: (Optional) Pre-processed conditioning inputs
            
        Returns:
            Generated samples as a JAX array:
            - For images: shape [batch_size, H, W, C]
            - For videos: shape [batch_size, sequence_length, H, W, C]
        """
        if rngstate is None:
            rngstate = RandomMarkovState(jax.random.PRNGKey(42))
            
        if start_step is None:
            start_step = self.noise_schedule.max_timesteps
            
        if priors is None:
            # Determine if we're generating videos or images based on sequence_length
            is_video = sequence_length is not None
            
            rngstate, newrngs = rngstate.get_random_key()
            
            # Get sample shape based on whether we're generating video or images
            if is_video:
                samples = self._get_initial_sequence_samples(
                    resolution, num_samples, sequence_length, newrngs, start_step
                )
            else:
                samples = self._get_initial_samples(resolution, num_samples, newrngs, start_step)
        else:
            print("Using priors")
            if self.autoencoder is not None:
                # Let the autoencoder handle both image and video priors
                priors = self.autoencoder.encode(priors)
            samples = priors
        
        if conditioning is not None:
            if model_conditioning_inputs is not None:
                raise ValueError("Cannot provide both conditioning and model_conditioning_inputs")
            print("Processing raw conditioning inputs to generate model conditioning inputs")
            separated: Dict[str, List] = {}
            for cond in self.input_config.conditions:
                separated[cond.encoder.key] = []
            # Separate the conditioning inputs, one for each condition
            for vals in conditioning:
                if isinstance(vals, tuple) or isinstance(vals, list): 
                    # If its a tuple, assume that the ordering aligns with the ordering of the conditions
                    # Thus, use the conditioning encoder key as the key
                    for cond, val in zip(self.input_config.conditions, vals):
                        separated[cond.encoder.key].append(val)
                elif isinstance(vals, dict):
                    # If its a dict, use the encoder key as the key
                    for cond in self.input_config.conditions:
                        if cond.encoder.key in vals:
                            separated[cond.encoder.key].append(vals[cond.encoder.key])
                        else:
                            raise ValueError(f"Conditioning input {cond.encoder.key} not found in provided dictionary")
                else:
                    # If its a single value, use the encoder key as the key
                    for cond in self.input_config.conditions:
                        separated[cond.encoder.key].append(vals)
                        
            # Now we have a dictionary of lists, one for each condition, encode them
            finals = []
            for cond in self.input_config.conditions:
                # Get the encoder for the condition
                encoder = cond.encoder
                encoded = encoder(separated[encoder.key])
                finals.append(encoded)
                
            model_conditioning_inputs = tuple(finals)
        
        if model_conditioning_inputs is None:
            model_conditioning_inputs = []

        def sample_model_fn(x_t, t, *additional_inputs):
            return self.sample_model(params, x_t, t, *additional_inputs)

        def sample_step(sample_model_fn, state: RandomMarkovState, samples, current_step, next_step):
            samples, state = self.sample_step(
                sample_model_fn=sample_model_fn, 
                current_samples=samples,
                current_step=current_step,
                model_conditioning_inputs=model_conditioning_inputs,
                state=state, 
                next_step=next_step
            )
            return samples, state

        if start_step is None:
            start_step = self.noise_schedule.max_timesteps

        if steps_override is not None:
            steps = steps_override
        else:
            steps = self.get_steps(start_step, end_step, diffusion_steps)

        for i in tqdm.tqdm(range(0, len(steps))):
            current_step = steps[i]
            next_step = steps[i+1] if i+1 < len(steps) else end_step
            
            if i != len(steps) - 1:
                samples, rngstate = sample_step(
                    sample_model_fn, rngstate, samples, current_step, next_step
                )
            else:
                step_ones = jnp.ones((samples.shape[0],), dtype=jnp.int32)
                samples, _, _ = sample_model_fn(
                    samples, current_step * step_ones, *model_conditioning_inputs
                )
        return self.post_process(samples)
    
    def _get_noise_parameters(self, resolution, start_step):
        """Calculate common noise parameters for sample generation.
        
        Args:
            start_step: Starting timestep for noise generation
            
        Returns:
            Tuple of (variance, image_size, image_channels)
        """
        alpha_n, sigma_n = self.noise_schedule.get_rates(start_step)
        variance = jnp.sqrt(alpha_n ** 2 + sigma_n ** 2)
        
        image_size = resolution
        image_channels = 3
        if self.autoencoder is not None:
            image_size = image_size // self.autoencoder.downscale_factor
            image_channels = self.autoencoder.latent_channels
            
        return variance, image_size, image_channels
    
    def _get_initial_samples(self, resolution, batch_size, rngs: jax.random.PRNGKey, start_step):
        """Generate initial noisy samples for image generation."""
        variance, image_size, image_channels = self._get_noise_parameters(resolution, start_step)
        
        # Standard image generation
        return jax.random.normal(
            rngs, 
            (batch_size, image_size, image_size, image_channels)
        ) * variance
    
    def _get_initial_sequence_samples(self, resolution, batch_size, sequence_length, rngs: jax.random.PRNGKey, start_step):
        """Generate initial noisy samples for sequence data (video/audio)."""
        variance, image_size, image_channels = self._get_noise_parameters(resolution, start_step)
        
        # Generate sequence data (like video)
        return jax.random.normal(
            rngs, 
            (batch_size, sequence_length, image_size, image_size, image_channels)
        ) * variance
    
    # Alias for backward compatibility
    generate_images = generate_samples
