from typing import Union, Type

import jax
import jax.numpy as jnp
import tqdm
from flax import linen as nn

from ..predictors import DiffusionPredictionTransform, EpsilonPredictionTransform
from ..schedulers import NoiseScheduler
from ..utils import RandomMarkovState, MarkovState, clip_images
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P


class DiffusionSampler:
    """Base class for diffusion samplers."""
    
    def __init__(
        self,
        model: nn.Module,
        params: dict,
        noise_schedule: NoiseScheduler,
        model_output_transform: DiffusionPredictionTransform,
        guidance_scale: float = 0.0,
        null_labels_seq: jax.Array = None,
        autoencoder=None,
        image_size=256,
        autoenc_scale_reduction=8,
        autoenc_latent_channels=4,
        
        # device_mesh: jax.sharding.Mesh = None,
    ):
        """Initialize the diffusion sampler.
        
        Args:
            model: Neural network model
            params: Model parameters
            noise_schedule: Noise scheduler
            model_output_transform: Transform for model predictions
            guidance_scale: Scale for classifier-free guidance (0.0 means disabled)
            null_labels_seq: Unconditional sequence for guidance
            autoencoder: Optional autoencoder for latent diffusion
            image_size: Size of generated images
            autoenc_scale_reduction: Scale reduction factor for autoencoder
            autoenc_latent_channels: Number of channels in latent space
        """
        self.model = model
        self.noise_schedule = noise_schedule
        self.params = params
        self.model_output_transform = model_output_transform
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.autoenc_scale_reduction = autoenc_scale_reduction
        self.autoenc_latent_channels = autoenc_latent_channels
        self.autoencoder = autoencoder

        if self.guidance_scale > 0:
            # Classifier free guidance
            assert null_labels_seq is not None, "Null labels sequence is required for classifier-free guidance"
            print("Using classifier-free guidance")

            def sample_model(params, x_t, t, *additional_inputs):
                # Concatenate unconditional and conditional inputs
                x_t_cat = jnp.concatenate([x_t] * 2, axis=0)
                t_cat = jnp.concatenate([t] * 2, axis=0)
                rates_cat = self.noise_schedule.get_rates(t_cat)
                c_in_cat = self.model_output_transform.get_input_scale(rates_cat)

                text_labels_seq, = additional_inputs
                text_labels_seq = jnp.concatenate(
                    [text_labels_seq, jnp.broadcast_to(null_labels_seq, text_labels_seq.shape)], 
                    axis=0
                )
                model_output = self.model.apply(
                    params, 
                    *self.noise_schedule.transform_inputs(x_t_cat * c_in_cat, t_cat), 
                    text_labels_seq
                )
                
                # Split model output into unconditional and conditional parts
                model_output_cond, model_output_uncond = jnp.split(model_output, 2, axis=0)
                model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

                x_0, eps = self.model_output_transform(x_t, model_output, t, self.noise_schedule)
                return x_0, eps, model_output
        else:
            # Unconditional sampling
            def sample_model(params, x_t, t, *additional_inputs):
                rates = self.noise_schedule.get_rates(t)
                c_in = self.model_output_transform.get_input_scale(rates)
                model_output = self.model.apply(
                    params, 
                    *self.noise_schedule.transform_inputs(x_t * c_in, t), 
                    *additional_inputs
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
        state: MarkovState = None
    ) -> tuple[jnp.ndarray, MarkovState]:
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
        step_ones = jnp.ones((len(current_samples),), dtype=jnp.int32)
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
        return NotImplementedError


    def scale_steps(self, steps):
        """Scale timesteps to match the noise schedule's range."""
        scale_factor = self.noise_schedule.max_timesteps / 1000
        return steps * scale_factor


    def get_steps(self, start_step, end_step, diffusion_steps):
        """Get the sequence of timesteps for the diffusion process."""
        step_range = start_step - end_step
        if diffusion_steps is None or diffusion_steps == 0:
            diffusion_steps = start_step - end_step
        diffusion_steps = min(diffusion_steps, step_range)
        steps = jnp.linspace(
            end_step, start_step,
            diffusion_steps, dtype=jnp.int16
        )[::-1]
        return steps


    def get_initial_samples(self, num_images, rngs: jax.random.PRNGKey, start_step):
        """Generate initial noisy samples for the diffusion process."""
        start_step = self.scale_steps(start_step)
        alpha_n, sigma_n = self.noise_schedule.get_rates(start_step)
        variance = jnp.sqrt(alpha_n ** 2 + sigma_n ** 2)
        
        image_size = self.image_size
        image_channels = 3
        if self.autoencoder is not None:
            image_size = image_size // self.autoenc_scale_reduction
            image_channels = self.autoenc_latent_channels
            
        return jax.random.normal(rngs, (num_images, image_size, image_size, image_channels)) * variance


    def generate_images(
        self,
        params: dict = None,
        num_images=16,
        diffusion_steps=1000,
        start_step: int = None,
        end_step: int = 0,
        steps_override=None,
        priors=None,
        rngstate: RandomMarkovState = None,
        model_conditioning_inputs: tuple = ()
    ) -> jnp.ndarray:
        """Generate images using the diffusion model.
        
        Args:
            params: Model parameters (uses self.params if None)
            num_images: Number of images to generate
            diffusion_steps: Number of diffusion steps to perform
            start_step: Starting timestep (defaults to max)
            end_step: Ending timestep
            steps_override: Override default timestep sequence
            priors: Prior samples to start from instead of noise
            rngstate: Random state for reproducibility
            model_conditioning_inputs: Conditioning inputs for the model
            
        Returns:
            Generated images as a JAX array
        """
        if rngstate is None:
            rngstate = RandomMarkovState(jax.random.PRNGKey(42))
            
        if priors is None:
            rngstate, newrngs = rngstate.get_random_key()
            samples = self.get_initial_samples(num_images, newrngs, start_step)
        else:
            print("Using priors")
            if self.autoencoder is not None:
                priors = self.autoencoder.encode(priors)
            samples = priors

        params = params if params is not None else self.params

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

        # print("Sample shape", samples.shape)

        for i in tqdm.tqdm(range(0, len(steps))):
            current_step = self.scale_steps(steps[i])
            next_step = self.scale_steps(steps[i+1] if i+1 < len(steps) else 0)
            
            if i != len(steps) - 1:
                samples, rngstate = sample_step(
                    sample_model_fn, rngstate, samples, current_step, next_step
                )
            else:
                step_ones = jnp.ones((num_images,), dtype=jnp.int32)
                samples, _, _ = sample_model_fn(
                    samples, current_step * step_ones, *model_conditioning_inputs
                )
        return self.post_process(samples)
