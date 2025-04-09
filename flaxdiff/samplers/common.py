from flax import linen as nn
import jax
import jax.numpy as jnp
import tqdm
from typing import Union, Type
from ..schedulers import NoiseScheduler
from ..utils import RandomMarkovState, MarkovState, clip_images
from ..predictors import DiffusionPredictionTransform, EpsilonPredictionTransform

class DiffusionSampler():
    def __init__(self, model:nn.Module, params:dict,  
                 noise_schedule:NoiseScheduler, 
                 model_output_transform:DiffusionPredictionTransform,
                 guidance_scale:float = 0.0,
                 null_labels_seq:jax.Array=None,
                 autoencoder=None,
                 image_size=256,
                 autoenc_scale_reduction=8,
                 autoenc_latent_channels=4,
                 ):
        self.model = model
        self.noise_schedule = noise_schedule
        self.params = params
        self.model_output_transform = model_output_transform
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.autoenc_scale_reduction = autoenc_scale_reduction
        self.autoencoder = autoencoder
        self.autoenc_latent_channels = autoenc_latent_channels
        
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
                text_labels_seq = jnp.concatenate([text_labels_seq, jnp.broadcast_to(null_labels_seq, text_labels_seq.shape)], axis=0)
                model_output = self.model.apply(params, *self.noise_schedule.transform_inputs(x_t_cat * c_in_cat, t_cat), text_labels_seq)
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
                model_output = self.model.apply(params, *self.noise_schedule.transform_inputs(x_t * c_in, t), *additional_inputs)
                x_0, eps = self.model_output_transform(x_t, model_output, t, self.noise_schedule)
                return x_0, eps, model_output
            
        # if jax.device_count() > 1:
        #     mesh = jax.sharding.Mesh(jax.devices(), 'data')
        #     sample_model = shard_map(sample_model, mesh=mesh, in_specs=(P('data'), P('data'), P('data')),
        #                              out_specs=(P('data'), P('data'), P('data')))
        sample_model = jax.jit(sample_model)
        self.sample_model = sample_model

    # Used to sample from the diffusion model
    def sample_step(self, sample_model_fn, current_samples:jnp.ndarray, current_step, model_conditioning_inputs, next_step=None, state:MarkovState=None) -> tuple[jnp.ndarray, MarkovState]:
        # First clip the noisy images
        step_ones = jnp.ones((current_samples.shape[0], ), dtype=jnp.int32)
        current_step = step_ones * current_step
        next_step = step_ones * next_step
        pred_images, pred_noise, _ = sample_model_fn(current_samples, current_step, *model_conditioning_inputs)
        # plotImages(pred_images)
        # pred_images = clip_images(pred_images)
        new_samples, state =  self.take_next_step(current_samples=current_samples, reconstructed_samples=pred_images, 
                                pred_noise=pred_noise, current_step=current_step, next_step=next_step, state=state,
                                model_conditioning_inputs=model_conditioning_inputs,
                                sample_model_fn=sample_model_fn,
                             )
        return new_samples, state

    def take_next_step(self, current_samples, reconstructed_samples, model_conditioning_inputs,
                 pred_noise, current_step, state:RandomMarkovState, sample_model_fn, next_step=1,) -> tuple[jnp.ndarray, RandomMarkovState]:
        # estimate the q(x_{t-1} | x_t, x_0). 
        # pred_images is x_0, noisy_images is x_t, steps is t
        return NotImplementedError
    
    def scale_steps(self, steps):
        scale_factor = self.noise_schedule.max_timesteps / 1000
        return steps * scale_factor

    def get_steps(self, start_step, end_step, diffusion_steps):
        step_range = start_step - end_step
        if diffusion_steps is None or diffusion_steps == 0:
            diffusion_steps = start_step - end_step
        diffusion_steps = min(diffusion_steps, step_range)
        steps = jnp.linspace(end_step, start_step, diffusion_steps, dtype=jnp.int16)[::-1]
        return steps
    
    def get_initial_samples(self, num_images, rngs:jax.random.PRNGKey, start_step):
        start_step = self.scale_steps(start_step)
        alpha_n, sigma_n = self.noise_schedule.get_rates(start_step)
        variance = jnp.sqrt(alpha_n ** 2 + sigma_n ** 2) 
        image_size = self.image_size
        image_channels = 3
        if self.autoencoder is not None:
            image_size = image_size // self.autoenc_scale_reduction
            image_channels = self.autoenc_latent_channels
        return jax.random.normal(rngs, (num_images, image_size, image_size, image_channels)) * variance

    def generate_images(self,
                        params:dict=None,
                        num_images=16, 
                        diffusion_steps=1000, 
                        start_step:int = None,
                        end_step:int = 0,
                        steps_override=None,
                        priors=None, 
                        rngstate:RandomMarkovState=None,
                        model_conditioning_inputs:tuple=()
                        ) -> jnp.ndarray:
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

        # @jax.jit
        def sample_step(sample_model_fn, state:RandomMarkovState, samples, current_step, next_step):
            samples, state = self.sample_step(sample_model_fn=sample_model_fn, current_samples=samples,
                                              current_step=current_step, 
                                              model_conditioning_inputs=model_conditioning_inputs,
                                              state=state, next_step=next_step)
            return samples, state

        if start_step is None:
            start_step = self.noise_schedule.max_timesteps

        if steps_override is not None:
            steps = steps_override
        else:
            steps = self.get_steps(start_step, end_step, diffusion_steps)

        # print("Sampling steps", steps)
        for i in tqdm.tqdm(range(0, len(steps))):
            current_step = self.scale_steps(steps[i])
            next_step = self.scale_steps(steps[i+1] if i+1 < len(steps) else 0)
            if i != len(steps) - 1:
                # print("normal step")
                samples, rngstate = sample_step(sample_model_fn, rngstate, samples, current_step, next_step)
            else:
                # print("last step")
                step_ones = jnp.ones((num_images, ), dtype=jnp.int32)
                samples, _, _ = sample_model_fn(samples, current_step * step_ones, *model_conditioning_inputs)
        if self.autoencoder is not None:
            samples = self.autoencoder.decode(samples)
        samples = clip_images(samples)
        return samples
    