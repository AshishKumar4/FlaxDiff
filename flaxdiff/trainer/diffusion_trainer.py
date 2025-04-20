import flax
from flax import linen as nn
import jax
from typing import Callable
from dataclasses import field
import jax.numpy as jnp
import traceback
import optax
import functools
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from typing import Dict, Callable, Sequence, Any, Union, Tuple, Type

from ..schedulers import NoiseScheduler, get_coeff_shapes_tuple
from ..predictors import DiffusionPredictionTransform, EpsilonPredictionTransform
from ..samplers.common import DiffusionSampler
from ..samplers.ddim import DDIMSampler

from flaxdiff.utils import RandomMarkovState

from .simple_trainer import SimpleTrainer, SimpleTrainState, Metrics

from flaxdiff.models.autoencoder.autoencoder import AutoEncoder
from flax.training import dynamic_scale as dynamic_scale_lib
from flaxdiff.inputs import TextEncoder, ConditioningEncoder

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

class DiffusionTrainer(SimpleTrainer):
    noise_schedule: NoiseScheduler
    model_output_transform: DiffusionPredictionTransform
    ema_decay: float = 0.999
    native_resolution: int = None

    def __init__(self,
                 model: nn.Module,
                 input_shapes: Dict[str, Tuple[int]],
                 optimizer: optax.GradientTransformation,
                 noise_schedule: NoiseScheduler,
                 rngs: jax.random.PRNGKey,
                 unconditional_prob: float = 0.12,
                 name: str = "Diffusion",
                 model_output_transform: DiffusionPredictionTransform = EpsilonPredictionTransform(),
                 autoencoder: AutoEncoder = None,
                 encoder: ConditioningEncoder = None,
                 native_resolution: int = None,
                 **kwargs
                 ):
        super().__init__(
            model=model,
            input_shapes=input_shapes,
            optimizer=optimizer,
            rngs=rngs,
            name=name,
            **kwargs
        )
        self.noise_schedule = noise_schedule
        self.model_output_transform = model_output_transform
        self.unconditional_prob = unconditional_prob
        
        if native_resolution is None:
            if 'image' in input_shapes:
                native_resolution = input_shapes['image'][1]
            elif 'x' in input_shapes:
                native_resolution = input_shapes['x'][1]
            elif 'sample' in input_shapes:
                native_resolution = input_shapes['sample'][1]
            else:
                raise ValueError("No image input shape found in input shapes")
            if autoencoder is not None:
                native_resolution = native_resolution * 8
                
        self.native_resolution = native_resolution
        
        self.autoencoder = autoencoder
        self.encoder = encoder

    def generate_states(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        existing_state: dict = None,
        existing_best_state: dict = None,
        model: nn.Module = None,
        param_transforms: Callable = None,
        use_dynamic_scale: bool = False
    ) -> Tuple[TrainState, TrainState]:
        print("Generating states for DiffusionTrainer")
        rngs, subkey = jax.random.split(rngs)

        if existing_state == None:
            input_vars = self.get_input_ones()
            params = model.init(subkey, **input_vars)
            new_state = {"params": params, "ema_params": params}
        else:
            new_state = existing_state

        if param_transforms is not None:
            params = param_transforms(params)

        state = TrainState.create(
            apply_fn=model.apply,
            params=new_state['params'],
            ema_params=new_state['ema_params'],
            tx=optimizer,
            rngs=rngs,
            metrics=Metrics.empty(),
            dynamic_scale = dynamic_scale_lib.DynamicScale() if use_dynamic_scale else None
        )
            
        if existing_best_state is not None:
            best_state = state.replace(
                params=existing_best_state['params'], ema_params=existing_best_state['ema_params'])
        else:
            best_state = state

        return state, best_state

    def _define_train_step(self, batch_size):
        noise_schedule: NoiseScheduler = self.noise_schedule
        model = self.model
        model_output_transform = self.model_output_transform
        loss_fn = self.loss_fn
        unconditional_prob = self.unconditional_prob
        
        null_labels_full = self.encoder([""])
        null_labels_seq = jnp.array(null_labels_full[0], dtype=jnp.float16)
        
        conditioning_encoder = self.encoder

        nS, nC = null_labels_seq.shape
        null_labels_seq = jnp.broadcast_to(
            null_labels_seq, (batch_size, nS, nC))

        distributed_training = self.distributed_training
        
        autoencoder = self.autoencoder

        # @jax.jit
        def train_step(train_state: TrainState, rng_state: RandomMarkovState, batch, local_device_index):
            """Train for a single step."""
            rng_state, subkey = rng_state.get_random_key()
            subkey = jax.random.fold_in(subkey, local_device_index.reshape())
            local_rng_state = RandomMarkovState(subkey)
            
            images = batch['image']
            
            local_batch_size = images.shape[0]
            
            # First get the standard deviation of the images
            # std = jnp.std(images, axis=(1, 2, 3))
            # is_non_zero = (std > 0)
            
            images = jnp.array(images, dtype=jnp.float32)
            # normalize image
            images = (images - 127.5) / 127.5
            
            if autoencoder is not None:
                # Convert the images to latent space
                local_rng_state, rngs = local_rng_state.get_random_key()
                images = autoencoder.encode(images, rngs)

            label_seq = conditioning_encoder.encode_from_tokens(batch['text'])

            # Generate random probabilities to decide how much of this batch will be unconditional
            local_rng_state, uncond_key = local_rng_state.get_random_key()
            # Efficient way to determine unconditional samples for JIT compatibility
            uncond_mask = jax.random.bernoulli(
                uncond_key,
                shape=(local_batch_size,),
                p=unconditional_prob
            )
            num_unconditional = jnp.sum(uncond_mask).astype(jnp.int32)

            label_seq = jnp.concatenate([null_labels_seq[:num_unconditional], label_seq[num_unconditional:]], axis=0)

            noise_level, local_rng_state = noise_schedule.generate_timesteps(local_batch_size, local_rng_state)
            
            local_rng_state, rngs = local_rng_state.get_random_key()
            noise: jax.Array = jax.random.normal(rngs, shape=images.shape, dtype=jnp.float32)
            
            # Make sure image is also float32
            images = images.astype(jnp.float32)
            
            rates = noise_schedule.get_rates(noise_level, get_coeff_shapes_tuple(images))
            noisy_images, c_in, expected_output = model_output_transform.forward_diffusion(images, noise, rates)

            def model_loss(params):
                preds = model.apply(params, *noise_schedule.transform_inputs(noisy_images*c_in, noise_level), label_seq)
                preds = model_output_transform.pred_transform(noisy_images, preds, rates)
                nloss = loss_fn(preds, expected_output)
                # Ignore the loss contribution of images with zero standard deviation
                nloss *= noise_schedule.get_weights(noise_level, get_coeff_shapes_tuple(nloss))
                nloss = jnp.mean(nloss)
                loss = nloss
                return loss
            
            
            if train_state.dynamic_scale is not None:
                # dynamic scale takes care of averaging gradients across replicas
                grad_fn = train_state.dynamic_scale.value_and_grad(
                    model_loss, axis_name="data"
                )
                dynamic_scale, is_fin, loss, grads = grad_fn(train_state.params)
                train_state = train_state.replace(dynamic_scale=dynamic_scale)
            else:
                grad_fn = jax.value_and_grad(model_loss)
                loss, grads = grad_fn(train_state.params)
                if distributed_training:
                    grads = jax.lax.pmean(grads, "data")
    
            new_state = train_state.apply_gradients(grads=grads)
            
            if train_state.dynamic_scale is not None:
                # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
                # params should be restored (= skip this step).
                select_fn = functools.partial(jnp.where, is_fin)
                new_state = new_state.replace(
                    opt_state=jax.tree_util.tree_map(
                        select_fn, new_state.opt_state, train_state.opt_state
                    ),
                    params=jax.tree_util.tree_map(
                        select_fn, new_state.params, train_state.params
                    ),
                )
    
            new_state = new_state.apply_ema(self.ema_decay)
            
            if distributed_training:
                loss = jax.lax.pmean(loss, "data")
            return new_state, loss, rng_state

        if distributed_training:
            train_step = shard_map(
                train_step, 
                mesh=self.mesh, 
                in_specs=(P(), P(), P('data'), P('data')), 
                out_specs=(P(), P(), P()),
            )
        train_step = jax.jit(
            train_step,
            donate_argnums=(2)
        )
            
        return train_step

    def _define_validation_step(self, sampler_class: Type[DiffusionSampler]=DDIMSampler, sampling_noise_schedule: NoiseScheduler=None):
        model = self.model
        encoder = self.encoder
        autoencoder = self.autoencoder
        
        null_labels_full = encoder([""])
        null_labels_full = null_labels_full.astype(jnp.float16)
        # null_labels_seq = jnp.array(null_labels_full[0], dtype=jnp.float16)
        
        if self.native_resolution is not None:
            image_size = self.native_resolution
        elif 'image' in self.input_shapes:
            image_size = self.input_shapes['image'][1]
        elif 'x' in self.input_shapes:
            image_size = self.input_shapes['x'][1]
        elif 'sample' in self.input_shapes:
            image_size = self.input_shapes['sample'][1]
        else:
            raise ValueError("No image input shape found in input shapes")
        
        sampler = sampler_class(
            model=model,
            noise_schedule=self.noise_schedule if sampling_noise_schedule is None else sampling_noise_schedule,
            model_output_transform=self.model_output_transform,
            null_labels_seq=null_labels_full,
            autoencoder=autoencoder,
            guidance_scale=3.0,
        )
        
        def generate_samples(
            val_state: TrainState,
            batch,
            sampler: DiffusionSampler, 
            diffusion_steps: int,
        ):
            labels_seq = encoder.encode_from_tokens(batch)
            labels_seq = jnp.array(labels_seq, dtype=jnp.float16)
            samples = sampler.generate_images(
                params=val_state.ema_params,
                resolution=image_size,
                num_samples=len(labels_seq),
                diffusion_steps=diffusion_steps,
                start_step=1000,
                end_step=0,
                priors=None,
                model_conditioning_inputs=(labels_seq,),
            )
            return samples
        
        return sampler, generate_samples

    def validation_loop(
        self,
        val_state: SimpleTrainState,
        val_step_fn: Callable,
        val_ds,
        val_steps_per_epoch,
        current_step,
        diffusion_steps=200,
    ):
        sampler, generate_samples = val_step_fn
        
        # sampler = generate_sampler(val_state)
        
        val_ds = iter(val_ds()) if val_ds else None
        # Evaluation step
        try:
            samples = generate_samples(
                val_state,
                next(val_ds),
                sampler,
                diffusion_steps,
            )
            
            # Put each sample on wandb
            if getattr(self, 'wandb', None) is not None and self.wandb:
                import numpy as np
                from wandb import Image as wandbImage
                wandb_images = []
                for i in range(samples.shape[0]):
                    # convert the sample to numpy
                    sample = np.array(samples[i])
                    # denormalize the image
                    sample = (sample + 1) * 127.5
                    sample = np.clip(sample, 0, 255).astype(np.uint8)
                    # add the image to the list
                    wandb_images.append(sample)
                    # log the images to wandb
                    self.wandb.log({
                        f"sample_{i}": wandbImage(sample, caption=f"Sample {i} at step {current_step}")
                    }, step=current_step)
        except Exception as e:
            print("Error logging images to wandb", e)
            traceback.print_exc()
    
    def fit(self, data, training_steps_per_epoch, epochs, val_steps_per_epoch=8, sampler_class: Type[DiffusionSampler]=DDIMSampler, sampling_noise_schedule: NoiseScheduler=None):
        local_batch_size = data['local_batch_size']
        validation_step_args = {
            "sampler_class": sampler_class,
            "sampling_noise_schedule": sampling_noise_schedule,
        }
        super().fit(
            data, 
            train_steps_per_epoch=training_steps_per_epoch, 
            epochs=epochs, 
            train_step_args={"batch_size": local_batch_size}, 
            val_steps_per_epoch=val_steps_per_epoch,
            validation_step_args=validation_step_args,
        )
