from flax import linen as nn
import jax
from typing import Callable
from dataclasses import field
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from typing import Dict, Callable, Sequence, Any, Union, Tuple

from ..schedulers import NoiseScheduler
from ..predictors import DiffusionPredictionTransform, EpsilonPredictionTransform

from flaxdiff.utils import RandomMarkovState

from .simple_trainer import SimpleTrainer, SimpleTrainState, Metrics

from flaxdiff.models.autoencoder.autoencoder import AutoEncoder

class AutoEncoderTrainer(SimpleTrainer):
    def __init__(self,
                 model: nn.Module,
                 input_shape: Union[int, int, int],
                 latent_dim: int,
                 spatial_scale: int,
                 optimizer: optax.GradientTransformation,
                 rngs: jax.random.PRNGKey,
                 name: str = "Autoencoder",
                 **kwargs
                 ):
        super().__init__(
            model=model,
            input_shapes={"image": input_shape},
            optimizer=optimizer,
            rngs=rngs,
            name=name,
            **kwargs
        )
        self.latent_dim = latent_dim
        self.spatial_scale = spatial_scale
        

    def generate_states(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        existing_state: dict = None,
        existing_best_state: dict = None,
        model: nn.Module = None,
        param_transforms: Callable = None
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
            metrics=Metrics.empty()
        )
            
        if existing_best_state is not None:
            best_state = state.replace(
                params=existing_best_state['params'], ema_params=existing_best_state['ema_params'])
        else:
            best_state = state

        return state, best_state

    def _define_train_step(self, batch_size, null_labels_seq, text_embedder):
        noise_schedule: NoiseScheduler = self.noise_schedule
        model = self.model
        model_output_transform = self.model_output_transform
        loss_fn = self.loss_fn
        unconditional_prob = self.unconditional_prob

        # Determine the number of unconditional samples
        num_unconditional = int(batch_size * unconditional_prob)

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
            
            if autoencoder is not None:
                # Convert the images to latent space
                local_rng_state, rngs = local_rng_state.get_random_key()
                images = autoencoder.encode(images, rngs)
            else:
                # normalize image
                images = (images - 127.5) / 127.5

            output = text_embedder(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            label_seq = output.last_hidden_state

            # Generate random probabilities to decide how much of this batch will be unconditional

            label_seq = jnp.concat(
                [null_labels_seq[:num_unconditional], label_seq[num_unconditional:]], axis=0)

            noise_level, local_rng_state = noise_schedule.generate_timesteps(images.shape[0], local_rng_state)
            
            local_rng_state, rngs = local_rng_state.get_random_key()
            noise: jax.Array = jax.random.normal(rngs, shape=images.shape)
            
            rates = noise_schedule.get_rates(noise_level)
            noisy_images, c_in, expected_output = model_output_transform.forward_diffusion(
                images, noise, rates)

            def model_loss(params):
                preds = model.apply(params, *noise_schedule.transform_inputs(noisy_images*c_in, noise_level), label_seq)
                preds = model_output_transform.pred_transform(
                    noisy_images, preds, rates)
                nloss = loss_fn(preds, expected_output)
                # nloss = jnp.mean(nloss, axis=1)
                nloss *= noise_schedule.get_weights(noise_level)
                nloss = jnp.mean(nloss)
                loss = nloss
                return loss
            
            loss, grads = jax.value_and_grad(model_loss)(train_state.params)
            if distributed_training:
                grads = jax.lax.pmean(grads, "data")
                loss = jax.lax.pmean(loss, "data")
            train_state = train_state.apply_gradients(grads=grads)
            train_state = train_state.apply_ema(self.ema_decay)
            return train_state, loss, rng_state

        if distributed_training:
            train_step = shard_map(train_step, mesh=self.mesh, in_specs=(P(), P(), P('data'), P('data')), 
                                   out_specs=(P(), P(), P()))
            train_step = jax.jit(train_step)
            
        return train_step

    def _define_compute_metrics(self):
        @jax.jit
        def compute_metrics(state: TrainState, expected, pred):
            loss = jnp.mean(jnp.square(pred - expected))
            metric_updates = state.metrics.single_from_model_output(loss=loss)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        return compute_metrics

    def fit(self, data, steps_per_epoch, epochs):
        null_labels_full = data['null_labels_full']
        local_batch_size = data['local_batch_size']
        text_embedder = data['model']
        super().fit(data, steps_per_epoch, epochs, {
            "batch_size": local_batch_size, "null_labels_seq": null_labels_full, "text_embedder": text_embedder})

def boolean_string(s):
    if type(s) == bool:
        return s
    return s == 'True'

