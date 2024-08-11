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
from flax.training.dynamic_scale import DynamicScale

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
        
        self.autoencoder = autoencoder

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
            new_state['params'] = param_transforms(new_state['params'])
            new_state['ema_params'] = param_transforms(new_state['ema_params'])

        state = TrainState.create(
            apply_fn=model.apply,
            params=new_state['params'],
            ema_params=new_state['ema_params'],
            tx=optimizer,
            rngs=rngs,
            metrics=Metrics.empty(),
            dynamic_scale = DynamicScale() if use_dynamic_scale else None
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
            images = jnp.array(images, dtype=jnp.float32)
            # normalize image
            images = (images - 127.5) / 127.5
            
            if autoencoder is not None:
                # Convert the images to latent space
                local_rng_state, rngs = local_rng_state.get_random_key()
                images = autoencoder.encode(images, rngs)

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
            
            if train_state.dynamic_scale:
                # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
                # params should be restored (= skip this step).
                select_fn = functools.partial(jnp.where, is_fin)
                new_state = train_state.replace(
                    opt_state=jax.tree_util.tree_map(
                        select_fn, new_state.opt_state, train_state.opt_state
                    ),
                    params=jax.tree_util.tree_map(
                        select_fn, new_state.params, train_state.params
                    ),
                )
    
            train_state = new_state.apply_ema(self.ema_decay)
            
            if distributed_training:
                loss = jax.lax.pmean(loss, "data")
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
