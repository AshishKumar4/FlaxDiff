import orbax.checkpoint
import tqdm
from flax import linen as nn
import jax
from typing import Callable
from dataclasses import field
import jax.numpy as jnp
import numpy as np
from functools import partial
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
import optax
from flax import struct                # Flax dataclasses
import flax
import time
import os
import orbax
from flax.training import orbax_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from orbax.checkpoint.utils import fully_replicated_host_local_array_to_global_array
from termcolor import colored
from typing import Dict, Callable, Sequence, Any, Union, Tuple
from flax.training.dynamic_scale import DynamicScale
from flaxdiff.utils import RandomMarkovState
from flax.training import dynamic_scale as dynamic_scale_lib
from dataclasses import dataclass
import gc

PROCESS_COLOR_MAP = {
    0: "green",
    1: "yellow",
    2: "magenta",
    3: "cyan", 
    4: "white",
    5: "light_blue",
    6: "light_red",
    7: "light_cyan"
}

def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], jax.sharding.NamedSharding]:
  sharding = jax.sharding.NamedSharding(global_mesh, P(global_mesh.axis_names))
  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]
  return global_shape, sharding


def form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
  """Put local sharded array into local devices"""
  global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)
  try:
    local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
  except ValueError as array_split_error:
    raise ValueError(
        f"Unable to put to devices shape {array.shape} with "
        f"local device count {len(global_mesh.local_devices)} "
    ) from array_split_error
  local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
  return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)

def convert_to_global_tree(global_mesh, pytree):
    return jax.tree_util.tree_map_with_path(partial(form_global_array, global_mesh=global_mesh), pytree)

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average#.from_output('loss')

# Define the TrainState
class SimpleTrainState(train_state.TrainState):
    metrics: Metrics
    dynamic_scale: dynamic_scale_lib.DynamicScale

@dataclass
class SimpleTrainer:
    state: SimpleTrainState
    best_state: SimpleTrainState
    best_loss: float
    model: nn.Module
    ema_decay: float = 0.999

    def __init__(self,
                 model: nn.Module,
                 input_shapes: Dict[str, Tuple[int]],
                 optimizer: optax.GradientTransformation,
                 rngs: jax.random.PRNGKey,
                 train_state: SimpleTrainState = None,
                 name: str = "Simple",
                 load_from_checkpoint: str = None,
                 loss_fn=optax.l2_loss,
                 param_transforms: Callable = None,
                 wandb_config: Dict[str, Any] = None,
                 distributed_training: bool = None,
                 checkpoint_base_path: str = "./checkpoints",
                 checkpoint_step: int = None,
                 use_dynamic_scale: bool = False,
                 max_checkpoints_to_keep: int = 2,
                 ):
        if distributed_training is None or distributed_training is True:
            # Auto-detect if we are running on multiple devices
            distributed_training = jax.device_count() > 1
            self.mesh = jax.sharding.Mesh(jax.devices(), 'data')
        else:
            self.mesh = None

        self.distributed_training = distributed_training
        self.model = model
        self.name = name
        self.loss_fn = loss_fn
        self.input_shapes = input_shapes
        self.checkpoint_base_path = checkpoint_base_path
        
        if wandb_config is not None and jax.process_index() == 0:
            import wandb
            run = wandb.init(resume='allow', **wandb_config)
            self.wandb = run
            
            # define our custom x axis metric
            self.wandb.define_metric("train/step")
            self.wandb.define_metric("train/epoch")
            
            self.wandb.define_metric("train/loss", step_metric="train/step")
            
            self.wandb.define_metric("train/epoch_time", step_metric="train/epoch")
            self.wandb.define_metric("train/avg_time_per_step", step_metric="train/epoch")
            self.wandb.define_metric("train/avg_loss", step_metric="train/epoch")
            self.wandb.define_metric("train/best_loss", step_metric="train/epoch")
            
            if self.wandb.sweep_id:
                api = wandb.Api()
                self.wandb_sweep = api.sweep(f"{self.wandb.entity}/{self.wandb.project}/{self.wandb.sweep_id}")
                print(f"Running sweep {self.wandb_sweep.id} with id {self.wandb.sweep_id}")
            
        # checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        async_checkpointer = orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=60)

        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=max_checkpoints_to_keep, create=True)
        self.checkpointer = orbax.checkpoint.CheckpointManager(
            self.checkpoint_path(), async_checkpointer, options)

        if load_from_checkpoint is not None:
            latest_epoch, latest_step, old_state, old_best_state, rngstate = self.load(load_from_checkpoint, checkpoint_step)
        else:
            latest_epoch, latest_step, old_state, old_best_state, rngstate = 0, 0, None, None, None

        self.latest_step = latest_step
        
        if rngstate:
            self.rngstate = RandomMarkovState(**rngstate)
        else:
            self.rngstate = RandomMarkovState(rngs)
            
        self.rngstate, subkey = self.rngstate.get_random_key()

        if train_state == None:
            state, best_state = self.generate_states(
                optimizer, subkey, old_state, old_best_state, model, param_transforms, use_dynamic_scale
            )
            self.init_state(state, best_state)
        else:
            self.state = train_state
            self.best_state = train_state
            self.best_loss = 1e9

    def get_input_ones(self):
        return {k: jnp.ones((1, *v)) for k, v in self.input_shapes.items()}

    def generate_states(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        existing_state: dict = None,
        existing_best_state: dict = None,
        model: nn.Module = None,
        param_transforms: Callable = None,
        use_dynamic_scale: bool = False
    ) -> Tuple[SimpleTrainState, SimpleTrainState]:
        print("Generating states for SimpleTrainer")
        rngs, subkey = jax.random.split(rngs)

        if existing_state == None:
            input_vars = self.get_input_ones()
            params = model.init(subkey, **input_vars)
        else:
            params = existing_state['params']

        state = SimpleTrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            metrics=Metrics.empty(),
            dynamic_scale = dynamic_scale_lib.DynamicScale() if use_dynamic_scale else None
        )
        if existing_best_state is not None:
            best_state = state.replace(
                params=existing_best_state['params'])
        else:
            best_state = state
            
        return state, best_state

    def init_state(
        self,
        state: SimpleTrainState,
        best_state: SimpleTrainState,
    ):
        self.best_loss = 1e9

        self.state = state
        self.best_state = best_state

    def get_state(self):
        return self.get_np_tree(self.state)

    def get_best_state(self):
        return self.get_np_tree(self.best_state)
        
    def get_rngstate(self):
        return self.get_np_tree(self.rngstate)
    
    def get_np_tree(self, pytree):
        return jax.tree_util.tree_map(lambda x : np.array(x), pytree)

    def checkpoint_path(self):
        path = os.path.join(self.checkpoint_base_path, self.name.replace(' ', '_').lower())
        # Convert the path to an absolute path
        path = os.path.abspath(path)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def tensorboard_path(self):
        experiment_name = self.name
        path = os.path.join(os.path.abspath('./tensorboard'), experiment_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def load(self, checkpoint_path=None, checkpoint_step=None):
        if checkpoint_path is None:
            checkpointer = self.checkpointer
        else:
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            options = orbax.checkpoint.CheckpointManagerOptions(
                max_to_keep=4, create=False)
            checkpointer = orbax.checkpoint.CheckpointManager(
                checkpoint_path, checkpointer, options)    
        
        if checkpoint_step is None:
            step = checkpointer.latest_step()
        else:
            step = checkpoint_step
        
        print("Loading model from checkpoint at step ", step)
        loaded_checkpoint_path = os.path.join(
            checkpoint_path if checkpoint_path else self.checkpoint_path(),
            f"{step}")
        self.loaded_checkpoint_path = loaded_checkpoint_path
        ckpt = checkpointer.restore(step)
        state = ckpt['state']
        best_state = ckpt['best_state']
        rngstate = ckpt['rngs']
        # Convert the state to a TrainState
        self.best_loss = ckpt['best_loss']
        if self.best_loss == 0:
            # It cant be zero as that must have been some problem
            self.best_loss = 1e9
        current_epoch = ckpt.get('epoch', step) # Must be a checkpoint from an older version which used epochs instead of steps
        print(
            f"Loaded model from checkpoint at epoch {current_epoch} step {step}", ckpt['best_loss'])
        return current_epoch, step, state, best_state, rngstate

    def save(self, epoch=0, step=0, state=None, rngstate=None):
        print(f"Saving model at epoch {epoch} step {step}")
        try:
            ckpt = {
                # 'model': self.model,
                'rngs': self.get_rngstate() if rngstate is None else self.get_np_tree(rngstate),
                'state': self.get_state() if state is None else self.get_np_tree(state),
                'best_state': self.get_best_state(),
                'best_loss': np.array(self.best_loss),
                'epoch': epoch,
            }
            try:
                save_args = orbax_utils.save_args_from_target(ckpt)
                self.checkpointer.save(step, ckpt, save_kwargs={
                                    'save_args': save_args}, force=True)
                self.checkpointer.wait_until_finished()
                pass
            except Exception as e:
                print("Error saving checkpoint", e)
        except Exception as e:
            print("Error saving checkpoint outer", e)

    def _define_train_step(self, **kwargs):
        model = self.model
        loss_fn = self.loss_fn
        distributed_training = self.distributed_training

        def train_step(train_state: SimpleTrainState, rng_state: RandomMarkovState, batch, local_device_indexes):
            """Train for a single step."""
            images = batch['image']
            labels = batch['label']

            def model_loss(params):
                preds = model.apply(params, images)
                expected_output = labels
                nloss = loss_fn(preds, expected_output)
                loss = jnp.mean(nloss)
                return loss
            loss, grads = jax.value_and_grad(model_loss)(train_state.params)
            if distributed_training:
                grads = jax.lax.pmean(grads, "data")
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss, rng_state
        
        if distributed_training:
            train_step = shard_map(train_step, mesh=self.mesh, in_specs=(P(), P(), P('data'), P('data')), out_specs=(P(), P('data'), P()))
            train_step = jax.pmap(train_step)
        return train_step

    def _define_validation_step(self):
        model = self.model
        loss_fn = self.loss_fn
        distributed_training = self.distributed_training

        def validation_step(state: SimpleTrainState, batch):
            preds = model.apply(state.params, batch['image'])
            expected_output = batch['label']
            loss = jnp.mean(loss_fn(preds, expected_output))
            if distributed_training:
                loss = jax.lax.pmean(loss, "data")
            metric_updates = state.metrics.single_from_model_output(
                loss=loss, logits=preds, labels=expected_output)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        if distributed_training:
            validation_step = shard_map(validation_step, mesh=self.mesh, in_specs=(P(), P('data')), out_specs=(P()))
            validation_step = jax.pmap(validation_step)
        return validation_step

    def summary(self):
        input_vars = self.get_input_ones()
        print(self.model.tabulate(jax.random.key(0), **input_vars,
              console_kwargs={"width": 200, "force_jupyter": True, }))

    def config(self):
        return {
            "model": self.model,
            "state": self.state,
            "name": self.name,
            "input_shapes": self.input_shapes
        }

    def init_tensorboard(self, batch_size, steps_per_epoch, epochs):
        from flax.metrics import tensorboard
        summary_writer = tensorboard.SummaryWriter(self.tensorboard_path())
        summary_writer.hparams({
            **self.config(),
            "steps_per_epoch": steps_per_epoch,
            "epochs": epochs,
            "batch_size": batch_size
        })
        return summary_writer
    
    def validation_loop(
        self,
        val_state: SimpleTrainState,
        val_step_fn: Callable,
        val_ds,
        val_steps_per_epoch,
        current_step,
    ):
        global_device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        process_index = jax.process_index()
        
        val_ds = iter(val_ds()) if val_ds else None
        # Evaluation step
        try:
            for i in range(val_steps_per_epoch):
                if val_ds is None:
                    batch = None
                else:
                    batch = next(val_ds)
                    if self.distributed_training and global_device_count > 1:
                        batch = convert_to_global_tree(self.mesh, batch)
                if i == 0:
                    print(f"Evaluation started for process index {process_index}")
                metrics = val_step_fn(val_state, batch)
                if self.wandb is not None:
                    # metrics is a dict of metrics
                    if metrics and type(metrics) == dict:
                        for key, value in metrics.items():
                            if isinstance(value, jnp.ndarray):
                                value = np.array(value)
                            self.wandb.log({
                                f"val/{key}": value,
                            }, step=current_step)
        except Exception as e:
            print("Error logging images to wandb", e)

    def train_loop(
        self,
        train_state: SimpleTrainState,
        train_step_fn: Callable,
        train_ds,
        train_steps_per_epoch,
        current_step,
        rng_state,
        save_every:int=None,
        val_every=None,
    ):
        global_device_count = jax.device_count()
        process_index = jax.process_index()
        if self.distributed_training:
            global_device_indexes = jnp.arange(global_device_count)
        else:
            global_device_indexes = 0
            
        epoch_loss = 0
        current_epoch = current_step // train_steps_per_epoch
        last_save_time = time.time()
        
        if process_index == 0:
            pbar = tqdm.tqdm(total=train_steps_per_epoch, desc=f'\t\tEpoch {current_epoch}', ncols=100, unit='step')
            
        for i in range(train_steps_per_epoch):
            batch = next(train_ds)
            # if i == 0:
            #     print(f"First batch loaded at step {current_step}")
                
            if self.distributed_training and global_device_count > 1:
            #     # Convert the local device batches to a unified global jax.Array 
                batch = convert_to_global_tree(self.mesh, batch)
            train_state, loss, rng_state = train_step_fn(train_state, rng_state, batch, global_device_indexes)

            if i == 0:
                print(f"Training started for process index {process_index} at step {current_step}")
                
            if self.distributed_training:
                # loss = jax.experimental.multihost_utils.process_allgather(loss)
                loss = jnp.mean(loss) # Just to make sure its a scaler value
                    
            if loss <= 1e-8 or jnp.isnan(loss) or jnp.isinf(loss):
                # If the loss is too low or NaN/Inf, log the issue and attempt recovery
                print(colored(f"Abnormal loss at step {current_step}: {loss}", 'red'))
                
                # Check model parameters for NaN/Inf values
                params = train_state.params
                has_nan_or_inf = False
                
                if isinstance(params, dict):
                    for key, value in params.items():
                        if isinstance(value, jnp.ndarray):
                            if jnp.isnan(value).any() or jnp.isinf(value).any():
                                print(colored(f"NaN/inf values found in params[{key}] at step {current_step}", 'red'))
                                has_nan_or_inf = True
                                break
                    
                    if not has_nan_or_inf:
                        print(colored(f"Model parameters seem valid despite abnormal loss", 'yellow'))
                
                # Try to recover - clear JAX caches and collect garbage
                gc.collect()
                if hasattr(jax, "clear_caches"):
                    jax.clear_caches()
                
                # If we have a best state and the loss is truly invalid, consider restoring
                if (loss <= 1e-8 or jnp.isnan(loss) or jnp.isinf(loss)) and self.best_state is not None:
                    print(colored(f"Attempting recovery by resetting model to last best state", 'yellow'))
                    train_state = self.best_state
                    loss = self.best_loss
                else:
                    # If we can't recover, skip this step but continue training
                    print(colored(f"Unable to recover - continuing with current state", 'yellow'))
                    if loss <= 1e-8:
                        loss = 1.0  # Set to a reasonable default to continue training
                            
            epoch_loss += loss
            current_step += 1
            if i % 100 == 0:
                if pbar is not None:
                    pbar.set_postfix(loss=f'{loss:.4f}')
                    pbar.update(100)
                    if self.wandb is not None:
                        self.wandb.log({
                            "train/step" : current_step,
                            "train/loss": loss,
                        }, step=current_step)
                # Save the model every few steps
                if save_every and i % save_every == 0 and i > 0:
                    print(f"Saving model after {save_every} step {current_step}")
                    print(f"Devices: {len(jax.devices())}") # To sync the devices
                    self.save(current_epoch, current_step, train_state, rng_state)
                    print(f"Saving done by process index {process_index}")
                    last_save_time = time.time()
        print(colored(f"Epoch done on index {process_index} => {current_epoch} Loss: {epoch_loss/train_steps_per_epoch}", 'green'))
        if pbar is not None:
            pbar.close()
        return epoch_loss, current_step, train_state, rng_state


    def fit(self, data, train_steps_per_epoch, epochs, train_step_args={}, val_steps_per_epoch=5, validation_step_args={}):
        train_ds = iter(data['train']())
        train_step = self._define_train_step(**train_step_args)
        val_step = self._define_validation_step(**validation_step_args)
        train_state = self.state
        rng_state = self.rngstate
        process_index = jax.process_index()
        
        if val_steps_per_epoch > 0:
            # We should first run a validation step to make sure the model is working
            print(f"Validation run for sanity check for process index {process_index}")
            # Validation step
            self.validation_loop(
                train_state,
                val_step,
                data.get('val', data.get('test', None)),
                val_steps_per_epoch,
                self.latest_step,
            )
            print(colored(f"Sanity Validation done on process index {process_index}", PROCESS_COLOR_MAP[process_index]))
                
        while self.latest_step < epochs * train_steps_per_epoch:
            current_epoch = self.latest_step // train_steps_per_epoch
            print(f"\nEpoch {current_epoch}/{epochs}")
            start_time = time.time()
            epoch_loss = 0
            
            epoch_loss, current_step, train_state, rng_state = self.train_loop(
                train_state,
                train_step,
                train_ds,
                train_steps_per_epoch,
                self.latest_step,
                rng_state,
            )
            print(colored(f"Epoch done on process index {process_index}", PROCESS_COLOR_MAP[process_index]))
            
            self.latest_step = current_step
            end_time = time.time()
            self.state = train_state
            self.rngstate = rng_state
            total_time = end_time - start_time
            avg_time_per_step = total_time / train_steps_per_epoch
            avg_loss = epoch_loss / train_steps_per_epoch
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = train_state
                self.save(current_epoch, current_step)
                
            if process_index == 0:
                if self.wandb is not None:
                    self.wandb.log({
                        "train/epoch_time": total_time,
                        "train/avg_time_per_step": avg_time_per_step,
                        "train/avg_loss": avg_loss,
                        "train/best_loss": self.best_loss,
                        "train/epoch": current_epoch,
                    }, step=current_step)
                print(colored(f"\n\tEpoch {current_epoch} completed. Avg Loss: {avg_loss}, Time: {total_time:.2f}s, Best Loss: {self.best_loss}", 'green'))
                    
            if val_steps_per_epoch > 0:
                print(f"Validation started for process index {process_index}")
                # Validation step
                self.validation_loop(
                    train_state,
                    val_step,
                    data.get('val', data.get('test', None)),
                    val_steps_per_epoch,
                    current_step,
                )
                print(colored(f"Validation done on process index {process_index}", PROCESS_COLOR_MAP[process_index]))
                
        self.save(epochs)
        return self.state
