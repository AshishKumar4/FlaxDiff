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

from flaxdiff.utils import RandomMarkovState

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
    loss: metrics.Average.from_output('loss')

# Define the TrainState
class SimpleTrainState(train_state.TrainState):
    rngs: jax.random.PRNGKey
    metrics: Metrics

    def get_random_key(self):
        rngs, subkey = jax.random.split(self.rngs)
        return self.replace(rngs=rngs), subkey

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
                 load_from_checkpoint: bool = False,
                 checkpoint_suffix: str = "",
                 checkpoint_id: str = None,
                 loss_fn=optax.l2_loss,
                 param_transforms: Callable = None,
                 wandb_config: Dict[str, Any] = None,
                 distributed_training: bool = None,
                 checkpoint_base_path: str = "./checkpoints",
                 ):
        if distributed_training is None or distributed_training is True:
            # Auto-detect if we are running on multiple devices
            distributed_training = jax.device_count() > 1
            self.mesh = jax.sharding.Mesh(jax.devices(), 'data')
            # self.sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec('data'))

        self.distributed_training = distributed_training
        self.model = model
        self.name = name
        self.loss_fn = loss_fn
        self.input_shapes = input_shapes
        self.checkpoint_base_path = checkpoint_base_path
        
        
        if wandb_config is not None and jax.process_index() == 0:
            import wandb
            run = wandb.init(**wandb_config)
            self.wandb = run
            
            # define our custom x axis metric
            self.wandb.define_metric("train/step")
            self.wandb.define_metric("train/epoch")
            
            self.wandb.define_metric("train/loss", step_metric="train/step")
            
            self.wandb.define_metric("train/epoch_time", step_metric="train/epoch")
            self.wandb.define_metric("train/avg_time_per_step", step_metric="train/epoch")
            self.wandb.define_metric("train/avg_loss", step_metric="train/epoch")
            self.wandb.define_metric("train/best_loss", step_metric="train/epoch")

        if checkpoint_id is None:
            self.checkpoint_id = name.replace(' ', '_').replace('-', '_').lower()
        else:
            self.checkpoint_id = checkpoint_id
            
        # checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        async_checkpointer = orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=60)

        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=4, create=True)
        self.checkpointer = orbax.checkpoint.CheckpointManager(
            self.checkpoint_path() + checkpoint_suffix, async_checkpointer, options)

        if load_from_checkpoint:
            latest_epoch, old_state, old_best_state, rngstate = self.load()
        else:
            latest_epoch, old_state, old_best_state, rngstate = 0, None, None, None

        self.latest_epoch = latest_epoch
        
        if rngstate:
            self.rngstate = RandomMarkovState(**rngstate)
        else:
            self.rngstate = RandomMarkovState(rngs)
            
        self.rngstate, subkey = self.rngstate.get_random_key()

        if train_state == None:
            state, best_state = self.generate_states(
                optimizer, subkey, old_state, old_best_state, model, param_transforms
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
        param_transforms: Callable = None
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
            metrics=Metrics.empty()
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
        # return fully_replicated_host_local_array_to_global_array()
        return jax.tree_util.tree_map(lambda x : np.array(x), self.state)

    def get_best_state(self):
        # return convert_to_global_tree(self.mesh, flax.jax_utils.replicate(self.best_state, jax.local_devices()))
        return jax.tree_util.tree_map(lambda x : np.array(x), self.best_state)
        
    def get_rngstate(self):
        # return convert_to_global_tree(self.mesh, flax.jax_utils.replicate(self.rngstate, jax.local_devices()))
        return jax.tree_util.tree_map(lambda x : np.array(x), self.rngstate)

    def checkpoint_path(self):
        path = os.path.join(self.checkpoint_base_path, self.checkpoint_id)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def tensorboard_path(self):
        experiment_name = self.name
        path = os.path.join(os.path.abspath('./tensorboard'), experiment_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def load(self):
        epoch = self.checkpointer.latest_step()
        print("Loading model from checkpoint", epoch)
        ckpt = self.checkpointer.restore(epoch)
        state = ckpt['state']
        best_state = ckpt['best_state']
        rngstate = ckpt['rngs']
        # Convert the state to a TrainState
        self.best_loss = ckpt['best_loss']
        print(
            f"Loaded model from checkpoint at epoch {epoch}", ckpt['best_loss'])
        return epoch, state, best_state, rngstate

    def save(self, epoch=0):
        print(f"Saving model at epoch {epoch}")
        ckpt = {
            # 'model': self.model,
            'rngs': self.get_rngstate(),
            'state': self.get_state(),
            'best_state': self.get_best_state(),
            'best_loss': np.array(self.best_loss),
        }
        try:
            save_args = orbax_utils.save_args_from_target(ckpt)
            self.checkpointer.save(epoch, ckpt, save_kwargs={
                                   'save_args': save_args}, force=True)
            self.checkpointer.wait_until_finished()
            pass
        except Exception as e:
            print("Error saving checkpoint", e)

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

    def _define_compute_metrics(self):
        model = self.model
        loss_fn = self.loss_fn

        @jax.jit
        def compute_metrics(state: SimpleTrainState, batch):
            preds = model.apply(state.params, batch['image'])
            expected_output = batch['label']
            loss = jnp.mean(loss_fn(preds, expected_output))
            metric_updates = state.metrics.single_from_model_output(
                loss=loss, logits=preds, labels=expected_output)
            metrics = state.metrics.merge(metric_updates)
            state = state.replace(metrics=metrics)
            return state
        return compute_metrics

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

    def fit(self, data, steps_per_epoch, epochs, train_step_args={}):
        train_ds = iter(data['train']())
        if 'test' in data:
            test_ds = data['test']
        else:
            test_ds = None
        train_step = self._define_train_step(**train_step_args)
        compute_metrics = self._define_compute_metrics()
        train_state = self.state
        rng_state = self.rngstate
        global_device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        process_index = jax.process_index()
        if self.distributed_training:
            global_device_indexes = jnp.arange(global_device_count)
        else:
            global_device_indexes = 0

        def train_loop(current_epoch, pbar: tqdm.tqdm, train_state, rng_state):
            epoch_loss = 0
            current_step = 0
            for i in range(steps_per_epoch):
                batch = next(train_ds)
                if self.distributed_training and global_device_count > 1:
                    # Convert the local device batches to a unified global jax.Array 
                    batch = convert_to_global_tree(self.mesh, batch)
                train_state, loss, rng_state = train_step(train_state, rng_state, batch, global_device_indexes)

                if self.distributed_training:
                    loss = jax.experimental.multihost_utils.process_allgather(loss)
                    loss = jnp.mean(loss) # Just to make sure its a scaler value
                            
                epoch_loss += loss
                    
                if pbar is not None:
                    if i % 100 == 0:
                        pbar.set_postfix(loss=f'{loss:.4f}')
                        pbar.update(100)
                        current_step = current_epoch*steps_per_epoch + i
                        if self.wandb is not None:
                            self.wandb.log({
                                "train/step" : current_step,
                                "train/loss": loss,
                            }, step=current_step)
            print(colored(f"Epoch done on index {process_index} => {current_epoch} Loss: {epoch_loss/steps_per_epoch}", 'green'))
            return epoch_loss, current_step, train_state, rng_state

        while self.latest_epoch < epochs:
            current_epoch = self.latest_epoch
            self.latest_epoch += 1
            print(f"\nEpoch {current_epoch}/{epochs}")
            start_time = time.time()
            epoch_loss = 0

            if process_index == 0:
                with tqdm.tqdm(total=steps_per_epoch, desc=f'\t\tEpoch {current_epoch}', ncols=100, unit='step') as pbar:
                    epoch_loss, current_step, train_state, rng_state = train_loop(current_epoch, pbar, train_state, rng_state)
            else:
                epoch_loss, current_step, train_state, rng_state = train_loop(current_epoch, None, train_state, rng_state)
                print(colored(f"Epoch done on process index {process_index}", PROCESS_COLOR_MAP.get(process_index, 'white')))
                
            end_time = time.time()
            self.state = train_state
            self.rngstate = rng_state
            total_time = end_time - start_time
            avg_time_per_step = total_time / steps_per_epoch
            avg_loss = epoch_loss / steps_per_epoch
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = train_state
                self.save(current_epoch)
                
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
        self.save(epochs)
        return self.state