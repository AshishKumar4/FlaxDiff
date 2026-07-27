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
from flaxdiff.utils import RandomMarkovState, convert_to_global_tree
from flax.training import dynamic_scale as dynamic_scale_lib
from dataclasses import dataclass
import shutil

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

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average#.from_output('loss')

# Define the TrainState
class SimpleTrainState(train_state.TrainState):
    metrics: Metrics
    dynamic_scale: dynamic_scale_lib.DynamicScale

def move_contents_to_subdir(target_dir, new_subdir_name):
    # --- 1. Validate Target Directory ---
    if not os.path.isdir(target_dir):
        print(f"Error: Target directory '{target_dir}' not found or is not a directory.")
        return
    # --- 2. Define Paths ---
    # Construct the full path for the new subdirectory
    new_subdir_path = os.path.join(target_dir, new_subdir_name)
    # --- 3. Create New Subdirectory ---
    try:
        # Create the subdirectory.
        # exist_ok=True prevents an error if the directory already exists.
        os.makedirs(new_subdir_path, exist_ok=True)
        print(f"Subdirectory '{new_subdir_path}' created or already exists.")
    except OSError as e:
        print(f"Error creating subdirectory '{new_subdir_path}': {e}")
        return # Stop execution if subdirectory creation fails
    # --- 4. List Contents of Target Directory ---
    try:
        items_to_move = os.listdir(target_dir)
    except OSError as e:
        print(f"Error listing contents of '{target_dir}': {e}")
        return # Stop if we can't list directory contents
    # --- 5. Move Items ---
    print(f"Moving items from '{target_dir}' to '{new_subdir_path}'...")
    moved_count = 0
    error_count = 0
    for item_name in items_to_move:
        # Construct the full path of the item in the target directory
        source_path = os.path.join(target_dir, item_name)
        # IMPORTANT: Skip the newly created subdirectory itself!
        if source_path == new_subdir_path:
            continue
        # Construct the destination path inside the new subdirectory
        destination_path = os.path.join(new_subdir_path, item_name)
        # Move the item
        try:
            shutil.move(source_path, destination_path)
            # print(f"  Moved: '{item_name}'") # Uncomment for verbose output
            moved_count += 1
        except Exception as e:
            print(f"  Error moving '{item_name}': {e}")
            error_count += 1
    print(f"\nOperation complete.")
    print(f"  Successfully moved: {moved_count} item(s).")
    if error_count > 0:
        print(f"  Errors encountered: {error_count} item(s).")

def load_from_checkpoint(
    checkpoint_dir: str,
):
    try:
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(create=False)
        # Convert checkpoint_dir to absolute path
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        manager = orbax.checkpoint.CheckpointManager(checkpoint_dir, checkpointer, options)
        ckpt = manager.restore(checkpoint_dir)
        # Extract as above
        state, best_state = None, None
        if 'state' in ckpt:
            state = ckpt['state']
        if 'best_state' in ckpt:
            best_state = ckpt['best_state']
        print(f"Loaded checkpoint from local dir {checkpoint_dir}")
        return state, best_state
    except Exception as e:
        print(f"Warning: Failed to load checkpoint from local dir: {e}")
        return None, None
    
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
                 wandb_config: Dict[str, Any] = None,
                 distributed_training: bool = None,
                 checkpoint_base_path: str = "./checkpoints",
                 checkpoint_step: int = None,
                 use_dynamic_scale: bool = False,
                 max_checkpoints_to_keep: int = 2,
                 train_start_step_override: int = None,
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
        
        load_directly_from_dir = False
        
        self.wandb = None
        if wandb_config is not None and jax.process_index() == 0:
            import wandb
            run = wandb.init(resume='allow', **wandb_config)
            self.wandb = run
            
            if 'id' in wandb_config:
                # If resuming from a previous run, and train_start_step_override is not set, 
                # set the start step to the last step of the previous run
                if train_start_step_override is None:
                    train_start_step_override = run.summary['train/step'] + 1
                print(f"Resuming from previous run {wandb_config['id']} with start step {train_start_step_override}")
                
                # If load_from_checkpoint is not set, and an artifact is found, load the artifact
                if load_from_checkpoint is None:
                    api_run = wandb.Api().run(f"{wandb_config['entity']}/{wandb_config['project']}/{wandb_config['id']}")
                    model_artifacts = [i for i in api_run.logged_artifacts() if i.type == 'model']
                    if model_artifacts:
                        artifact = model_artifacts[0]
                        artifact_dir = artifact.download()
                        print(f"Loading model from artifact {artifact.name} at {artifact_dir}")
                        # Move the artifact's contents
                        load_from_checkpoint = artifact_dir
                        load_directly_from_dir = True
            
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

        self.rngstate = RandomMarkovState(rngs)
        self.rngstate, subkey = self.rngstate.get_random_key()

        if train_state == None:
            state, best_state = self.generate_states(
                optimizer, subkey, model, use_dynamic_scale
            )
            self.init_state(state, best_state)
        else:
            self.state = train_state
            self.best_state = train_state
            self.best_loss = 1e9

        self.latest_step = 0
        if load_from_checkpoint is not None:
            self.latest_step = self.load(load_from_checkpoint, checkpoint_step, load_directly_from_dir)

        if train_start_step_override is not None:
            self.latest_step = train_start_step_override
            print(f"Overriding start step to {self.latest_step}")

    def get_input_ones(self):
        return {k: jnp.ones((1, *v)) for k, v in self.input_shapes.items()}

    def generate_states(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        model: nn.Module = None,
        use_dynamic_scale: bool = False
    ) -> Tuple[SimpleTrainState, SimpleTrainState]:
        print("Generating states for SimpleTrainer")
        rngs, subkey = jax.random.split(rngs)

        input_vars = self.get_input_ones()
        params = model.init(subkey, **input_vars)

        state = SimpleTrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            metrics=Metrics.empty(),
            dynamic_scale = dynamic_scale_lib.DynamicScale() if use_dynamic_scale else None
        )
        return state, state

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

    def load(self, checkpoint_path, checkpoint_step=None, load_directly_from_dir=False):
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

        # Restore against the freshly-initialized states as a template so orbax
        # rebuilds the exact pytree types - optimizer state and step included.
        # Restoring untyped used to silently discard opt_state and reset the
        # step counter (and with it the lr schedule) on every resume.
        template = {
            'rngs': self.get_rngstate(),
            'state': self.get_state(),
            'best_state': self.get_best_state(),
            'best_loss': np.array(self.best_loss),
            'epoch': 0,
        }
        ckpt = checkpointer.restore(step, items=template) if not load_directly_from_dir else checkpointer.restore(checkpoint_path, items=template)
        
        self.state = ckpt['state']
        self.best_state = ckpt['best_state']
        self.rngstate = ckpt['rngs']
        self.best_loss = float(ckpt['best_loss'])
        if self.best_loss == 0:
            # It cant be zero as that must have been some problem
            self.best_loss = 1e9
        print(f"Loaded model from checkpoint at step {step}", self.best_loss)
        return step

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
        raise NotImplementedError("Subclasses must define their train step")

    def _define_validation_step(self, **kwargs):
        raise NotImplementedError("Subclasses must define their validation step")

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
        bad_loss_steps = 0
        current_epoch = current_step // train_steps_per_epoch
        
        if process_index == 0:
            pbar = tqdm.tqdm(total=train_steps_per_epoch, desc=f'\t\tEpoch {current_epoch}', ncols=100, unit='step')
        else:
            pbar = None
            
        for i in range(train_steps_per_epoch):
            batch = next(train_ds)
            # if i == 0:
            #     print(f"First batch loaded at step {current_step}")
                
            if self.distributed_training and global_device_count > 1:
            #     # Convert the local device batches to a unified global jax.Array 
                batch = convert_to_global_tree(self.mesh, batch)
            train_state, loss, aux, rng_state = train_step_fn(train_state, rng_state, batch, global_device_indexes)

            if i == 0:
                print(f"Training started for process index {process_index} at step {current_step}")
                
            if self.distributed_training:
                # loss = jax.experimental.multihost_utils.process_allgather(loss)
                loss = jnp.mean(loss) # Just to make sure its a scaler value
                    
            if not jnp.isfinite(loss):
                # No silent recovery: a diverged run must fail loudly, not be
                # papered over with a stale best_state and a cosmetic loss value
                print(colored(f"Non-finite loss at step {current_step}: {loss}", 'red'))
                bad_loss_steps += 1
                if bad_loss_steps >= 5:
                    raise RuntimeError(
                        f"Loss has been non-finite for {bad_loss_steps} consecutive steps, stopping"
                    )
            else:
                bad_loss_steps = 0

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
                            **{f"train/{k}": v for k, v in aux.items()},
                        }, step=current_step)
                # Save the model every few steps
                if save_every and i % save_every == 0 and i > 0:
                    print(f"Saving model after {save_every} step {current_step}")
                    print(f"Devices: {len(jax.devices())}") # To sync the devices
                    self.save(current_epoch, current_step, train_state, rng_state)
                    print(f"Saving done by process index {process_index}")
                    print(colored(f"Epoch done on index {process_index} => {current_epoch} Loss: {epoch_loss/train_steps_per_epoch}", 'green'))
        if pbar is not None:
            pbar.close()
        return epoch_loss, current_step, train_state, rng_state


    def fit(self, data, train_steps_per_epoch, epochs, train_step_args={}, val_steps_per_epoch=5, validation_step_args={}):
        train_ds = iter(data['train']())
        val_ds = data.get('val', data.get('test', None))
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
                val_ds,
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
            
            if val_steps_per_epoch > 0:
                print(f"Validation started for process index {process_index}")
                # Validation step
                self.validation_loop(
                    train_state,
                    val_step,
                    val_ds,
                    val_steps_per_epoch,
                    current_step,
                )
                print(colored(f"Validation done on process index {process_index}", PROCESS_COLOR_MAP[process_index]))
            
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
                    
                
        self.save(epochs)#
        return self.state
