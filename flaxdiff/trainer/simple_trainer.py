import tqdm
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
import optax
from flax import struct                # Flax dataclasses
import time
import os
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding, PartitionSpec as P
from termcolor import colored
from typing import Dict, Callable, Any, Tuple
from flaxdiff.utils import (
    DEFAULT_MIN_SHARD_SIZE, DevicePrefetchIterator, RandomMarkovState, batch_sharding,
    build_mesh, enable_compilation_cache, model_flops_utilization, shard_batch,
    state_sharding_tree, step_flops,
)
from flax.training import dynamic_scale as dynamic_scale_lib
from dataclasses import dataclass

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

@dataclass
class SimpleTrainer:
    state: SimpleTrainState
    best_state: Any
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
                 fsdp_size: int = 1,
                 fsdp_min_param_size: int = DEFAULT_MIN_SHARD_SIZE,
                 compilation_cache_dir: str = None,
                 profile_steps: int = 0,
                 log_every: int = 100,
                 max_bad_loss_steps: int = 5,
                 ):
        if compilation_cache_dir:
            enable_compilation_cache(compilation_cache_dir)

        # One code path for every topology: the mesh spans all devices unless
        # the caller explicitly opts out, and a 1x1 mesh behaves exactly like
        # the old single-device path.
        if distributed_training is None:
            distributed_training = jax.device_count() > 1
        self.distributed_training = distributed_training
        devices = jax.devices() if distributed_training else jax.devices()[:1]
        self.mesh = build_mesh(fsdp_size, devices=devices)
        self.batch_sharding = batch_sharding(self.mesh)
        self.replicated = NamedSharding(self.mesh, P())
        self.fsdp_min_param_size = fsdp_min_param_size

        self.model = model
        self.name = name
        self.loss_fn = loss_fn
        self.input_shapes = input_shapes
        self.checkpoint_base_path = checkpoint_base_path
        self.profile_steps = profile_steps
        self.log_every = log_every
        self.max_bad_loss_steps = max_bad_loss_steps
        # Measured from the compiled step the first time it runs.
        self.flops_per_step = None
        self.global_batch_size = 0

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
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_checkpoints_to_keep, create=True,
            enable_async_checkpointing=True)
        self.checkpointer = ocp.CheckpointManager(self.checkpoint_path(), options=options)

        self.rngstate = RandomMarkovState(rngs)
        self.rngstate, subkey = self.rngstate.get_random_key()

        self.best_loss = 1e9
        if train_state is None:
            self.state = self.generate_states(optimizer, subkey, model, use_dynamic_scale)
        else:
            self.state = train_state
            self.state_sharding = jax.tree.map(lambda x: x.sharding, train_state)
        # Host-side copy: aliasing a live train state here would pin its buffers
        # and block donating them to the training step.
        self.best_state = self.get_np_tree(self.state)
        # Position of the data iterator, carried through checkpoints so a resume
        # continues mid-epoch instead of replaying from the top.
        self.dataset_state = None

        self.latest_step = 0
        if load_from_checkpoint is not None:
            self.latest_step = self.load(load_from_checkpoint, checkpoint_step, load_directly_from_dir)

        if train_start_step_override is not None:
            self.latest_step = train_start_step_override
            print(f"Overriding start step to {self.latest_step}")

    def get_input_ones(self):
        return {k: jnp.ones((1, *v)) for k, v in self.input_shapes.items()}

    def _build_state(self, init_fn) -> SimpleTrainState:
        """Materialise a train state directly into its sharded layout.

        The sharding is derived from the abstract state, so optimizer moments
        and EMA copies inherit their params' layout through tx.init without any
        model or optimizer having to declare partitioning.
        """
        self.state_sharding = state_sharding_tree(
            self.mesh, jax.eval_shape(init_fn), self.fsdp_min_param_size)
        return jax.jit(init_fn, out_shardings=self.state_sharding)()

    def generate_states(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        model: nn.Module = None,
        use_dynamic_scale: bool = False
    ) -> SimpleTrainState:
        print("Generating states for SimpleTrainer")

        def init_fn():
            _, subkey = jax.random.split(rngs)
            return SimpleTrainState.create(
                apply_fn=model.apply,
                params=model.init(subkey, **self.get_input_ones()),
                tx=optimizer,
                metrics=Metrics.empty(),
                dynamic_scale=dynamic_scale_lib.DynamicScale() if use_dynamic_scale else None,
            )

        return self._build_state(init_fn)

    def get_state(self):
        return self.get_np_tree(self.state)

    def get_best_state(self):
        return self.best_state

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

    def _checkpoint_template(self, stored_keys):
        """Restore template plus the per-leaf args that place arrays on the mesh.

        Shapes and types come from the freshly built state, so a checkpoint
        written on one mesh restores onto whatever mesh this run is using.
        Restoring untyped used to silently discard opt_state and reset the step
        counter (and with it the lr schedule) on every resume.

        The template must name exactly the keys the checkpoint holds: asking for
        one it lacks is an error, and checkpoints predating iterator tracking -
        or written from an iterator that cannot report a position - have no
        dataset_state.
        """
        template = {
            'rngs': self.get_rngstate(),
            'state': jax.eval_shape(lambda: self.state),
            'best_state': self.best_state,
            'best_loss': np.array(self.best_loss),
            'epoch': 0,
        }
        if 'dataset_state' in stored_keys:
            # Length varies with the iterator's position, so orbax takes the
            # shape from the checkpoint rather than from this placeholder.
            template['dataset_state'] = np.zeros((1,), np.uint8)
        restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(), template)
        # Only the train state is placed onto the mesh; everything else is
        # bookkeeping that belongs on the host.
        restore_args['state'] = jax.tree.map(
            lambda s: ocp.ArrayRestoreArgs(sharding=s), self.state_sharding)
        return template, restore_args

    def load(self, checkpoint_path, checkpoint_step=None, load_directly_from_dir=False):
        # The handler has to be registered for item_metadata to report the
        # checkpoint's structure, which is what the template is built against.
        manager = ocp.CheckpointManager(
            checkpoint_path, options=ocp.CheckpointManagerOptions(max_to_keep=4, create=False),
            item_handlers=ocp.PyTreeCheckpointHandler())

        step = manager.latest_step() if checkpoint_step is None else checkpoint_step
        print("Loading model from checkpoint at step ", step)
        self.loaded_checkpoint_path = os.path.join(
            checkpoint_path if checkpoint_path else self.checkpoint_path(), f"{step}")

        target = checkpoint_path if load_directly_from_dir else step
        template, restore_args = self._checkpoint_template(manager.item_metadata(target).keys())
        ckpt = manager.restore(
            target, args=ocp.args.PyTreeRestore(item=template, restore_args=restore_args))

        self.state = ckpt['state']
        self.best_state = ckpt['best_state']
        self.rngstate = ckpt['rngs']
        stored_position = ckpt.get('dataset_state')
        self.dataset_state = (
            None if stored_position is None
            else np.asarray(stored_position, np.uint8).tobytes())
        self.best_loss = float(ckpt['best_loss'])
        if self.best_loss == 0:
            # It cant be zero as that must have been some problem
            self.best_loss = 1e9
        print(f"Loaded model from checkpoint at step {step}", self.best_loss)
        return step

    def save(self, epoch=0, step=0, state=None, rngstate=None):
        print(f"Saving model at epoch {epoch} step {step}")
        # Sharded arrays go straight to orbax: gathering them onto the host
        # first would serialise the whole state through one process and undo
        # the point of an async checkpointer.
        ckpt = {
            'rngs': self.get_rngstate() if rngstate is None else self.get_np_tree(rngstate),
            'state': self.state if state is None else state,
            'best_state': self.best_state,
            'best_loss': np.array(self.best_loss),
            'epoch': epoch,
        }
        if self.dataset_state is not None:
            # Grain reports its position as JSON bytes, which tensorstore has no
            # dtype for; the raw bytes ride along as a uint8 array instead.
            ckpt['dataset_state'] = np.frombuffer(self.dataset_state, np.uint8)
        try:
            self.checkpointer.save(step, args=ocp.args.PyTreeSave(ckpt), force=True)
        except Exception as e:
            print("Error saving checkpoint", e)

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
        process_index = jax.process_index()
        
        val_ds = iter(val_ds()) if val_ds else None
        # Evaluation step
        try:
            for i in range(val_steps_per_epoch):
                if val_ds is None:
                    batch = None
                else:
                    batch = shard_batch(self.batch_sharding, next(val_ds))
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
        process_index = jax.process_index()
        log_every = self.log_every

        epoch_loss = 0
        current_epoch = current_step // train_steps_per_epoch

        # Both counters live on device so the loop never blocks on a result.
        # `worst_bad_run` remembers the longest streak of non-finite losses seen
        # since the last host check, which is what decides whether to stop.
        bad_run = jnp.zeros((), jnp.int32)
        worst_bad_run = jnp.zeros((), jnp.int32)

        if process_index == 0:
            pbar = tqdm.tqdm(total=train_steps_per_epoch, desc=f'\t\tEpoch {current_epoch}', ncols=100, unit='step')
        else:
            pbar = None

        last_log_time = time.time()
        steps_since_log = 0

        for i in range(train_steps_per_epoch):
            batch = next(train_ds)
            if i == 0 and self.profile_steps:
                jax.profiler.start_trace(self.profile_path())

            train_state, loss, rng_state, is_finite = train_step_fn(train_state, rng_state, batch)
            # No stale alias may outlive the step: its buffers were donated.
            self.state, self.rngstate = train_state, rng_state
            self.dataset_state = getattr(train_ds, 'source_state', None)

            bad_run = jnp.where(is_finite, 0, bad_run + 1)
            worst_bad_run = jnp.maximum(worst_bad_run, bad_run)

            if i == 0:
                print(f"Training started for process index {process_index} at step {current_step}")
                if self.flops_per_step is None:
                    self.flops_per_step = step_flops(
                        train_step_fn, train_state, rng_state, batch)

            epoch_loss += loss
            current_step += 1
            steps_since_log += 1

            if self.profile_steps and i + 1 == self.profile_steps:
                loss.block_until_ready()
                jax.profiler.stop_trace()
                print(f"Wrote profile for {self.profile_steps} steps to {self.profile_path()}")

            if i % log_every == 0:
                self._check_finite(worst_bad_run, current_step)
                worst_bad_run = jnp.zeros((), jnp.int32)
                if pbar is not None:
                    # The one place per interval where waiting on the device is
                    # justified: the numbers below are meaningless without it.
                    loss.block_until_ready()
                    now = time.time()
                    elapsed = now - last_log_time
                    pbar.set_postfix(loss=f'{loss:.4f}')
                    pbar.update(log_every)
                    if self.wandb is not None:
                        self.wandb.log({
                            "train/step": current_step,
                            "train/loss": loss,
                            **self._throughput_metrics(elapsed, steps_since_log),
                        }, step=current_step)
                    last_log_time, steps_since_log = now, 0
                # Save the model every few steps
                if save_every and i % save_every == 0 and i > 0:
                    print(f"Saving model after {save_every} step {current_step}")
                    self.save(current_epoch, current_step, train_state, rng_state)
                    print(f"Saving done by process index {process_index}")
                    print(colored(f"Epoch done on index {process_index} => {current_epoch} Loss: {epoch_loss/train_steps_per_epoch}", 'green'))
        self._check_finite(worst_bad_run, current_step)
        if pbar is not None:
            pbar.close()
        return epoch_loss, current_step, train_state, rng_state

    def profile_path(self):
        return os.path.join(self.checkpoint_path(), 'profile')

    def _check_finite(self, worst_bad_run, current_step):
        """Fail a diverged run loudly rather than papering over it.

        Deferred to the logging cadence so the step loop never synchronises;
        detection is late by at most that many steps, never missed.
        """
        streak = int(worst_bad_run)
        if streak >= self.max_bad_loss_steps:
            raise RuntimeError(
                f"Loss has been non-finite for {streak} consecutive steps "
                f"ending near step {current_step}, stopping")
        if streak:
            print(colored(f"Non-finite loss for {streak} step(s) before {current_step}", 'red'))

    def _throughput_metrics(self, elapsed: float, steps: int) -> Dict[str, float]:
        if elapsed <= 0 or steps <= 0:
            return {}
        step_time = elapsed / steps
        metrics = {
            "train/step_time_ms": step_time * 1000,
            "train/samples_per_sec": self.global_batch_size / step_time,
        }
        mfu = model_flops_utilization(self.flops_per_step, step_time,
                                      self.mesh.devices.size)
        if mfu is not None:
            metrics["train/mfu"] = mfu
        return metrics


    def fit(self, data, train_steps_per_epoch, epochs, train_step_args={}, val_steps_per_epoch=5, validation_step_args={}):
        local_batch_size = data.get('local_batch_size', 0)
        self.global_batch_size = data.get(
            'global_batch_size', local_batch_size * jax.process_count())
        train_ds = DevicePrefetchIterator(
            data['train'](), self.batch_sharding, source_state=self.dataset_state)
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
                self.best_state = self.get_np_tree(train_state)
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
                    
                
        self.save(epochs)
        self.wait_for_checkpoints()
        return self.state

    def wait_for_checkpoints(self):
        """Block until pending async checkpoint writes have landed on disk.

        Saving is async so it stays off the training loop's critical path;
        anything that reads the checkpoint back has to call this first.
        """
        self.checkpointer.wait_until_finished()
