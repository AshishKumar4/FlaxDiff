import json
import numpy as np
import flax
from flax import linen as nn
import jax
from typing import Callable, List, Dict, Tuple, Union, Any, Sequence, Optional
from dataclasses import field, dataclass
import jax.numpy as jnp
import optax
import functools
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from ..schedulers import NoiseScheduler
from ..predictors import DiffusionPredictionTransform, EpsilonPredictionTransform

from flaxdiff.utils import RandomMarkovState, serialize_model, get_latest_checkpoint
from flaxdiff.inputs import ConditioningEncoder, ConditionalInputConfig, DiffusionInputConfig

from .simple_trainer import SimpleTrainer, SimpleTrainState, Metrics, convert_to_global_tree
from .objectives import Objective, DiffusionObjective

from flaxdiff.models.autoencoder.autoencoder import AutoEncoder
from flax.training import dynamic_scale as dynamic_scale_lib
import shutil


def _replace_subtree(tree, path, value):
    if not path:
        return value
    return {**tree, path[0]: _replace_subtree(tree[path[0]], path[1:], value)}


def _subtree(tree, path):
    for key in path:
        tree = tree[key]
    return tree


class TrainState(SimpleTrainState):
    rngs: jax.random.PRNGKey
    ema_params: dict

    def apply_ema(self, decay: float = 0.999, path: Tuple[str, ...] = ()):
        """EMA over a subtree of the parameters (the whole tree by default).

        JEPA's target encoder is the EMA of the context encoder alone, so the
        predictor's slice of the tree must be left out of the average.
        """
        new_subtree = jax.tree_util.tree_map(
            lambda ema, param: decay * ema + (1 - decay) * param,
            _subtree(self.ema_params, path),
            _subtree(self.params, path),
        )
        return self.replace(ema_params=_replace_subtree(self.ema_params, path, new_subtree))


from flaxdiff.metrics.common import EvaluationMetric

class GeneralDiffusionTrainer(SimpleTrainer):
    """
    General trainer for diffusion models supporting both images and videos.
    
    Extends DiffusionTrainer to support:
    1. Both image data (4D tensors: B,H,W,C) and video data (5D tensors: B,T,H,W,C)
    2. Multiple conditioning inputs
    3. Various model architectures
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: optax.GradientTransformation,
                 input_config: DiffusionInputConfig,
                 rngs: jax.random.PRNGKey,
                 noise_schedule: NoiseScheduler = None,
                 objective: Objective = None,
                 unconditional_prob: float = 0.12,
                 name: str = "GeneralDiffusion",
                 model_output_transform: DiffusionPredictionTransform = EpsilonPredictionTransform(),
                 autoencoder: AutoEncoder = None,
                 native_resolution: int = None,
                 frames_per_sample: int = None,
                 wandb_config: Dict[str, Any] = None,
                 eval_metrics: List[EvaluationMetric] = None,
                 best_tracker_metric: str = "train/best_loss",
                 ema_decay: float = 0.999,
                 loss_fn: Callable = optax.l2_loss,
                 **kwargs
                 ):
        """
        Initialize the general diffusion trainer.

        Args:
            model: Neural network model
            optimizer: Optimization algorithm
            input_config: Configuration for input data, including keys, shapes and conditioning inputs
            rngs: Random number generator keys
            noise_schedule: Noise scheduler for the default diffusion objective
            objective: What to optimize. Defaults to diffusion over `model`.
            unconditional_prob: Probability of training with unconditional samples
            name: Name of this trainer
            model_output_transform: Transform for model predictions
            autoencoder: Optional autoencoder for latent diffusion
            native_resolution: Native resolution of the data
            frames_per_sample: Number of frames per video sample (for video only)
            **kwargs: Additional arguments for parent class
        """
        # Initialize with parent DiffusionTrainer but without encoder parameter
        input_shapes = input_config.get_input_shapes(
            autoencoder=autoencoder,
        )
        self.input_config = input_config
        self.eval_metrics = eval_metrics

        self.noise_schedule = noise_schedule
        self.model_output_transform = model_output_transform
        self.unconditional_prob = unconditional_prob
        self.autoencoder = autoencoder
        self.ema_decay = ema_decay

        if native_resolution is None:
            sample_shape = input_config.sample_data_shape
            native_resolution = sample_shape[-2]
            if autoencoder is not None:
                native_resolution = native_resolution * autoencoder.downscale_factor
        self.native_resolution = native_resolution

        if objective is None:
            objective = DiffusionObjective(
                model=model,
                noise_schedule=noise_schedule,
                model_output_transform=model_output_transform,
                input_config=input_config,
                input_shapes=input_shapes,
                autoencoder=autoencoder,
                unconditional_prob=unconditional_prob,
                loss_fn=loss_fn,
                ema_decay=ema_decay,
                native_resolution=native_resolution,
            )
        self.objective = objective

        if wandb_config is not None:
            # If input_config is not in wandb_config, add it
            if 'input_config' not in wandb_config['config']:
                wandb_config['config']['input_config'] = input_config.serialize()
            # If model is not in wandb_config, add it
            if 'model' not in wandb_config['config']:
                wandb_config['config']['model'] = serialize_model(model)
            if 'autoencoder' not in wandb_config['config'] and autoencoder is not None:
                wandb_config['config']['autoencoder'] = autoencoder.name
                wandb_config['config']['autoencoder_opts'] = json.dumps(autoencoder.serialize())

            # Names the wandb model artifact, so runs of different objectives on
            # the same dataset and resolution do not collide
            dataset_name = wandb_config['config']['arguments']['dataset']
            self.modelname = (
                f"{objective.tag}-{dataset_name}-res{input_config.sample_data_shape[-2]}")
            print("Model name:", self.modelname)
            wandb_config['config']['modelname'] = self.modelname

        super().__init__(
            model=model,
            input_shapes=input_shapes,
            optimizer=optimizer,
            rngs=rngs,
            name=name,
            wandb_config=wandb_config,
            loss_fn=loss_fn,
            **kwargs
        )
        
        self.best_tracker_metric = best_tracker_metric
        
        # Store video-specific parameters
        self.frames_per_sample = frames_per_sample
        
        # List of conditional inputs
        self.conditional_inputs = input_config.conditions
        self.best_val_metrics = {}

        # val/<metric_name> -> True if higher is better (default lower-is-better)
        self.metric_higher_is_better = {}
        if eval_metrics is not None:
            for m in eval_metrics:
                self.metric_higher_is_better[f"val/{m.name}"] = getattr(m, 'higher_is_better', False)
    
    def generate_states(
        self,
        optimizer: optax.GradientTransformation,
        rngs: jax.random.PRNGKey,
        model: nn.Module = None,
        use_dynamic_scale: bool = False
    ) -> Tuple[TrainState, TrainState]:
        print("Generating states for DiffusionTrainer")
        rngs, subkey = jax.random.split(rngs)

        params = self.objective.init_params(subkey)

        state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            ema_params=params,
            tx=optimizer,
            rngs=rngs,
            metrics=Metrics.empty(),
            dynamic_scale = dynamic_scale_lib.DynamicScale() if use_dynamic_scale else None
        )
        return state, state

    def fit(self, data, training_steps_per_epoch, epochs, val_steps_per_epoch=8, **validation_step_args):
        local_batch_size = data['local_batch_size']
        return super().fit(
            data,
            train_steps_per_epoch=training_steps_per_epoch,
            epochs=epochs,
            train_step_args={"batch_size": local_batch_size},
            val_steps_per_epoch=val_steps_per_epoch,
            validation_step_args=validation_step_args,
        )

    def _define_train_step(self, batch_size):
        """
        Define the training step: the objective supplies the loss, the trainer
        supplies gradients, sharding, EMA and the state update.
        """
        objective = self.objective
        distributed_training = self.distributed_training
        ema = objective.ema

        def train_step(train_state: TrainState, rng_state: RandomMarkovState, batch, local_device_index):
            """Training step optimized for distributed execution."""
            rng_state, key_fold = rng_state.get_random_key()
            folded_key = jax.random.fold_in(key_fold, local_device_index.reshape())

            def objective_loss(params):
                return objective.loss(params, train_state.ema_params, batch,
                                      folded_key, train_state.step)

            if train_state.dynamic_scale is not None:
                # Mixed precision training with dynamic scale
                grad_fn = train_state.dynamic_scale.value_and_grad(
                    objective_loss, has_aux=True, axis_name="data")
                dynamic_scale, is_finite, (loss, aux), grads = grad_fn(train_state.params)

                train_state = train_state.replace(dynamic_scale=dynamic_scale)
                new_state = train_state.apply_gradients(grads=grads)

                # Handle NaN/Inf gradients
                select_fn = functools.partial(jnp.where, is_finite)
                new_state = new_state.replace(
                    opt_state=jax.tree.map(select_fn, new_state.opt_state, train_state.opt_state),
                    params=jax.tree.map(select_fn, new_state.params, train_state.params)
                )
            else:
                grad_fn = jax.value_and_grad(objective_loss, has_aux=True)
                (loss, aux), grads = grad_fn(train_state.params)

                if distributed_training:
                    grads = jax.lax.pmean(grads, axis_name="data")

                new_state = train_state.apply_gradients(grads=grads)

            new_state = new_state.apply_ema(ema.decay(train_state.step), ema.path)

            if distributed_training:
                loss = jax.lax.pmean(loss, axis_name="data")
                aux = jax.tree.map(lambda v: jax.lax.pmean(v, axis_name="data"), aux)

            return new_state, loss, aux, rng_state

        # Apply sharding for distributed training
        if distributed_training:
            train_step = shard_map(
                train_step,
                mesh=self.mesh,
                in_specs=(P(), P(), P('data'), P('data')),
                out_specs=(P(), P(), P(), P()),
            )

        # Apply JIT compilation
        train_step = jax.jit(train_step, donate_argnums=(2))
        return train_step

    def _define_validation_step(self, **kwargs):
        return self.objective.make_validation_step(**kwargs)

    def validation_loop(
        self,
        val_state: SimpleTrainState,
        val_step_fn: Callable,
        val_ds,
        val_steps_per_epoch,
        current_step,
    ):
        """
        Score the objective's validation artifacts and let it visualize them.
        """
        global_device_count = jax.device_count()
        local_device_count = jax.local_device_count()
        process_index = jax.process_index()
        generate_samples = val_step_fn
        
        val_ds = iter(val_ds()) if val_ds else None
        print(f"Validation loop started for process index {process_index} with {global_device_count} devices.")
        # Evaluation step
        try:
            metrics = {metric.name: [] for metric in self.eval_metrics} if self.eval_metrics else {}
            for i in range(val_steps_per_epoch):
                if val_ds is None:
                    batch = None
                else:
                    batch = next(val_ds)
                    if self.distributed_training and global_device_count > 1:
                        batch = convert_to_global_tree(self.mesh, batch)
                samples = generate_samples(val_state, batch)

                if self.eval_metrics is not None:
                    for metric in self.eval_metrics:
                        try:
                            # Evaluate metrics
                            metric_val = metric.function(samples, batch)
                            metrics[metric.name].append(metric_val)
                        except Exception as e:
                            print("Error in evaluation metrics:", e)
                            import traceback
                            traceback.print_exc()
                            pass
                    
                if i == 0:
                    print(f"Evaluation started for process index {process_index}")
                    if self.wandb is not None and self.wandb:
                        self.objective.log_validation_artifacts(
                            self.wandb, samples, current_step)

            # Flatten the metrics
            if metrics:
                metrics = {k: np.mean(v) for k, v in metrics.items()}
                # Update the best validation metrics (min or max per metric direction)
                for key, value in metrics.items():
                    final_key = f"val/{key}"
                    higher_is_better = self.metric_higher_is_better.get(final_key, False)
                    if final_key not in self.best_val_metrics:
                        self.best_val_metrics[final_key] = value
                    else:
                        prev = self.best_val_metrics[final_key]
                        self.best_val_metrics[final_key] = max(prev, value) if higher_is_better else min(prev, value)
                # Log the best validation metrics
                if self.wandb is not None and self.wandb:
                    # Log the metrics
                    for key, value in metrics.items():
                        if isinstance(value, jnp.ndarray):
                            value = np.array(value)
                        self.wandb.log({
                            f"val/{key}": value,
                        }, step=current_step)
                    # Log the best validation metrics
                    for key, value in self.best_val_metrics.items():
                        if isinstance(value, jnp.ndarray):
                            value = np.array(value)
                        self.wandb.log({
                            f"best_{key}": value,
                        }, step=current_step)
                print(f"Validation metrics for process index {process_index}: {metrics}")
            
            # Close validation dataset iterator
            del val_ds
        except StopIteration:
            print(f"Validation dataset exhausted for process index {process_index}")
        except Exception as e:
            print(f"Error during validation for process index {process_index}: {e}")
            import traceback
            traceback.print_exc()


    def push_to_registry(
        self,
        registry_name: str = 'wandb-registry-model',
        aliases: List[str] = [],
    ):
        """
        Push the model to wandb registry.
        Args:
            registry_name: Name of the model registry.
            aliases: List of aliases for the model.
        """
        if self.wandb is None:
            raise ValueError("Wandb is not initialized. Cannot push to registry.")
        
        modelname = self.modelname
        if hasattr(self, "wandb_sweep"):
            modelname = f"{modelname}-sweep-{self.wandb_sweep.id}"
        
        latest_checkpoint_path = get_latest_checkpoint(self.checkpoint_path())
        logged_artifact = self.wandb.log_artifact(
            artifact_or_path=latest_checkpoint_path,
            name=modelname,
            type="model",
            aliases=['latest'] + aliases,
        )
        
        target_path = f"{registry_name}/{modelname}"
        
        self.wandb.link_artifact(
            artifact=logged_artifact,
            target_path=target_path,
            aliases=aliases,
        )
        print(f"Model pushed to registry at {target_path}")
        return logged_artifact
    
    def __get_best_sweep_runs__(
        self,
        metric: str = "train/best_loss",
        top_k: int = 5,
    ):
        """
        Get the best runs from a wandb sweep.
        Args:
            metric: Metric to sort by.
            top_k: Number of top runs to return.
        """
        if self.wandb is None:
            raise ValueError("Wandb is not initialized. Cannot get best runs.")
        
        if not hasattr(self, "wandb_sweep"):
            raise ValueError("Wandb sweep is not initialized. Cannot get best runs.")
        
        print(f"Getting best runs from sweep {self.wandb_sweep.id}...")
        # Get the sweep runs
        runs = sorted(self.wandb_sweep.runs, key=lambda x: x.summary.get(metric, float('inf')))
        best_runs = runs[:top_k]
        lower_bound = best_runs[-1].summary.get(metric, float('inf'))
        upper_bound = best_runs[0].summary.get(metric, float('inf'))
        print(f"Best runs from sweep {self.wandb_sweep.id}:")
        for run in best_runs:
            print(f"\t\tRun ID: {run.id}, Metric: {run.summary.get(metric, float('inf'))}")
        return best_runs, (min(lower_bound, upper_bound), max(lower_bound, upper_bound))
    
    def __get_best_general_runs__(
        self,
        metric: str = "train/best_loss",
        top_k: int = 5,
    ):
        """
        Get the best runs from wandb.
        Args:
            metric: Metric to sort by.
            top_k: Number of top runs to return.
        """
        if self.wandb is None:
            raise ValueError("Wandb is not initialized. Cannot get best runs.")
        import wandb
        # Get the sweep runs
        runs = [i for i in wandb.Api().runs(path=f"{self.wandb.entity}/{self.wandb.project}", filters={"config.dataset.name": self.wandb.config['dataset']['name']})]
        if not runs:
            raise ValueError("No runs found in wandb.")
        print(f"Getting best runs from wandb {self.wandb.id}...")

        # sort descending for higher-is-better metrics so top_k has the best runs
        higher_is_better = self.metric_higher_is_better.get(metric, False)
        if higher_is_better:
            sort_key = lambda x: x.summary.get(f"best_{metric}", float('-inf'))
            runs = sorted(runs, key=sort_key, reverse=True)
        else:
            sort_key = lambda x: x.summary.get(f"best_{metric}", float('inf'))
            runs = sorted(runs, key=sort_key)

        best_runs = runs[:top_k]
        default = float('-inf') if higher_is_better else float('inf')
        lower_bound = best_runs[-1].summary.get(f"best_{metric}", default)
        upper_bound = best_runs[0].summary.get(f"best_{metric}", default)
        print(f"Best runs from wandb {self.wandb.id}:")
        for run in best_runs:
            print(f"\t\tRun ID: {run.id}, Metric: {run.summary.get(metric, default)}")
        return best_runs, (min(lower_bound, upper_bound), max(lower_bound, upper_bound))
    
    def __compare_run_against_best__(self, top_k=2, metric="train/best_loss", from_sweeps=False):
        """
        Compare the current run against the best runs from wandb.
        Args:
            top_k: Number of top runs to consider.
            metric: Metric to compare against.
            from_sweeps: Whether to consider runs from sweeps.
        Returns:
            is_good: Whether the current run is among the best.
            is_best: Whether the current run is the best.
        """
        # Get best runs
        if from_sweeps:
            best_runs, bounds = self.__get_best_sweep_runs__(metric=metric, top_k=top_k)
        else:
            best_runs, bounds = self.__get_best_general_runs__(metric=metric, top_k=top_k)

        # use the metric's declared direction; for losses lower is always better
        if metric in self.metric_higher_is_better:
            is_lower_better = not self.metric_higher_is_better[metric]
        else:
            is_lower_better = True

        # Check if current run is one of the best
        if metric == "train/best_loss":
            current_run_metric = self.best_loss
        elif metric in self.best_val_metrics:
            print(f"Fetching best validation metric {metric} from local")
            current_run_metric = self.best_val_metrics[metric]
        else:
            current_run_metric = self.wandb.summary.get(metric, float('inf') if is_lower_better else float('-inf'))

        print(f"Current run {self.wandb.id} metric: {current_run_metric}, Best bounds: {bounds}")
        # Check based on bounds
        if (is_lower_better and current_run_metric < bounds[1]) or (not is_lower_better and current_run_metric > bounds[0]):
            print(f"Current run {self.wandb.id} meets performance criteria. Current metric: {current_run_metric}, Best bounds: {bounds}")
            is_best = (is_lower_better and current_run_metric < bounds[0]) or (not is_lower_better and current_run_metric > bounds[1])
            return True, is_best

        return False, False
            
    def save(self, epoch=0, step=0, state=None, rngstate=None):
        super().save(epoch=epoch, step=step, state=state, rngstate=rngstate)
        
        if self.wandb is not None:
            checkpoint = get_latest_checkpoint(self.checkpoint_path())
            try:
                is_good, is_best = self.__compare_run_against_best__(top_k=5, metric=self.best_tracker_metric, from_sweeps=hasattr(self, "wandb_sweep"))
                if is_good:
                    # Push to registry with appropriate aliases
                    aliases = []
                    if is_best:
                        aliases.append("best")
                    self.push_to_registry(aliases=aliases)
                    print("Model pushed to registry successfully with aliases:", aliases)
                    # Only delete after a successful registry push - the local
                    # checkpoint is the only copy otherwise
                    shutil.rmtree(checkpoint, ignore_errors=True)
                    print(f"Checkpoint deleted at {checkpoint}")
                else:
                    print("Current run is not one of the best runs. Keeping local checkpoint.")
            except Exception as e:
                print(f"Error during registry operations: {e}")
                print(f"Checkpoint preserved at {checkpoint}")
