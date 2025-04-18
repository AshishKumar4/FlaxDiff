import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import json
import wandb
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable, List, Tuple, Type

from flaxdiff.trainer import (
    SimpleTrainer,
    SimpleTrainState,
    TrainState,
    DiffusionTrainer,
)
from flaxdiff.samplers import (
    DiffusionSampler,
)
from flaxdiff.schedulers import (
    NoiseScheduler,
)
from flaxdiff.predictors import (
    DiffusionPredictionTransform,
)
from flaxdiff.models.common import kernel_init
from flaxdiff.models.simple_unet import Unet
from flaxdiff.models.simple_vit import UViT
from flaxdiff.models.general import BCHWModelWrapper
from flaxdiff.models.autoencoder import AutoEncoder
from flaxdiff.models.autoencoder.diffusers import StableDiffusionVAE
from flaxdiff.inputs import DiffusionInputConfig, ConditionalInputConfig
from flaxdiff.utils import defaultTextEncodeModel, RandomMarkovState
from flaxdiff.samplers.euler import EulerAncestralSampler

from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
import os
from .utils import get_wandb_run, parse_config

@dataclass
class InferencePipeline:
    """Inference pipeline for a general model."""
    model: nn.Module = None
    state: SimpleTrainState = None
    best_state: SimpleTrainState = None
    
    def from_wandb(
        self,
        wandb_run: str,
        wandb_project: str,
        wandb_entity: str,
    ):
        raise NotImplementedError("InferencePipeline does not support from_wandb.")    
    
def get_latest_checkpoint(checkpoint_path):
    checkpoint_files = os.listdir(checkpoint_path)
    # Sort files by step number
    checkpoint_files = sorted([int(i) for i in checkpoint_files])
    latest_step = checkpoint_files[-1]
    latest_checkpoint = os.path.join(checkpoint_path, str(latest_step))
    return latest_checkpoint

def load_from_checkpoint(
    checkpoint_dir: str,
):
    try:
        checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(create=False)
        # Convert checkpoint_dir to absolute path
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        manager = CheckpointManager(checkpoint_dir, checkpointer, options)
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

def load_from_wandb(
    wandb_project: str,
    wandb_modelname: str,
    wandb_entity: str = None,
    version: str = 'latest',
    wandb_run_id: str = None,
):
    """
    Loads model from wandb model registry.
    """
    # Get the model version from wandb
    try:
        run = wandb.init(
            id=wandb_run_id,
            project=wandb_project,
            entity=wandb_entity,
            resume='allow'
        )
        print(f"Loading model from wandb: {wandb_modelname}:{version}")
        ckpt_dir = run.use_model(f"{wandb_modelname}:{version}")
        print(f"Loaded model from wandb: {wandb_modelname}:{version} at path {ckpt_dir}")
        # Load the model from the checkpoint directory
        states = load_from_checkpoint(ckpt_dir)
    except Exception as e:
        print(f"Warning: Failed to load model from wandb: {e}")
        states = None, None
    return states

@dataclass
class DiffusionInferencePipeline(InferencePipeline):
    """Inference pipeline for diffusion models.
    
    This pipeline handles loading models from wandb and generating samples using the
    DiffusionSampler from FlaxDiff.
    """
    state: TrainState = None
    best_state: TrainState = None
    rngstate: Optional[RandomMarkovState] = None
    noise_schedule: NoiseScheduler = None
    model_output_transform: DiffusionPredictionTransform = None
    autoencoder: AutoEncoder = None
    input_config: DiffusionInputConfig = None
    samplers: Dict[Type[DiffusionSampler], Dict[float, DiffusionSampler]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_wandb(
        cls,
        wandb_run: str,
        wandb_project: str = "mlops-msml605-project",
        wandb_entity: str = "umd-projects",
        wandb_modelname: str = None,
        checkpoint_step: int = None,
        checkpoint_base_path: str = None,
        config_overrides: Dict = None,
    ):
        """Create an inference pipeline from a wandb run.
        
        Args:
            wandb_run: Run ID or display name
            wandb_project: Wandb project name
            wandb_entity: Wandb entity name
            wandb_modelname: Model name in wandb registry (if None, loads from checkpoint)
            checkpoint_step: Specific checkpoint step to load (if None, loads latest)
            config_overrides: Optional dictionary to override config values
            checkpoint_base_path: Base path for checkpoint storage
            
        Returns:
            DiffusionInferencePipeline instance
        """
        # Get wandb run
        run = get_wandb_run(wandb_run, wandb_project, wandb_entity)
        if run is None:
            raise ValueError(f"Run {wandb_run} not found in project {wandb_project}.")
        
        # Load run configuration
        run_config = run.config
        
        # Parse the configuration
        parsed_config = parse_config(run_config, config_overrides)
        
        # Load model
        if checkpoint_base_path is not None and os.path.exists(checkpoint_base_path):
            print(f"Loading model from wandb run {wandb_run} with config: {parsed_config}")
            if checkpoint_step is None:
                checkpoint_dir = get_latest_checkpoint(checkpoint_base_path)
            else:
                checkpoint_dir = os.path.join(checkpoint_base_path, checkpoint_step)
                
            state, best_state = load_from_checkpoint(checkpoint_dir)
        else:
            if wandb_modelname is None:
                raise ValueError("No wandb_modelname provided and checkpoint_base_path is None.")
            print(f"Loading model from wandb model registry {wandb_modelname}")
            state, best_state = load_from_wandb(
                wandb_project=wandb_project,
                wandb_modelname=wandb_modelname,
                wandb_entity=wandb_entity,
                wandb_run_id=run.id,
            )
            
        if state is None:
            raise ValueError("Failed to load model parameters from wandb.")
        # Create the pipeline
        pipeline = cls.create(
            config=parsed_config,
            state=state,
            best_state=best_state,
            rngstate=RandomMarkovState(jax.random.PRNGKey(42)),
        )
        return pipeline
            
    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        state: Dict[str, Any],
        best_state: Optional[Dict[str, Any]] = None,
        rngstate: Optional[RandomMarkovState] = None,
    ):
        if rngstate is None:
            rngstate = RandomMarkovState(jax.random.PRNGKey(42))
        # Build and return pipeline
        return cls(
            model=config['model'],
            state=state,
            best_state=best_state,
            rngstate=rngstate,
            noise_schedule=config['noise_schedule'],
            model_output_transform=config['prediction_transform'],
            autoencoder=config['autoencoder'],
            input_config=config['input_config'],
            config=config,
        )
    
    def get_sampler(
        self, 
        guidance_scale: float = 3.0,
        sampler_class=EulerAncestralSampler, 
    ) -> DiffusionSampler:
        """Get (or create) a sampler for generating samples.
        
        This method caches samplers by their class and guidance scale for reuse.
        
        Args:
            sampler_class: Class for the diffusion sampler
            guidance_scale: Classifier-free guidance scale (0.0 to disable)
            
        Returns:
            DiffusionSampler instance
        """
        # Get or create dictionary for this sampler class
        if sampler_class not in self.samplers:
            self.samplers[sampler_class] = {}
        
        # Check if we already have a sampler with this guidance scale
        if guidance_scale not in self.samplers[sampler_class]:
            # Create unconditional embeddings if using guidance
            null_embeddings = None
            if guidance_scale > 0.0:
                null_text = self.input_config.conditions[0].get_unconditional()
                null_embeddings = null_text
                print(f"Created null embeddings for guidance with shape {null_embeddings.shape}")
            
            # Create and cache the sampler
            self.samplers[sampler_class][guidance_scale] = sampler_class(
                model=self.model,
                noise_schedule=self.noise_schedule,
                model_output_transform=self.model_output_transform,
                guidance_scale=guidance_scale,
                input_config=self.input_config,
                autoencoder=self.autoencoder,
            )
        
        return self.samplers[sampler_class][guidance_scale]
    
    def generate_samples(
        self,
        num_samples: int,
        resolution: int,
        conditioning_data: Optional[List[Union[Tuple, Dict]]] = None,  # one list per modality or list of tuples
        sequence_length: Optional[int] = None,
        diffusion_steps: int = 50,
        guidance_scale: float = 1.0,
        sampler_class=EulerAncestralSampler,
        timestep_spacing: str = 'linear',
        seed: Optional[int] = None,
        start_step: Optional[int] = None,
        end_step: int = 0,
        steps_override=None,
        priors=None,
        use_best_params: bool = False,
        use_ema: bool = False,
    ):
        # Setup RNG
        rngstate = self.rngstate or RandomMarkovState(jax.random.PRNGKey(seed or 0))
        
        # Get cached or new sampler
        sampler = self.get_sampler(
            guidance_scale=guidance_scale,
            sampler_class=sampler_class,
        )
        if hasattr(sampler, 'timestep_spacing'):
            sampler.timestep_spacing = timestep_spacing
        print(f"Generating samples: steps={diffusion_steps}, num_samples={num_samples}, guidance={guidance_scale}")
        
        if use_best_params:
            state = self.best_state
        else:
            state = self.state
            
        if use_ema:
            params = state['ema_params']
        else:
            params = state['params']
            
             
        return sampler.generate_samples(
            params=params,
            num_samples=num_samples,
            resolution=resolution,
            sequence_length=sequence_length,
            diffusion_steps=diffusion_steps,
            start_step=start_step,
            end_step=end_step,
            steps_override=steps_override,
            priors=priors,
            rngstate=rngstate,
            conditioning=conditioning_data
        )