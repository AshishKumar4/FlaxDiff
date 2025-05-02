import jax
import flax.linen as nn
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List, Tuple, Type

from flaxdiff.trainer import (
    SimpleTrainState,
    TrainState,
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
from flaxdiff.models.autoencoder import AutoEncoder
from flaxdiff.inputs import DiffusionInputConfig
from flaxdiff.utils import defaultTextEncodeModel, RandomMarkovState
from flaxdiff.samplers.euler import EulerAncestralSampler
from flaxdiff.inference.utils import parse_config, load_from_wandb_run, load_from_wandb_registry

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
    wandb_run = None
    samplers: Dict[Type[DiffusionSampler], Dict[float, DiffusionSampler]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_wandb_run(
        cls,
        wandb_run: str,
        project: str,
        entity: str,
    ):
        """Create an inference pipeline from a wandb run.
        
        Args:
            wandb_run: Run ID or display name
            project: Wandb project name
            entity: Wandb entity name
            wandb_modelname: Model name in wandb registry (if None, loads from checkpoint)
            checkpoint_step: Specific checkpoint step to load (if None, loads latest)
            config_overrides: Optional dictionary to override config values
            checkpoint_base_path: Base path for checkpoint storage
            
        Returns:
            DiffusionInferencePipeline instance
        """
        states, config, run = load_from_wandb_run(
            wandb_run,
            project=project,
            entity=entity,
        )
            
        if states is None:
            raise ValueError("Failed to load model parameters from wandb.")
        
        state, best_state = states
        parsed_config = parse_config(config)
        
        # Create the pipeline
        pipeline = cls.create(
            config=parsed_config,
            state=state,
            best_state=best_state,
            rngstate=RandomMarkovState(jax.random.PRNGKey(42)),
            run=run,
        )
        return pipeline
    
    @classmethod
    def from_wandb_registry(
        cls,
        modelname: str,
        project: str,
        entity: str = None,
        version: str = 'latest',
        registry: str = 'wandb-registry-model',
    ):
        """Create an inference pipeline from a wandb model registry.
        
        Args:
            modelname: Model name in wandb registry
            project: Wandb project name
            entity: Wandb entity name
            version: Version of the model to load (default is 'latest')
            registry: Registry name (default is 'wandb-registry-model')
            
        Returns:
            DiffusionInferencePipeline instance
        """
        states, config, run = load_from_wandb_registry(
            modelname=modelname,
            project=project,
            entity=entity,
            version=version,
            registry=registry,
        )
        
        if states is None:
            raise ValueError("Failed to load model parameters from wandb.")
        
        state, best_state = states
        parsed_config = parse_config(config)
        
        # Create the pipeline
        pipeline = cls.create(
            config=parsed_config,
            state=state,
            best_state=best_state,
            rngstate=RandomMarkovState(jax.random.PRNGKey(42)),
            run=run,
        )
        return pipeline
            
    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        state: Dict[str, Any],
        best_state: Optional[Dict[str, Any]] = None,
        rngstate: Optional[RandomMarkovState] = None,
        run=None,
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
            wandb_run=run,
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
        conditioning_data: List[Union[Tuple, Dict]] = None,
        conditioning_data_tokens: Tuple = None,
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
            conditioning=conditioning_data,
            model_conditioning_inputs=conditioning_data_tokens,
        )