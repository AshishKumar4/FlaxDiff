import jax
import jax.numpy as jnp
import json
from flaxdiff.schedulers import (
    CosineNoiseScheduler,
    KarrasVENoiseScheduler,
)
from flaxdiff.predictors import (
    VPredictionTransform,
    KarrasPredictionTransform,
)
from flaxdiff.models.common import kernel_init
from flaxdiff.models.simple_unet import Unet
from flaxdiff.models.simple_vit import UViT
from flaxdiff.models.general import BCHWModelWrapper
from flaxdiff.models.autoencoder.diffusers import StableDiffusionVAE
from flaxdiff.inputs import DiffusionInputConfig, ConditionalInputConfig
from flaxdiff.utils import defaultTextEncodeModel
from diffusers import FlaxUNet2DConditionModel
import wandb
from flaxdiff.models.simple_unet import Unet
from flaxdiff.models.simple_vit import UViT
from flaxdiff.models.general import BCHWModelWrapper
from flaxdiff.models.autoencoder.diffusers import StableDiffusionVAE
from flaxdiff.inputs import DiffusionInputConfig, ConditionalInputConfig
from flaxdiff.utils import defaultTextEncodeModel

from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
import os

import warnings

def get_wandb_run(wandb_run: str, project, entity):
    """
    Try to get the wandb run for the given experiment name and project.
    Return None if not found.
    """
    import wandb
    wandb_api = wandb.Api()
    # First try to get the run by treating wandb_run as a run ID
    try:
        run = wandb_api.run(f"{entity}/{project}/{wandb_run}")
        print(f"Found run: {run.name} ({run.id})")
        return run
    except wandb.Error as e:
        print(f"Run not found by ID: {e}")
        # If that fails, try to get the run by treating wandb_run as a display name
        # This is a bit of a hack, but it works for now.
        # Note: this will return all runs with the same display name, so be careful.
        print(f"Trying to get run by display name: {wandb_run}")
    runs = wandb_api.runs(path=f"{entity}/{project}", filters={"displayName": wandb_run})
    for run in runs:
        print(f"Found run: {run.name} ({run.id})")
        return run
    return None

def parse_config(config, overrides=None):
    """Parse configuration for inference pipeline.
    
    Args:
        config: Configuration dictionary from wandb run
        overrides: Optional dictionary of overrides for config parameters
        
    Returns:
        Dictionary containing model, sampler, scheduler, and other required components
        including DiffusionInputConfig for the general diffusion framework
    """
    warnings.filterwarnings("ignore")
    
    # Merge config with overrides if provided
    if overrides is not None:
        # Create a deep copy of config to avoid modifying the original
        merged_config = dict(config)
        # Update arguments with overrides
        if 'arguments' in merged_config:
            merged_config['arguments'] = {**merged_config['arguments'], **overrides}
            # Also update top-level config for key parameters
            for key in overrides:
                if key in merged_config:
                    merged_config[key] = overrides[key]
    else:
        merged_config = config
    
    # Parse configuration from config dict
    conf = merged_config
    
    # Setup mappings for dtype, precision, and activation
    DTYPE_MAP = {
        'bfloat16': jnp.bfloat16,
        'float32': jnp.float32,
        'jax.numpy.float32': jnp.float32,
        'jax.numpy.bfloat16': jnp.bfloat16,
        'None': None,
        None: None,
    }
    
    PRECISION_MAP = {
        'high': jax.lax.Precision.HIGH,
        'HIGH': jax.lax.Precision.HIGH,
        'default': jax.lax.Precision.DEFAULT,
        'DEFAULT': jax.lax.Precision.DEFAULT,
        'highest': jax.lax.Precision.HIGHEST,
        'HIGHEST': jax.lax.Precision.HIGHEST,
        'None': None,
        None: None,
    }
    
    ACTIVATION_MAP = {
        'swish': jax.nn.swish,
        'silu': jax.nn.silu,
        'jax._src.nn.functions.silu': jax.nn.silu,
        'mish': jax.nn.mish,
    }
    
    # Get model class based on architecture
    MODEL_CLASSES = {
        'unet': Unet,
        'uvit': UViT,
        'diffusers_unet_simple': FlaxUNet2DConditionModel
    }
    
    # Map all the leaves of the model config, converting strings to appropriate types
    def map_nested_config(config):
        new_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                new_config[key] = map_nested_config(value)
            elif isinstance(value, list):
                new_config[key] = [map_nested_config(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, str):
                if value in DTYPE_MAP:
                    new_config[key] = DTYPE_MAP[value]
                elif value in PRECISION_MAP:
                    new_config[key] = PRECISION_MAP[value]
                elif value in ACTIVATION_MAP:
                    new_config[key] = ACTIVATION_MAP[value]
                elif value == 'None':
                    new_config[key] = None
                elif '.'in value:
                    # Ignore any other string that contains a dot
                    print(f"Ignoring key {key} with value {value} as it contains a dot.")
                else:
                    new_config[key] = value
            else:
                new_config[key] = value
        return new_config

    # Parse architecture and model config
    model_config = conf['model']
        
    # Get architecture type
    architecture = conf.get('architecture', conf.get('arguments', {}).get('architecture', 'unet'))
        
    # Handle autoencoder
    autoencoder_name = conf.get('autoencoder', conf.get('arguments', {}).get('autoencoder'))
    autoencoder_opts_str = conf.get('autoencoder_opts', conf.get('arguments', {}).get('autoencoder_opts', '{}'))
    autoencoder = None
    autoencoder_opts = None
    
    if autoencoder_name:
        print(f"Using autoencoder: {autoencoder_name}")
        if isinstance(autoencoder_opts_str, str):
            autoencoder_opts = json.loads(autoencoder_opts_str)
        else:
            autoencoder_opts = autoencoder_opts_str
            
        if autoencoder_name == 'stable_diffusion':
            print("Using Stable Diffusion Autoencoder for Latent Diffusion Modeling")
            autoencoder_opts = map_nested_config(autoencoder_opts)
            autoencoder = StableDiffusionVAE(**autoencoder_opts)
            
    input_config = conf.get('input_config', None)
    
    # If not provided, create one based on the older format (backward compatibility)
    if input_config is None:
        # Warn if input_config is not provided
        print("No input_config provided, creating a default one.")
        image_size = conf['arguments'].get('image_size', 128)
        image_channels = 3  # Default number of channels
        # Create text encoder
        text_encoder = defaultTextEncodeModel()
        # Create a conditional input config for text conditioning
        text_conditional_config = ConditionalInputConfig(
            encoder=text_encoder,
            conditioning_data_key='text',
            pretokenized=True,
            unconditional_input="",
            model_key_override="textcontext"
        )
        
        # Create the main input config
        input_config = DiffusionInputConfig(
            sample_data_key='image',
            sample_data_shape=(image_size, image_size, image_channels),
            conditions=[text_conditional_config]
        )
    else:
        # Deserialize the input config if it's a string
        input_config = DiffusionInputConfig.deserialize(input_config)
    
    model_kwargs = map_nested_config(model_config)
    
    print(f"Model kwargs after mapping: {model_kwargs}")
    
    model_class = MODEL_CLASSES.get(architecture)
    if not model_class:
        raise ValueError(f"Unknown architecture: {architecture}. Supported architectures: {', '.join(MODEL_CLASSES.keys())}")
    
    # Instantiate the model
    model = model_class(**model_kwargs)
    
    # If using diffusers UNet, wrap it for consistent interface
    if 'diffusers' in architecture:
        model = BCHWModelWrapper(model)
    
    # Create noise scheduler based on configuration
    noise_schedule_type = conf.get('noise_schedule', conf.get('arguments', {}).get('noise_schedule', 'edm'))
    if noise_schedule_type in ['edm', 'karras']:
        # For both EDM and karras, we use the karras scheduler for inference
        noise_schedule = KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)
        prediction_transform = KarrasPredictionTransform(sigma_data=noise_schedule.sigma_data)
    elif noise_schedule_type == 'cosine':
        noise_schedule = CosineNoiseScheduler(1000, beta_end=1)
        prediction_transform = VPredictionTransform()
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule_type}")
    
    # Prepare return dictionary with all components
    result = {
        'model': model,
        'model_config': model_kwargs,
        'architecture': architecture,
        'autoencoder': autoencoder,
        'noise_schedule': noise_schedule,
        'prediction_transform': prediction_transform,
        'input_config': input_config,
        'raw_config': conf,
    }
    
    return result

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
    
def load_from_wandb_run(
    run,
    project: str,
    entity: str = None,
):
    """
    Loads model from wandb model registry.
    """
    # Get the model version from wandb
    states = None
    config = None
    try:
        if isinstance(run, str):
            run = get_wandb_run(run, project, entity)
        # Search for model artifact
        models = [i for i in run.logged_artifacts() if i.type == 'model']
        if len(models) == 0:
            raise ValueError(f"No model artifacts found in run {run.id}")
        # Pick out any model artifact
        highest_version = max([{'version':int(i.version[1:]), 'name': i.qualified_name} for i in models], key=lambda x: x['version'])
        wandb_modelname = highest_version['name']
        
        print(f"Loading model from wandb: {wandb_modelname} out of versions {[i.version for i in models]}")
        artifact = run.use_artifact(wandb.Api().artifact(wandb_modelname))
        ckpt_dir = artifact.download()
        print(f"Loaded model from wandb: {wandb_modelname} at path {ckpt_dir}")
        # Load the model from the checkpoint directory
        states = load_from_checkpoint(ckpt_dir)
        config = run.config
    except Exception as e:
        print(f"Warning: Failed to load model from wandb: {e}")
    return states, config, run, artifact

def load_from_wandb_registry(
    modelname: str,
    project: str,
    entity: str = None,
    version: str = 'latest',
    registry: str = 'wandb-registry-model',
):
    """
    Loads model from wandb model registry.
    """
    # Get the model version from wandb
    states = None
    config = None
    run = None
    try:
        artifact = wandb.Api().artifact(f"{registry}/{modelname}:{version}")
        ckpt_dir = artifact.download()
        print(f"Loaded model from wandb registry: {modelname} at path {ckpt_dir}")
        # Load the model from the checkpoint directory
        states = load_from_checkpoint(ckpt_dir)
        run = artifact.logged_by()
        config = run.config
    except Exception as e:
        print(f"Warning: Failed to load model from wandb: {e}")
    return states, config, run, artifact