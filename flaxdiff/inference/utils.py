import jax
import jax.numpy as jnp
import json
import os
import warnings

import wandb
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer

from flaxdiff.predictors import get_diffusion_preset
from flaxdiff.models.registry import build_model, canonicalize_architecture, map_config_strings
from flaxdiff.models.autoencoder.diffusers import StableDiffusionVAE
from flaxdiff.inputs import DiffusionInputConfig, ConditionalInputConfig
from flaxdiff.utils import defaultTextEncodeModel

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
            autoencoder_opts = map_config_strings(autoencoder_opts)
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
    
    model = build_model(architecture, model_config)
    model_kwargs = map_config_strings(model_config)
    
    # Same preset as training, so the sampling convention always matches
    noise_schedule_type = conf.get('noise_schedule', conf.get('arguments', {}).get('noise_schedule', 'edm'))
    _, noise_schedule, prediction_transform = get_diffusion_preset(noise_schedule_type)
    
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
    """Restore (state, best_state) from a single orbax checkpoint directory.

    Raises if the checkpoint cannot be read: the callers below decide what a
    failed load means, and reporting it as an empty pair made an unusable
    pipeline look like a successful one.
    """
    checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(create=False)
    # Convert checkpoint_dir to absolute path
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    manager = CheckpointManager(checkpoint_dir, checkpointer, options)
    ckpt = manager.restore(checkpoint_dir)
    print(f"Loaded checkpoint from local dir {checkpoint_dir}")
    return ckpt.get('state'), ckpt.get('best_state')

def load_from_wandb_run(
    run,
    project: str,
    entity: str = None,
):
    """
    Loads model from a wandb run's latest model artifact.

    Returns (states, config, run, artifact); every element is None if the
    lookup failed, so a caller can tell a miss from a load.
    """
    states = None
    config = None
    artifact = None
    try:
        if isinstance(run, str):
            run = get_wandb_run(run, project, entity)
        if run is None:
            raise ValueError("No wandb run found")
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

    Returns (states, config, run, artifact); every element is None if the
    lookup failed, so a caller can tell a miss from a load.
    """
    states = None
    config = None
    run = None
    artifact = None
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