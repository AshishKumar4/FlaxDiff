"""Single source of truth for architecture names and model construction.

Both training.py and the inference pipeline build models through here, so a
config logged at training time always reconstructs the same model at
inference time. Architecture names may carry +2d/+hilbert/+zigzag suffixes;
they canonicalize to the base architecture plus the matching config flags.
"""

import dataclasses

import jax
import jax.numpy as jnp

from .simple_unet import Unet
from .simple_vit import UViT, SimpleUDiT
from .simple_dit import SimpleDiT
from .simple_mmdit import SimpleMMDiT, HierarchicalMMDiT
from .ssm_dit import HybridSSMAttentionDiT
from .video_dit import VideoDiT
from .unet_3d import UNet3D

MODEL_REGISTRY = {
    'unet': Unet,
    'uvit': UViT,
    'simple_udit': SimpleUDiT,
    'simple_dit': SimpleDiT,
    'simple_mmdit': SimpleMMDiT,
    'hierarchical_mmdit': HierarchicalMMDiT,
    'hybrid_dit': HybridSSMAttentionDiT,
    'video_dit': VideoDiT,
    'unet_3d': UNet3D,
}

ARCHITECTURE_SUFFIX_FLAGS = {
    '+2d': 'use_2d_fusion',
    '+hilbert': 'use_hilbert',
    '+zigzag': 'use_zigzag',
}

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
    'mish': jax.nn.mish,
    'gelu': jax.nn.gelu,
    'relu': jax.nn.relu,
}


def canonicalize_architecture(architecture: str) -> tuple[str, dict]:
    """Strip +2d/+hilbert/+zigzag suffixes into their matching config flags."""
    flags = {}
    canonical = architecture
    for suffix, flag in ARCHITECTURE_SUFFIX_FLAGS.items():
        if suffix in canonical:
            canonical = canonical.replace(suffix, '')
            flags[flag] = True
    return canonical, flags


def map_config_strings(config: dict):
    """Convert the string leaves of a logged config back to objects.

    dtypes, precisions and activations are stored as strings in the wandb
    config; function paths like 'jax.nn.mish' (or the 'jax._src.nn.functions.*'
    that older configs contain) resolve by attribute walk.
    """
    if isinstance(config, dict):
        return {k: map_config_strings(v) for k, v in config.items()}
    if isinstance(config, list):
        return [map_config_strings(v) for v in config]
    if isinstance(config, str):
        if config in DTYPE_MAP:
            return DTYPE_MAP[config]
        if config in PRECISION_MAP:
            return PRECISION_MAP[config]
        if config in ACTIVATION_MAP:
            return ACTIVATION_MAP[config]
        if config == 'None':
            return None
        if config.startswith('jax.') or config.startswith('jax._src.'):
            attr_path = config.replace('jax._src.nn.functions', 'jax.nn')
            obj = jax
            for part in attr_path.split('.')[1:]:
                obj = getattr(obj, part)
            return obj
    return config


def build_model(architecture: str, config: dict):
    """Construct a model from an architecture name and a plain config dict.

    Config keys the model has no field for are dropped with a notice (older
    runs logged since-removed flags), so old configs keep reconstructing.
    """
    canonical, flags = canonicalize_architecture(architecture)

    if canonical == 'diffusers_unet_simple':
        # Third-party linen model, kept behind a lazy import and the BCHW seam
        from diffusers import FlaxUNet2DConditionModel
        from .general import BCHWModelWrapper
        return BCHWModelWrapper(FlaxUNet2DConditionModel(**map_config_strings(config)))

    model_class = MODEL_REGISTRY.get(canonical)
    if model_class is None:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported: {', '.join(MODEL_REGISTRY.keys())}")

    kwargs = {**map_config_strings(config), **flags}
    valid_fields = {f.name for f in dataclasses.fields(model_class)}
    dropped = sorted(set(kwargs) - valid_fields)
    if dropped:
        print(f"Dropping config keys not accepted by {model_class.__name__}: {dropped}")
    kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

    return model_class(**kwargs)
