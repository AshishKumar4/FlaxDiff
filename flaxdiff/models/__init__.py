"""Model backbones.

`registry.build_model` is the primary entry point: training and inference both
construct models from a logged architecture name plus config, so a checkpoint
always reconstructs the model it was trained with. The classes are exported for
direct use in notebooks and tests.
"""

from typing import Any

from .simple_unet import Unet
from .simple_vit import UViT, SimpleUDiT
from .simple_dit import SimpleDiT
from .simple_mmdit import SimpleMMDiT, HierarchicalMMDiT
from .ssm_dit import HybridSSMAttentionDiT
from .video_dit import VideoDiT
from .unet_3d import UNet3D

__all__ = [
    # Primary construction API
    "build_model",
    "canonicalize_architecture",
    # Image backbones
    "Unet",
    "UViT",
    "SimpleUDiT",
    "SimpleDiT",
    "SimpleMMDiT",
    "HierarchicalMMDiT",
    "HybridSSMAttentionDiT",
    # Video backbones
    "VideoDiT",
    "UNet3D",
]

_REGISTRY_EXPORTS = ("build_model", "canonicalize_architecture", "MODEL_REGISTRY")


def __getattr__(name: str) -> Any:
    """Resolve the registry exports on first use.

    registry.py reaches the JEPA models, which pull in the trainer, which pulls
    in flaxdiff.utils, which imports flaxdiff.inputs, which imports
    models.autoencoder — importing the registry eagerly here would therefore
    close that loop and break `import flaxdiff.schedulers` on a cold
    interpreter. Deferring it keeps `from flaxdiff.models import build_model`
    working without the cycle.
    """
    if name in _REGISTRY_EXPORTS:
        from . import registry
        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
