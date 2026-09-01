"""Model backbones.

`registry.build_model` is the primary entry point: training and inference both
construct models from a logged architecture name plus config, so a checkpoint
always reconstructs the model it was trained with. The classes are exported for
direct use in notebooks and tests.
"""

from .simple_unet import Unet
from .simple_vit import UViT, SimpleUDiT
from .simple_dit import SimpleDiT
from .simple_mmdit import SimpleMMDiT, HierarchicalMMDiT
from .ssm_dit import HybridSSMAttentionDiT
from .video_dit import VideoDiT
from .unet_3d import UNet3D
from .registry import build_model, canonicalize_architecture

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
