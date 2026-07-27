import jax
import jax.numpy as jnp
import flax.struct as struct
import flax.linen as nn
from typing import Any
from functools import partial
import numpy as np
import os
from jax.sharding import Mesh, PartitionSpec as P
from flaxdiff.inputs import TextEncoder, CLIPTextEncoder

# Setup mappings for dtype, precision, and activation
def serialize_model(model: nn.Module):
    """
    Serializes the model to a dictionary format.
    """
    model_dict = model.__dict__
    model_dict = {k: v for k, v in model_dict.items() if not k.startswith('_')}
    # Convert all callable attributes to their string representation
    def map(model_dict):
        for k, v in model_dict.items():
            if isinstance(v, dict):
                # Recursively serialize nested dictionaries
                model_dict[k] = map(v)
            elif isinstance(v, list):
                # Recursively serialize lists
                [map(item) if isinstance(item, dict) else item for item in v]
            elif callable(v):
                # If the attribute has __name__, use that as the key
                if hasattr(v, '__name__'):
                    model_dict[k] = v.__name__
                else:
                    model_dict[k] = str(v).split('.')[-1]
    map(model_dict)
    return model_dict

def get_latest_checkpoint(checkpoint_path):
    checkpoint_files = os.listdir(checkpoint_path)
    # Sort files by step number
    checkpoint_files = sorted([int(i) for i in checkpoint_files])
    latest_step = checkpoint_files[-1]
    latest_checkpoint = os.path.join(checkpoint_path, str(latest_step))
    return latest_checkpoint

class MarkovState(struct.PyTreeNode):
    pass

class RandomMarkovState(MarkovState):
    rng: jax.random.PRNGKey
    def get_random_key(self):
        rng, subkey = jax.random.split(self.rng)
        return RandomMarkovState(rng), subkey

def clip_images(images, clip_min=-1, clip_max=1):
    """Clip image values to a specified range.
    
    Args:
        images: Images to clip
        clip_min: Minimum value
        clip_max: Maximum value
    
    Returns:
        Clipped images
    """
    return jnp.clip(images, clip_min, clip_max)

def denormalize_images(images, target_type=jnp.uint8, source_range=(-1, 1), target_range=(0, 255)):
    """Convert images from normalized range (e.g. [-1, 1]) to target range (e.g. [0, 255]).
    
    Args:
        images: Normalized images
        target_type: Target dtype (e.g. jnp.uint8 for standard images)
        source_range: Tuple of (min, max) for the source normalization range
        target_range: Tuple of (min, max) for the target range
        
    Returns:
        Denormalized images in the target dtype
    """
    src_min, src_max = source_range
    tgt_min, tgt_max = target_range
    
    # First clip to ensure we're in the expected source range
    images = clip_images(images, src_min, src_max)
    
    # Scale to [0, 1]
    images = (images - src_min) / (src_max - src_min)
    
    # Scale to target range
    images = images * (tgt_max - tgt_min) + tgt_min
    
    # Convert to target dtype if needed
    if target_type is not None:
        images = images.astype(target_type)
    
    return images

def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh
) -> tuple[tuple[int, ...], jax.sharding.NamedSharding]:
  sharding = jax.sharding.NamedSharding(global_mesh, P(global_mesh.axis_names))
  global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]
  return global_shape, sharding


def form_global_array(path, array: np.ndarray, global_mesh: Mesh) -> jax.Array:
  """Put local sharded array into local devices"""
  global_shape, sharding = _build_global_shape_and_sharding(np.shape(array), global_mesh)
  try:
    local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
  except ValueError as array_split_error:
    raise ValueError(
        f"Unable to put to devices shape {array.shape} with "
        f"local device count {len(global_mesh.local_devices)} "
    ) from array_split_error
  local_device_buffers = jax.device_put(local_device_arrays, global_mesh.local_devices)
  return jax.make_array_from_single_device_arrays(global_shape, sharding, local_device_buffers)

def convert_to_global_tree(global_mesh, pytree):
    return jax.tree_util.tree_map_with_path(partial(form_global_array, global_mesh=global_mesh), pytree)

class AutoTextTokenizer:
    def __init__(self, tensor_type="pt", modelname="openai/clip-vit-large-patch14"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.tensor_type = tensor_type

    def __call__(self, inputs):
        # print(caption)
        tokens = self.tokenizer(inputs, padding="max_length", max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors=self.tensor_type)
        # print(tokens.keys())
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "caption": inputs,
        }

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AutoAudioProcessor:
    """Turn raw audio waveforms into model inputs, for any HF audio model.

    The audio counterpart of AutoTextTokenizer: it runs on CPU in the grain
    workers so the device only sees ready tensors. Whatever keys the model's
    feature extractor emits (`input_values` for wav2vec2/HuBERT,
    `input_features` for Whisper/AST, ...) are passed through unchanged, so
    switching audio models needs no change here.
    """
    def __init__(self, tensor_type="np", modelname="facebook/wav2vec2-base-960h",
                 sampling_rate=None):
        from transformers import AutoFeatureExtractor
        self.processor = AutoFeatureExtractor.from_pretrained(modelname)
        self.tensor_type = tensor_type
        self.modelname = modelname
        # The processor knows the rate its model was trained at
        self.sampling_rate = sampling_rate or getattr(self.processor, "sampling_rate", 16000)

    def __call__(self, audio):
        features = self.processor(audio, sampling_rate=self.sampling_rate,
                                  padding=True, return_tensors=self.tensor_type)
        return dict(features)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def defaultTextEncodeModel(modelname = "openai/clip-vit-large-patch14", backend="jax"):
    """Default text encoder model."""
    return CLIPTextEncoder.from_modelname(modelname=modelname, backend=backend)