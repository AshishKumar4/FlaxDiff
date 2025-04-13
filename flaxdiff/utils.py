import jax
import jax.numpy as jnp
import flax.struct as struct
import flax.linen as nn
from typing import Any
from functools import partial
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from flaxdiff.inputs import TextEncoder

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

class RMSNorm(nn.Module):
    """
    From "Root Mean Square Layer Normalization" by https://arxiv.org/abs/1910.07467

    Adapted from flax.linen.LayerNorm
    """

    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    use_scale: bool = True
    scale_init: Any = jax.nn.initializers.ones

    @nn.compact
    def __call__(self, x):
        reduction_axes = (-1,)
        feature_axes = (-1,)

        rms_sq = self._compute_rms_sq(x, reduction_axes)

        return self._normalize(
            self,
            x,
            rms_sq,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_scale,
            self.scale_init,
        )

    def _compute_rms_sq(self, x, axes):
        x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
        rms_sq = jnp.mean(jax.lax.square(x), axes)
        return rms_sq

    def _normalize(
        self,
        mdl,
        x,
        rms_sq,
        reduction_axes,
        feature_axes,
        dtype,
        param_dtype,
        epsilon,
        use_scale,
        scale_init,
    ):
        reduction_axes = nn.normalization._canonicalize_axes(x.ndim, reduction_axes)
        feature_axes = nn.normalization._canonicalize_axes(x.ndim, feature_axes)
        stats_shape = list(x.shape)
        for axis in reduction_axes:
            stats_shape[axis] = 1
        rms_sq = rms_sq.reshape(stats_shape)
        feature_shape = [1] * x.ndim
        reduced_feature_shape = []
        for ax in feature_axes:
            feature_shape[ax] = x.shape[ax]
            reduced_feature_shape.append(x.shape[ax])
        mul = jax.lax.rsqrt(rms_sq + epsilon)
        if use_scale:
            scale = mdl.param(
                "scale", scale_init, reduced_feature_shape, param_dtype
            ).reshape(feature_shape)
            mul *= scale
        y = mul * x
        return jnp.asarray(y, dtype)

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

def defaultTextEncodeModel(backend="jax"):
    from transformers import (
        CLIPTextModel,
        FlaxCLIPTextModel,
        AutoTokenizer,
    )
    modelname = "openai/clip-vit-large-patch14"
    if backend == "jax":
        model = FlaxCLIPTextModel.from_pretrained(
            modelname, dtype=jnp.bfloat16)
    else:
        model = CLIPTextModel.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname, dtype=jnp.float16)
    return TextEncoder(model, tokenizer)