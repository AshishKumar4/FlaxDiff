import jax
import jax.numpy as jnp
import flax.struct as struct
import flax.linen as nn
from typing import Iterator, Optional
import numpy as np
import os
import queue
import threading
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
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

# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

DATA_AXIS = 'data'
FSDP_AXIS = 'fsdp'

# Batches are split across every device, whichever axis it sits on; only
# parameters distinguish the two axes.
BATCH_SPEC = P((DATA_AXIS, FSDP_AXIS))

# Below this many elements a parameter costs more in collectives than it saves
# in memory, so it stays replicated.
DEFAULT_MIN_SHARD_SIZE = 2 ** 16


def build_mesh(fsdp_size: int = 1, devices: Optional[list] = None) -> Mesh:
    """Two-axis device mesh: parameters shard over 'fsdp', replicate over 'data'.

    fsdp_size=1 degenerates to plain data parallelism, so the same code path
    serves both without a flag. Axes are Auto so GSPMD infers the collectives
    rather than us writing them by hand.
    """
    devices = list(devices) if devices is not None else jax.devices()
    if fsdp_size < 1 or len(devices) % fsdp_size:
        raise ValueError(
            f"fsdp_size {fsdp_size} must be a positive divisor of device count {len(devices)}")
    return jax.make_mesh(
        (len(devices) // fsdp_size, fsdp_size),
        (DATA_AXIS, FSDP_AXIS),
        devices=devices,
        axis_types=(AxisType.Auto, AxisType.Auto),
    )


def parameter_spec(shape: tuple, fsdp_size: int, min_shard_size: int) -> P:
    """Shard the largest evenly-divisible axis over 'fsdp', else replicate.

    Applied to every leaf of the train state, not just params: optimizer moments
    and EMA copies have the same shapes as the params they track, so they pick
    up the same spec without anyone having to describe the optimizer's layout.
    """
    if fsdp_size == 1 or int(np.prod(shape, dtype=np.int64)) < min_shard_size:
        return P()
    for axis in sorted(range(len(shape)), key=lambda i: -shape[i]):
        if shape[axis] % fsdp_size == 0:
            return P(*([None] * axis), FSDP_AXIS)
    return P()


def state_sharding_tree(
    mesh: Mesh, abstract_state, min_shard_size: int = DEFAULT_MIN_SHARD_SIZE
):
    """Map a train state of ShapeDtypeStructs to its NamedSharding tree."""
    fsdp_size = mesh.shape[FSDP_AXIS]
    return jax.tree.map(
        lambda x: NamedSharding(mesh, parameter_spec(x.shape, fsdp_size, min_shard_size)),
        abstract_state,
    )


def batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, BATCH_SPEC)


def shard_batch(sharding: NamedSharding, batch):
    """Assemble this process's slice of each array into a globally sharded one."""
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(sharding, np.asarray(x)), batch)


class DevicePrefetchIterator:
    """Runs the host-to-device batch transfer a few batches ahead of the loop.

    Without this the transfer sits on the critical path between steps, because
    the loop only starts moving batch N+1 after step N has been dispatched.
    """

    def __init__(self, iterator: Iterator, sharding: NamedSharding, depth: int = 2,
                 source_state=None):
        self._iterator = iter(iterator)
        self._sharding = sharding
        self._queue = queue.Queue(maxsize=depth)
        self._terminal: Optional[BaseException] = None
        self._checkpointable = hasattr(self._iterator, 'get_state')
        if source_state is not None:
            if not self._checkpointable:
                raise TypeError(
                    f"{type(self._iterator).__name__} cannot resume from a saved position")
            self._iterator.set_state(source_state)
        # Position of the source iterator as of the batch most recently handed
        # out, so a checkpoint resumes at the next unseen batch rather than at
        # whatever the prefetch thread has already raced ahead to.
        self.source_state = source_state
        self._thread = threading.Thread(target=self._prefetch, daemon=True)
        self._thread.start()

    def _prefetch(self):
        try:
            while True:
                batch = next(self._iterator)
                state = self._iterator.get_state() if self._checkpointable else None
                self._queue.put((shard_batch(self._sharding, batch), state))
        except StopIteration:
            self._queue.put(StopIteration())
        except BaseException as error:  # surfaced on the consumer's thread
            self._queue.put(error)

    def __iter__(self):
        return self

    def __next__(self):
        if self._terminal is not None:
            raise self._terminal
        item = self._queue.get()
        if isinstance(item, BaseException):
            self._terminal = item
            raise item
        batch, self.source_state = item
        return batch

# ---------------------------------------------------------------------------
# Throughput accounting
# ---------------------------------------------------------------------------

# Dense bf16 peak per chip, from the vendors' own spec sheets. Only used to turn
# measured FLOPs into a utilisation percentage; unknown hardware just skips MFU.
PEAK_FLOPS_PER_DEVICE = {
    'TPU v2': 45e12,
    'TPU v3': 123e12,
    'TPU v4': 275e12,
    'TPU v5 lite': 197e12,
    'TPU v5e': 197e12,
    'TPU v5': 459e12,
    'TPU v5p': 459e12,
    'TPU v6 lite': 918e12,
    'TPU v6e': 918e12,
    'NVIDIA A100': 312e12,
    'NVIDIA H100': 989e12,
    'NVIDIA H200': 989e12,
}


def step_flops(jitted, *args, **kwargs) -> Optional[float]:
    """FLOPs for one call of a jitted function, straight from the compiler.

    Measured rather than derived from a hand-written parameter-count formula, so
    it stays honest across architectures, remat and gradient accumulation.
    """
    analysis = jitted.lower(*args, **kwargs).compile().cost_analysis()
    if isinstance(analysis, (list, tuple)):
        analysis = analysis[0] if analysis else None
    if not analysis or 'flops' not in analysis:
        return None
    return float(analysis['flops'])


def model_flops_utilization(
    flops_per_step: Optional[float], step_time: float, device_count: int
) -> Optional[float]:
    """Fraction of the cluster's peak FLOPs the training step actually achieved."""
    if not flops_per_step or step_time <= 0:
        return None
    peak = PEAK_FLOPS_PER_DEVICE.get(jax.devices()[0].device_kind)
    if peak is None:
        return None
    return flops_per_step / step_time / (peak * device_count)


def enable_compilation_cache(path: str):
    """Persist compiled executables so restarts skip XLA compilation.

    The dominant cost of a restart-heavy TPU workflow, where every run otherwise
    recompiles the same step function from scratch.
    """
    os.makedirs(path, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', path)
    # Defaults skip small/fast compilations; a training step is neither, and
    # caching everything keeps startup predictable.
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)


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