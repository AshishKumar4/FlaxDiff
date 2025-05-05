import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Optional
import einops
from flax.typing import Dtype, PrecisionLike

from .attention import NormalAttention

def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * \
        channels == x.shape[2], f"Invalid shape: {x.shape}, should be {h*w}, {patch_size**2*channels}"
    x = einops.rearrange(
        x, 'B (h w) (p1 p2 C) -> B (h p1) (w p2) C', h=h, p1=patch_size, p2=patch_size)
    return x


class PatchEmbedding(nn.Module):
    patch_size: int
    embedding_dim: int
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGH

    @nn.compact
    def __call__(self, x):
        batch, height, width, channels = x.shape
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        x = nn.Conv(features=self.embedding_dim,
                    kernel_size=(self.patch_size, self.patch_size),
                    strides=(self.patch_size, self.patch_size),
                    dtype=self.dtype,
                    precision=self.precision)(x)
        x = jnp.reshape(x, (batch, -1, self.embedding_dim))
        return x


class PositionalEncoding(nn.Module):
    max_len: int
    embedding_dim: int

    @nn.compact
    def __call__(self, x):
        pe = self.param('pos_encoding',
                        jax.nn.initializers.zeros,
                        (1, self.max_len, self.embedding_dim))
        return x + pe[:, :x.shape[1], :]


# --- Rotary Positional Embedding (RoPE) ---
# Adapted from https://github.com/google-deepmind/ring_attention/blob/main/ring_attention/layers/rotary.py


def _rotate_half(x: jax.Array) -> jax.Array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rotary_embedding(
    x: jax.Array, freqs_cos: jax.Array, freqs_sin: jax.Array
) -> jax.Array:
    """Applies rotary embedding to the input tensor using rotate_half method."""
    # x shape: [..., Sequence, Dimension] e.g. [B, H, S, D] or [B, S, D]
    # freqs_cos/sin shape: [Sequence, Dimension / 2]

    # Expand dims for broadcasting: [1, 1, S, D/2] or [1, S, D/2]
    if x.ndim == 4:  # [B, H, S, D]
        cos_freqs = jnp.expand_dims(freqs_cos, axis=(0, 1))
        sin_freqs = jnp.expand_dims(freqs_sin, axis=(0, 1))
    elif x.ndim == 3:  # [B, S, D]
        cos_freqs = jnp.expand_dims(freqs_cos, axis=0)
        sin_freqs = jnp.expand_dims(freqs_sin, axis=0)

    # Duplicate cos and sin for the full dimension D
    # Shape becomes [..., S, D]
    cos_freqs = jnp.concatenate([cos_freqs, cos_freqs], axis=-1)
    sin_freqs = jnp.concatenate([sin_freqs, sin_freqs], axis=-1)

    # Apply rotation: x * cos + rotate_half(x) * sin
    x_rotated = x * cos_freqs + _rotate_half(x) * sin_freqs
    return x_rotated.astype(x.dtype)

class RotaryEmbedding(nn.Module):
    dim: int
    max_seq_len: int = 4096  # Increased default based on SimpleDiT
    base: int = 10000
    dtype: Dtype = jnp.float32

    def setup(self):
        inv_freq = 1.0 / (
            self.base ** (jnp.arange(0, self.dim, 2,
                          dtype=jnp.float32) / self.dim)
        )
        t = jnp.arange(self.max_seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        self.freqs_cos = jnp.cos(freqs)
        self.freqs_sin = jnp.sin(freqs)

    def __call__(self, seq_len: int):
        if seq_len > self.max_seq_len:
            # Dynamically extend frequencies if needed (more robust)
            t = jnp.arange(seq_len, dtype=jnp.float32)
            inv_freq = 1.0 / (
                self.base ** (jnp.arange(0, self.dim, 2,
                              dtype=jnp.float32) / self.dim)
            )
            freqs = jnp.outer(t, inv_freq)
            freqs_cos = jnp.cos(freqs)
            freqs_sin = jnp.sin(freqs)
            # Consider caching extended freqs if this happens often
            return freqs_cos, freqs_sin
            # Or raise error like before:
            # raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        return self.freqs_cos[:seq_len, :], self.freqs_sin[:seq_len, :]

    
# --- Attention with RoPE ---


class RoPEAttention(NormalAttention):
    rope_emb: RotaryEmbedding = None

    @nn.compact
    def __call__(self, x, context=None, freqs_cis=None):
        orig_x_shape = x.shape
        is_4d = len(orig_x_shape) == 4
        if is_4d:
            B, H, W, C = x.shape
            seq_len = H * W
            x = x.reshape((B, seq_len, C))
        else:
            B, seq_len, C = x.shape

        context = x if context is None else context
        if len(context.shape) == 4:
            _B, _H, _W, _C = context.shape
            context_seq_len = _H * _W
            context = context.reshape((B, context_seq_len, _C))
        # else: # context is already [B, S_ctx, C]

        query = self.query(x)      # [B, S, H, D]
        key = self.key(context)    # [B, S_ctx, H, D]
        value = self.value(context)  # [B, S_ctx, H, D]

        if freqs_cis is None and self.rope_emb is not None:
            seq_len_q = query.shape[1]  # Use query's sequence length
            freqs_cos, freqs_sin = self.rope_emb(seq_len_q)
        elif freqs_cis is not None:
            freqs_cos, freqs_sin = freqs_cis
        else:
            # Should not happen if rope_emb is provided or freqs_cis are passed
            raise ValueError("RoPE frequencies not provided.")

        # Apply RoPE to query and key
        # Permute to [B, H, S, D] for RoPE application
        query = einops.rearrange(query, 'b s h d -> b h s d')
        key = einops.rearrange(key, 'b s h d -> b h s d')

        # Apply RoPE only up to the context sequence length for keys if different
        # Assuming self-attention or context has same seq len for simplicity here
        query = apply_rotary_embedding(query, freqs_cos, freqs_sin)
        key = apply_rotary_embedding(
            key, freqs_cos, freqs_sin)  # Apply same freqs to key

        # Permute back to [B, S, H, D] for dot_product_attention
        query = einops.rearrange(query, 'b h s d -> b s h d')
        key = einops.rearrange(key, 'b h s d -> b s h d')

        hidden_states = nn.dot_product_attention(
            query, key, value, dtype=self.dtype, broadcast_dropout=False,
            dropout_rng=None, precision=self.precision, force_fp32_for_softmax=self.force_fp32_for_softmax,
            deterministic=True
        )

        proj = self.proj_attn(hidden_states)

        if is_4d:
            proj = proj.reshape(orig_x_shape)

        return proj


# --- adaLN-Zero ---


class AdaLNZero(nn.Module):
    features: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-5  # Standard LayerNorm epsilon

    @nn.compact
    def __call__(self, x, conditioning):
        # Project conditioning signal to get scale and shift parameters
        # Conditioning shape: [B, D_cond] -> [B, 1, ..., 1, 6 * features] for broadcasting
        # Or [B, 1, 6*features] if x is [B, S, F]

        # Ensure conditioning has seq dim if x does
        # x=[B,S,F], cond=[B,D_cond]
        if x.ndim == 3 and conditioning.ndim == 2:
            conditioning = jnp.expand_dims(
                conditioning, axis=1)  # cond=[B,1,D_cond]

        # Project conditioning to get 6 params per feature (scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn)
        # Using nn.DenseGeneral for flexibility if needed, but nn.Dense is fine if cond is [B, D_cond] or [B, 1, D_cond]
        ada_params = nn.Dense(
            features=6 * self.features,
            dtype=self.dtype,
            precision=self.precision,
            # Initialize projection to zero (Zero init)
            kernel_init=nn.initializers.zeros,
            name="ada_proj"
        )(conditioning)

        # Split into scale, shift, gate for MLP and Attention
        scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn = jnp.split(
            ada_params, 6, axis=-1)

        scale_mlp = jnp.clip(scale_mlp, -10.0, 10.0)
        shift_mlp = jnp.clip(shift_mlp, -10.0, 10.0)
        # Apply Layer Normalization
        norm = nn.LayerNorm(epsilon=self.norm_epsilon,
                            use_scale=False, use_bias=False, dtype=self.dtype)
        # norm = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype) # Alternative: RMSNorm

        norm_x = norm(x)

        # Modulate for Attention path
        x_attn = norm_x * (1 + scale_attn) + shift_attn

        # Modulate for MLP path
        x_mlp = norm_x * (1 + scale_mlp) + shift_mlp

        # Return modulated outputs and gates
        return x_attn, gate_attn, x_mlp, gate_mlp

class AdaLNParams(nn.Module): # Renamed for clarity
    features: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, conditioning):
        # Ensure conditioning is broadcastable if needed (e.g., [B, 1, D_cond])
        if conditioning.ndim == 2:
             conditioning = jnp.expand_dims(conditioning, axis=1)

        # Project conditioning to get 6 params per feature
        ada_params = nn.Dense(
            features=6 * self.features,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,
            name="ada_proj"
        )(conditioning)
        # Return all params (or split if preferred, but maybe return tuple/dict)
        # Shape: [B, 1, 6*F]
        return ada_params # Or split and return tuple: jnp.split(ada_params, 6, axis=-1)
    