# flaxdiff/models/better_uvit.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Any, Optional, Tuple, Sequence, Union
import einops
from functools import partial

# Re-use existing components if they are suitable
from .common import kernel_init, FourierEmbedding, TimeProjection, hilbert_indices, inverse_permutation
from .attention import NormalAttention # Using NormalAttention for RoPE integration
from flax.typing import Dtype, PrecisionLike

# --- Rotary Positional Embedding (RoPE) ---
# Adapted from https://github.com/google-deepmind/ring_attention/blob/main/ring_attention/layers/rotary.py

def _rotate_half(x: jax.Array) -> jax.Array:
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return jnp.concatenate((-x2, x1), axis=-1)

def apply_rotary_embedding(
    x: jax.Array, freqs_cis: jax.Array
) -> jax.Array:
  """Applies rotary embedding to the input tensor using rotate_half method."""
  # x shape: [..., Sequence, Dimension] e.g. [B, H, S, D] or [B, S, D]
  # freqs_cis shape: complex [Sequence, Dimension / 2]

  # Extract cos and sin from the complex freqs_cis
  cos_freqs = jnp.real(freqs_cis) # Shape [S, D/2]
  sin_freqs = jnp.imag(freqs_cis) # Shape [S, D/2]

  # Expand dims for broadcasting: [1, 1, S, D/2] or [1, S, D/2]
  if x.ndim == 4: # [B, H, S, D]
      cos_freqs = jnp.expand_dims(cos_freqs, axis=(0, 1))
      sin_freqs = jnp.expand_dims(sin_freqs, axis=(0, 1))
  elif x.ndim == 3: # [B, S, D]
      cos_freqs = jnp.expand_dims(cos_freqs, axis=0)
      sin_freqs = jnp.expand_dims(sin_freqs, axis=0)
      
  # Duplicate cos and sin for the full dimension D
  # Shape becomes [..., S, D]
  cos_freqs = jnp.concatenate([cos_freqs, cos_freqs], axis=-1)
  sin_freqs = jnp.concatenate([sin_freqs, sin_freqs], axis=-1)

  # Apply rotation: x * cos + rotate_half(x) * sin
  x_rotated = x * cos_freqs + _rotate_half(x) * sin_freqs
  return x_rotated.astype(x.dtype)


class RotaryEmbedding(nn.Module):
  dim: int # Dimension of the head
  max_seq_len: int = 2048
  base: int = 10000
  dtype: Dtype = jnp.float32

  def setup(self):
    inv_freq = 1.0 / (
        self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
    )
    t = jnp.arange(self.max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq) # Shape: [max_seq_len, dim / 2]
    
    # Precompute the complex form: cos(theta) + i * sin(theta)
    self.freqs_cis_complex = jnp.cos(freqs) + 1j * jnp.sin(freqs)
    # Shape: [max_seq_len, dim / 2]

  def __call__(self, seq_len: int):
    if seq_len > self.max_seq_len:
        raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
    # Return complex shape [seq_len, dim / 2]
    return self.freqs_cis_complex[:seq_len, :]

# --- Attention with RoPE ---

class RoPEAttention(NormalAttention):
    rope_emb: RotaryEmbedding

    @nn.compact
    def __call__(self, x, context=None, freqs_cis=None):
        # x has shape [B, H, W, C] or [B, S, C]
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
        else:
            _B, context_seq_len, _C = context.shape

        query = self.query(x) # [B, S, H, D]
        key = self.key(context) # [B, S_ctx, H, D]
        value = self.value(context) # [B, S_ctx, H, D]

        # Apply RoPE to query and key
        if freqs_cis is not None:
            # Permute to [B, H, S, D] for RoPE application if needed by apply_rotary_embedding
            query = einops.rearrange(query, 'b s h d -> b h s d')
            key = einops.rearrange(key, 'b s h d -> b h s d')
            
            query = apply_rotary_embedding(query, freqs_cis)
            key = apply_rotary_embedding(key, freqs_cis) # Apply to key as well

            # Permute back to [B, S, H, D] for dot_product_attention
            query = einops.rearrange(query, 'b h s d -> b s h d')
            key = einops.rearrange(key, 'b h s d -> b s h d')

        hidden_states = nn.dot_product_attention(
            query, key, value, dtype=self.dtype, broadcast_dropout=False,
            dropout_rng=None, precision=self.precision, force_fp32_for_softmax=self.force_fp32_for_softmax,
            deterministic=True
        ) # Output shape [B, S, H, D]

        proj = self.proj_attn(hidden_states) # Output shape [B, S, C]
        
        if is_4d:
            proj = proj.reshape(orig_x_shape) # Reshape back if input was 4D
            
        return proj

# --- adaLN-Zero ---

class AdaLNZero(nn.Module):
    features: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-5 # Standard LayerNorm epsilon

    @nn.compact
    def __call__(self, x, conditioning):
        # Project conditioning signal to get scale and shift parameters
        # Conditioning shape: [B, D_cond] -> [B, 1, ..., 1, 6 * features] for broadcasting
        # Or [B, 1, 6*features] if x is [B, S, F]
        
        # Ensure conditioning has seq dim if x does
        if x.ndim == 3 and conditioning.ndim == 2: # x=[B,S,F], cond=[B,D_cond]
            conditioning = jnp.expand_dims(conditioning, axis=1) # cond=[B,1,D_cond]
            
        # Project conditioning to get 6 params per feature (scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn)
        # Using nn.DenseGeneral for flexibility if needed, but nn.Dense is fine if cond is [B, D_cond] or [B, 1, D_cond]
        ada_params = nn.Dense(
            features=6 * self.features, 
            dtype=self.dtype, 
            precision=self.precision,
            kernel_init=nn.initializers.zeros, # Initialize projection to zero (Zero init)
            name="ada_proj"
        )(conditioning)

        # Split into scale, shift, gate for MLP and Attention
        scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn = jnp.split(ada_params, 6, axis=-1)

        # Apply Layer Normalization
        norm = nn.LayerNorm(epsilon=self.norm_epsilon, use_scale=False, use_bias=False, dtype=self.dtype)
        # norm = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype) # Alternative: RMSNorm
        
        norm_x = norm(x)

        # Modulate for Attention path
        x_attn = norm_x * (1 + scale_attn) + shift_attn

        # Modulate for MLP path
        x_mlp = norm_x * (1 + scale_mlp) + shift_mlp

        # Return modulated outputs and gates
        return x_attn, gate_attn, x_mlp, gate_mlp


# --- DiT Block ---

class DiTBlock(nn.Module):
    features: int
    num_heads: int
    mlp_ratio: int = 4
    dropout_rate: float = 0.0 # Typically dropout is not used in diffusion models
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_flash_attention: bool = False # Keep option, but RoPEAttention uses NormalAttention base
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    rope_emb: RotaryEmbedding # Pass RoPE module

    def setup(self):
        hidden_features = int(self.features * self.mlp_ratio)
        self.ada_ln_zero = AdaLNZero(self.features, dtype=self.dtype, precision=self.precision, norm_epsilon=self.norm_epsilon)
        
        # Use RoPEAttention
        self.attention = RoPEAttention(
            query_dim=self.features,
            heads=self.num_heads,
            dim_head=self.features // self.num_heads,
            dtype=self.dtype,
            precision=self.precision,
            use_bias=True, # Bias is common in DiT attention proj
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            rope_emb=self.rope_emb # Pass RoPE module instance
        )
        
        # Standard MLP block
        self.mlp = nn.Sequential([
            nn.Dense(features=hidden_features, dtype=self.dtype, precision=self.precision),
            nn.gelu,
            nn.Dense(features=self.features, dtype=self.dtype, precision=self.precision)
        ])

    @nn.compact
    def __call__(self, x, conditioning, freqs_cis):
        # x shape: [B, S, F]
        # conditioning shape: [B, D_cond]
        
        residual = x
        
        # Apply adaLN-Zero to get modulated inputs and gates
        x_attn, gate_attn, x_mlp, gate_mlp = self.ada_ln_zero(x, conditioning)
        
        # Attention block
        attn_output = self.attention(x_attn, context=None, freqs_cis=freqs_cis) # Self-attention only
        x = residual + gate_attn * attn_output
        
        # MLP block
        mlp_output = self.mlp(x_mlp)
        x = x + gate_mlp * mlp_output
        
        return x

# --- Patch Embedding (reuse or define if needed) ---
# Assuming PatchEmbedding exists in simple_vit.py and is suitable
from .simple_vit import PatchEmbedding, unpatchify 

# --- Better UViT (DiT Style) ---

class BetterUViT(nn.Module):
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout_rate: float = 0.0 # Typically 0 for diffusion
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_flash_attention: bool = False # Passed down, but RoPEAttention uses NormalAttention
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False # Option to predict sigma like in DiT paper
    use_hilbert: bool = False  # Toggle Hilbert patch reorder

    def setup(self):
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size, 
            embedding_dim=self.emb_features,
            dtype=self.dtype,
            precision=self.precision
        )
        
        # Time embedding projection
        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features),
            TimeProjection(features=self.emb_features * self.mlp_ratio), # Project to MLP dim
            nn.Dense(features=self.emb_features, dtype=self.dtype, precision=self.precision) # Final projection
        ])
        
        # Text context projection (if used)
        # Assuming textcontext is already projected to some dimension, project it to match emb_features
        # This might need adjustment based on how text context is provided
        self.text_proj = nn.Dense(features=self.emb_features, dtype=self.dtype, precision=self.precision, name="text_context_proj")

        # Rotary Positional Embedding
        # Max length needs to be estimated or set large enough. 
        # For images, seq len = (H/P) * (W/P). Example: 256/16 * 256/16 = 16*16 = 256
        # Add 1 if a class token is used, or more for text tokens if concatenated.
        # Let's assume max seq len accommodates patches + time + text tokens if needed, or just patches.
        # If only patches use RoPE, max_len = max_image_tokens
        # If time/text are concatenated *before* blocks, max_len needs to include them.
        # DiT typically applies PE only to patch tokens. Let's follow that.
        # max_len should be max number of patches.
        # Example: max image size 512x512, patch 16 -> (512/16)^2 = 32^2 = 1024 patches
        self.rope = RotaryEmbedding(dim=self.emb_features // self.num_heads, max_seq_len=4096, dtype=self.dtype) # Dim per head

        # Transformer Blocks
        self.blocks = [
            DiTBlock(
                features=self.emb_features,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                use_flash_attention=self.use_flash_attention,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                norm_epsilon=self.norm_epsilon,
                rope_emb=self.rope, # Pass RoPE instance
                name=f"dit_block_{i}"
            ) for i in range(self.num_layers)
        ]

        # Final Layer (Normalization + Linear Projection)
        self.final_norm = nn.LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="final_norm")
        # self.final_norm = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="final_norm")
        
        # Predict patch pixels + potentially sigma
        output_dim = self.patch_size * self.patch_size * self.output_channels
        if self.learn_sigma:
            output_dim *= 2 # Predict both mean and variance (or log_variance)
            
        self.final_proj = nn.Dense(
            features=output_dim, 
            dtype=self.dtype, 
            precision=self.precision,
            kernel_init=nn.initializers.zeros, # Initialize final layer to zero
            name="final_proj"
        )

    @nn.compact
    def __call__(self, x, temb, textcontext=None):
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        
        # 1. Patch Embedding
        patches = self.patch_embed(x) # Shape: [B, num_patches, emb_features]
        num_patches = patches.shape[1]

        # Optional Hilbert reorder
        if self.use_hilbert:
            idx = hilbert_indices(H // self.patch_size, W // self.patch_size)
            inv_idx = inverse_permutation(idx)
            patches = patches[:, idx, :]

        # replace x with patches
        x_seq = patches

        # 2. Prepare Conditioning Signal (Time + Text Context)
        t_emb = self.time_embed(temb) # Shape: [B, emb_features]
        
        cond_emb = t_emb
        if textcontext is not None:
            text_emb = self.text_proj(textcontext) # Shape: [B, num_text_tokens, emb_features]
            # Pool or select text embedding (e.g., mean pool or use CLS token)
            # Assuming mean pooling for simplicity
            text_emb_pooled = jnp.mean(text_emb, axis=1) # Shape: [B, emb_features]
            cond_emb = cond_emb + text_emb_pooled # Combine time and text embeddings

        # 3. Apply RoPE
        # Get RoPE frequencies for the sequence length (number of patches)
        freqs_cis = self.rope(seq_len=num_patches) # Shape [num_patches, D_head/2]

        # 4. Apply Transformer Blocks with adaLN-Zero conditioning
        for block in self.blocks:
            x_seq = block(x_seq, conditioning=cond_emb, freqs_cis=freqs_cis)

        # 5. Final Layer
        x_out = self.final_norm(x_seq)
        x_out = self.final_proj(x_out) # Shape: [B, num_patches, patch_pixels (*2 if learn_sigma)]

        # Optional Hilbert inverse reorder
        if self.use_hilbert:
            x_out = x_out[:, inv_idx, :]

        # 6. Unpatchify
        if self.learn_sigma:
            # Split into mean and variance predictions
            x_mean, x_logvar = jnp.split(x_out, 2, axis=-1)
            x = unpatchify(x_mean, channels=self.output_channels)
            # Return both mean and logvar if needed by the loss function
            # For now, just returning the mean prediction like standard diffusion models
            # logvar = unpatchify(x_logvar, channels=self.output_channels) 
            # return x, logvar 
            return x
        else:
            x = unpatchify(x_out, channels=self.output_channels) # Shape: [B, H, W, C]
            return x

