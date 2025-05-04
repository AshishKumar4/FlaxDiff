import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Any, Optional, Tuple, Sequence, Union, List
import einops
from functools import partial
from flax.typing import Dtype, PrecisionLike

# Imports from local modules
from .simple_vit import PatchEmbedding, unpatchify
from .common import kernel_init, FourierEmbedding, TimeProjection
from .attention import NormalAttention  # Base for RoPEAttention
# Replace common.hilbert_indices with improved implementation from hilbert.py
from .hilbert import hilbert_indices, inverse_permutation, hilbert_patchify, hilbert_unpatchify

# --- Rotary Positional Embedding (RoPE) ---
# Re-used from simple_dit.py


def _rotate_half(x: jax.Array) -> jax.Array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_embedding(
    x: jax.Array, freqs_cos: jax.Array, freqs_sin: jax.Array
) -> jax.Array:
    """Applies rotary embedding to the input tensor using rotate_half method."""
    if x.ndim == 4:  # [B, H, S, D]
        cos_freqs = jnp.expand_dims(freqs_cos, axis=(0, 1))
        sin_freqs = jnp.expand_dims(freqs_sin, axis=(0, 1))
    elif x.ndim == 3:  # [B, S, D]
        cos_freqs = jnp.expand_dims(freqs_cos, axis=0)
        sin_freqs = jnp.expand_dims(freqs_sin, axis=0)
    else:
        raise ValueError(f"Unsupported input dimension: {x.ndim}")

    cos_freqs = jnp.concatenate([cos_freqs, cos_freqs], axis=-1)
    sin_freqs = jnp.concatenate([sin_freqs, sin_freqs], axis=-1)

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
# Re-used from simple_dit.py


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


# --- MM-DiT AdaLN-Zero ---
class MMAdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization Zero (AdaLN-Zero) tailored for MM-DiT.
    Projects time and text embeddings separately, combines them, and then
    generates modulation parameters (scale, shift, gate) for attention and MLP paths.
    """
    features: int
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    norm_epsilon: float = 1e-5
    use_mean_pooling: bool = True  # Whether to use mean pooling for sequence inputs

    @nn.compact
    def __call__(self, x, t_emb, text_emb):
        # x shape: [B, S, F]
        # t_emb shape: [B, D_t]
        # text_emb shape: [B, S_text, D_text] or [B, D_text]

        # First normalize the input features
        norm = nn.LayerNorm(epsilon=self.norm_epsilon,
                            use_scale=False, use_bias=False, dtype=self.dtype)
        norm_x = norm(x)  # Shape: [B, S, F]

        # Process time embedding: ensure it has a sequence dimension for later broadcasting
        if t_emb.ndim == 2:  # [B, D_t]
            t_emb = jnp.expand_dims(t_emb, axis=1)  # [B, 1, D_t]
            
        # Process text embedding: if it has a sequence dimension different from x
        if text_emb.ndim == 2:  # [B, D_text]
            text_emb = jnp.expand_dims(text_emb, axis=1)  # [B, 1, D_text]
        elif text_emb.ndim == 3 and self.use_mean_pooling and text_emb.shape[1] != x.shape[1]:
            # Mean pooling is standard in MM-DiT for handling different sequence lengths
            text_emb = jnp.mean(text_emb, axis=1, keepdims=True)  # [B, 1, D_text]

        # Project time embedding 
        t_params = nn.Dense(
            features=6 * self.features,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,  # Zero init is standard in AdaLN-Zero
            name="ada_t_proj"
        )(t_emb)  # Shape: [B, 1, 6*F]

        # Project text embedding
        text_params = nn.Dense(
            features=6 * self.features,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,  # Zero init
            name="ada_text_proj"
        )(text_emb)  # Shape: [B, 1, 6*F] or [B, S_text, 6*F]

        # If text_params still has a sequence dim different from t_params, mean pool it
        if t_params.shape[1] != text_params.shape[1]:
            text_params = jnp.mean(text_params, axis=1, keepdims=True)
            
        # Combine parameters (summing is standard in MM-DiT)
        ada_params = t_params + text_params  # Shape: [B, 1, 6*F]

        # Split into scale, shift, gate for MLP and Attention
        scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn = jnp.split(
            ada_params, 6, axis=-1)  # Each shape: [B, 1, F]

        scale_mlp = jnp.clip(scale_mlp, -10.0, 10.0)
        shift_mlp = jnp.clip(shift_mlp, -10.0, 10.0)
        # Apply modulation for Attention path (broadcasting handled by JAX)
        x_attn = norm_x * (1 + scale_attn) + shift_attn

        # Apply modulation for MLP path
        x_mlp = norm_x * (1 + scale_mlp) + shift_mlp

        # Return modulated outputs and gates
        return x_attn, gate_attn, x_mlp, gate_mlp


# --- MM-DiT Block ---
class MMDiTBlock(nn.Module):
    """
    A Transformer block adapted for MM-DiT, using MMAdaLNZero for conditioning.
    """
    features: int
    num_heads: int
    rope_emb: RotaryEmbedding  # Pass RoPE module
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    # Keep option, though RoPEAttention doesn't use it
    use_flash_attention: bool = False
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5

    def setup(self):
        hidden_features = int(self.features * self.mlp_ratio)
        # Use the new MMAdaLNZero block
        self.ada_ln_zero = MMAdaLNZero(
            self.features, dtype=self.dtype, precision=self.precision, norm_epsilon=self.norm_epsilon)

        # RoPEAttention remains the same
        self.attention = RoPEAttention(
            query_dim=self.features,
            heads=self.num_heads,
            dim_head=self.features // self.num_heads,
            dtype=self.dtype,
            precision=self.precision,
            use_bias=True,  # Bias is common in DiT attention proj
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            rope_emb=self.rope_emb  # Pass RoPE module instance
        )

        # Standard MLP block remains the same
        self.mlp = nn.Sequential([
            nn.Dense(features=hidden_features, dtype=self.dtype,
                     precision=self.precision),
            nn.gelu,  # Consider swish/silu if preferred
            nn.Dense(features=self.features, dtype=self.dtype,
                     precision=self.precision)
        ])

    @nn.compact
    def __call__(self, x, t_emb, text_emb, freqs_cis):
        # x shape: [B, S, F]
        # t_emb shape: [B, D_t] or [B, 1, D_t]
        # text_emb shape: [B, D_text] or [B, 1, D_text]

        residual = x

        # Apply MMAdaLNZero with separate time and text embeddings
        x_attn, gate_attn, x_mlp, gate_mlp = self.ada_ln_zero(
            x, t_emb, text_emb)

        # Attention block (remains the same)
        attn_output = self.attention(
            x_attn, context=None, freqs_cis=freqs_cis)  # Self-attention only
        x = residual + gate_attn * attn_output

        # MLP block (remains the same)
        mlp_output = self.mlp(x_mlp)
        x = x + gate_mlp * mlp_output

        return x


# --- SimpleMMDiT ---
class SimpleMMDiT(nn.Module):
    """
    A Simple Multi-Modal Diffusion Transformer (MM-DiT) implementation.
    Integrates time and text conditioning using separate projections within
    each transformer block, following the MM-DiT approach. Uses RoPE for
    patch positional encoding.
    """
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout_rate: float = 0.0  # Typically 0 for diffusion
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    # Passed down, but RoPEAttention uses NormalAttention
    use_flash_attention: bool = False
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False  # Option to predict sigma like in DiT paper
    use_hilbert: bool = False  # Toggle Hilbert patch reorder
    norm_groups: int = 0
    activation: Callable = jax.nn.swish

    def setup(self):
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            embedding_dim=self.emb_features,
            dtype=self.dtype,
            precision=self.precision
        )

        # Time embedding projection (output dim: emb_features)
        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features),
            TimeProjection(features=self.emb_features *
                           self.mlp_ratio),  # Intermediate projection
            nn.Dense(features=self.emb_features, dtype=self.dtype,
                     precision=self.precision)  # Final projection
        ], name="time_embed")

        # Add projection layer for Hilbert patches
        if self.use_hilbert:
            self.hilbert_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection"
            )
        # Text context projection (output dim: emb_features)
        # Input dim depends on the text encoder output, assumed to be handled externally
        self.text_proj = nn.Dense(features=self.emb_features, dtype=self.dtype,
                                  precision=self.precision, name="text_context_proj")

        # Rotary Positional Embedding (for patches)
        # Dim per head, max_len should cover max number of patches
        self.rope = RotaryEmbedding(
            dim=self.emb_features // self.num_heads, max_seq_len=4096, dtype=self.dtype)

        # Transformer Blocks (use MMDiTBlock)
        self.blocks = [
            MMDiTBlock(
                features=self.emb_features,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                use_flash_attention=self.use_flash_attention,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                norm_epsilon=self.norm_epsilon,
                rope_emb=self.rope,  # Pass RoPE instance
                name=f"mmdit_block_{i}"
            ) for i in range(self.num_layers)
        ]

        # Final Layer (Normalization + Linear Projection)
        self.final_norm = nn.LayerNorm(
            epsilon=self.norm_epsilon, dtype=self.dtype, name="final_norm")
        # self.final_norm = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="final_norm") # Alternative

        # Predict patch pixels + potentially sigma
        output_dim = self.patch_size * self.patch_size * self.output_channels
        if self.learn_sigma:
            output_dim *= 2  # Predict both mean and variance (or log_variance)

        self.final_proj = nn.Dense(
            features=output_dim,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,  # Initialize final layer to zero
            name="final_proj"
        )

    @nn.compact
    def __call__(self, x, temb, textcontext):  # textcontext is required
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"
        assert textcontext is not None, "textcontext must be provided for SimpleMMDiT"

        # 1. Patch Embedding
        if self.use_hilbert:
            # Use hilbert_patchify which handles both patchification and reordering
            patches_raw, hilbert_inv_idx = hilbert_patchify(x, self.patch_size) # Shape [B, S, P*P*C]
            # Apply projection
            patches = self.hilbert_proj(patches_raw) # Shape [B, S, emb_features]
        else:
            patches = self.patch_embed(x)  # Shape: [B, num_patches, emb_features]
            hilbert_inv_idx = None

        num_patches = patches.shape[1]
        x_seq = patches

        # 2. Prepare Conditioning Signals
        t_emb = self.time_embed(temb)      # Shape: [B, emb_features]
        # Assuming textcontext is [B, context_seq_len, context_dim] or [B, context_dim]
        # If [B, context_seq_len, context_dim], usually mean/pool or take CLS token first.
        # Assuming textcontext is already pooled/CLS token: [B, context_dim]
        text_emb = self.text_proj(textcontext)  # Shape: [B, emb_features]

        # 3. Apply RoPE Frequencies (only to patch tokens)
        seq_len = x_seq.shape[1]
        freqs_cos, freqs_sin = self.rope(seq_len)  # Shapes: [S, D_head/2]

        # 4. Apply Transformer Blocks
        for block in self.blocks:
            # Pass t_emb and text_emb separately to the block
            x_seq = block(x_seq, t_emb, text_emb,
                          freqs_cis=(freqs_cos, freqs_sin))

        # 5. Final Layer
        x_seq = self.final_norm(x_seq)
        # Shape: [B, num_patches, P*P*C (*2 if learn_sigma)]
        x_seq = self.final_proj(x_seq)

        # 6. Unpatchify
        if self.use_hilbert:
            # For Hilbert mode, we need to use the specialized unpatchify function
            if self.learn_sigma:
                # Split into mean and variance predictions
                x_mean, x_logvar = jnp.split(x_seq, 2, axis=-1)
                x_image = hilbert_unpatchify(x_mean, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
                # If needed, also unpack the logvar
                # logvar_image = hilbert_unpatchify(x_logvar, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
                # return x_image, logvar_image
                return x_image
            else:
                x_image = hilbert_unpatchify(x_seq, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
                return x_image
        else:
            # Standard patch ordering - use the existing unpatchify function
            if self.learn_sigma:
                # Split into mean and variance predictions
                x_mean, x_logvar = jnp.split(x_seq, 2, axis=-1)
                x = unpatchify(x_mean, channels=self.output_channels)
                # Return both mean and logvar if needed by the loss function
                # For now, just returning the mean prediction like standard diffusion models
                # logvar = unpatchify(x_logvar, channels=self.output_channels)
                # return x, logvar
                return x
            else:
                # Shape: [B, H, W, C]
                x = unpatchify(x_seq, channels=self.output_channels)
                return x



# --- Hierarchical MM-DiT components ---

class PatchMerging(nn.Module):
    """
    Merges a group of patches into a single patch with increased feature dimensions.
    Used in the hierarchical structure to reduce spatial resolution and increase channels.
    """
    out_features: int
    merge_size: int = 2  # Default 2x2 patch merging
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, H_patches, W_patches):
        # x shape: [B, H*W, C]
        B, L, C = x.shape
        assert L == H_patches * W_patches, f"Input length {L} doesn't match {H_patches}*{W_patches}"
        
        # Reshape to [B, H, W, C]
        x = x.reshape(B, H_patches, W_patches, C)
        
        # Merge patches - rearrange to group nearby patches
        merged = einops.rearrange(
            x, 
            'b (h p1) (w p2) c -> b h w (p1 p2 c)',
            p1=self.merge_size, p2=self.merge_size
        )
        
        # Project to new dimension
        merged = nn.Dense(
            features=self.out_features, 
            dtype=self.dtype,
            precision=self.precision
        )(merged)
        
        # Flatten back to sequence
        new_H = H_patches // self.merge_size
        new_W = W_patches // self.merge_size
        merged = merged.reshape(B, new_H * new_W, self.out_features)
        
        return merged, new_H, new_W

class PatchExpanding(nn.Module):
    """
    Expands patches to increase spatial resolution.
    Used in the hierarchical structure decoder path.
    """
    out_features: int
    expand_size: int = 2  # Default 2x2 patch expansion
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, H_patches, W_patches):
        # x shape: [B, H*W, C]
        B, L, C = x.shape
        assert L == H_patches * W_patches, f"Input length {L} doesn't match {H_patches}*{W_patches}"
        
        # Reshape to [B, H, W, C]
        x = x.reshape(B, H_patches, W_patches, C)
        
        # Project to expanded dimension
        expanded_features = self.expand_size * self.expand_size * self.out_features
        x = nn.Dense(
            features=expanded_features,
            dtype=self.dtype,
            precision=self.precision
        )(x)
        
        # Rearrange to expand spatial dims
        expanded = einops.rearrange(
            x,
            'b h w (p1 p2 c) -> b (h p1) (w p2) c',
            p1=self.expand_size, p2=self.expand_size, c=self.out_features
        )
        
        # Flatten back to sequence
        new_H = H_patches * self.expand_size
        new_W = W_patches * self.expand_size
        expanded = expanded.reshape(B, new_H * new_W, self.out_features)
        
        return expanded, new_H, new_W


# --- Hierarchical MM-DiT ---
class HierarchicalMMDiT(nn.Module):
    """
    A Hierarchical Multi-Modal Diffusion Transformer (MM-DiT) implementation
    based on the PixArt-α architecture. Processes images at multiple resolutions
    with skip connections between encoder and decoder paths.
    """
    output_channels: int = 3
    base_patch_size: int = 8  # Initial patch size
    emb_features: Sequence[int] = (512, 768, 1024)  # Feature dimensions for each stage
    num_layers: Sequence[int] = (4, 4, 14)  # Layers per stage
    num_heads: Sequence[int] = (8, 12, 16)  # Heads per stage
    mlp_ratio: int = 4
    dropout_rate: float = 0.0 
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_flash_attention: bool = False
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False
    use_hilbert: bool = False
    norm_groups: int = 0
    activation: Callable = jax.nn.swish
    
    def setup(self):
        assert len(self.emb_features) == len(self.num_layers) == len(self.num_heads), \
            "Feature dimensions, layers, and heads must have the same number of stages"
        
        num_stages = len(self.emb_features)
        
        # Initial patch embedding (coarsest level)
        self.patch_embed = PatchEmbedding(
            patch_size=self.base_patch_size * (2 ** (num_stages - 1)),
            embedding_dim=self.emb_features[-1],  # Start with the largest embedding
            dtype=self.dtype,
            precision=self.precision
        )

        # Time embedding projection
        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features[-1]),  # Use largest dim
            TimeProjection(features=self.emb_features[-1] * self.mlp_ratio),
            nn.Dense(features=self.emb_features[-1], dtype=self.dtype, precision=self.precision)
        ], name="time_embed")

        # Text context projection
        self.text_proj = nn.Dense(
            features=self.emb_features[-1],  # Use largest dim 
            dtype=self.dtype,
            precision=self.precision, 
            name="text_context_proj"
        )
        
        # Add projection layer for Hilbert patches
        if self.use_hilbert:
            self.hilbert_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection"
            )

        # Create RoPE embeddings for each stage
        self.ropes = [
            RotaryEmbedding(
                dim=self.emb_features[i] // self.num_heads[i], 
                max_seq_len=4096, 
                dtype=self.dtype
            )
            for i in range(num_stages)
        ]
        
        # Encoder blocks (from coarse to fine)
        self.encoder_blocks = []
        for stage in range(num_stages):
            stage_blocks = [
                MMDiTBlock(
                    features=self.emb_features[stage],
                    num_heads=self.num_heads[stage],
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype,
                    precision=self.precision,
                    use_flash_attention=self.use_flash_attention,
                    force_fp32_for_softmax=self.force_fp32_for_softmax,
                    norm_epsilon=self.norm_epsilon,
                    rope_emb=self.ropes[stage],
                    name=f"encoder_block_stage{stage}_{i}"
                ) 
                for i in range(self.num_layers[stage] // 2)  # Half for encoder, half for decoder
            ]
            self.encoder_blocks.append(stage_blocks)
        
        # Patch expanding layers (from coarse to fine)
        if num_stages > 1:
            self.patch_expanders = [
                PatchExpanding(
                    out_features=self.emb_features[stage-1],  # Target: next finer resolution
                    dtype=self.dtype,
                    precision=self.precision,
                    name=f"patch_expander_{stage}"
                )
                for stage in range(num_stages-1, 0, -1)
            ]
        
        # Decoder blocks (from coarse to fine)
        self.decoder_blocks = []
        for stage in range(num_stages-1, -1, -1):
            stage_blocks = [
                MMDiTBlock(
                    features=self.emb_features[stage],
                    num_heads=self.num_heads[stage],
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype,
                    precision=self.precision,
                    use_flash_attention=self.use_flash_attention,
                    force_fp32_for_softmax=self.force_fp32_for_softmax,
                    norm_epsilon=self.norm_epsilon,
                    rope_emb=self.ropes[stage],
                    name=f"decoder_block_stage{stage}_{i}"
                ) 
                for i in range(self.num_layers[stage] // 2)  # Half for encoder, half for decoder
            ]
            self.decoder_blocks.append(stage_blocks)
            
        # Fusion layers for skip connections
        if num_stages > 1:
            self.fusion_layers = [
                nn.Dense(
                    features=self.emb_features[stage],
                    dtype=self.dtype,
                    precision=self.precision,
                    name=f"fusion_layer_{stage}"
                )
                for stage in range(num_stages-1, -1, -1)
            ]
        
        # Final Layer (Normalization + Linear Projection)
        self.final_norm = nn.LayerNorm(
            epsilon=self.norm_epsilon, dtype=self.dtype, name="final_norm")
        
        # Output projection to pixels
        output_dim = self.base_patch_size * self.base_patch_size * self.output_channels
        if self.learn_sigma:
            output_dim *= 2  # Predict both mean and variance
            
        self.final_proj = nn.Dense(
            features=output_dim,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,  # Zero init
            name="final_proj"
        )

    def __call__(self, x, temb, textcontext):
        B, H, W, C = x.shape
        num_stages = len(self.emb_features)
        
        # Calculate base patch dimensions
        finest_patch_size = self.base_patch_size
        coarsest_patch_size = finest_patch_size * (2 ** (num_stages - 1))
        
        assert H % coarsest_patch_size == 0 and W % coarsest_patch_size == 0, \
            f"Image dimensions must be divisible by coarsest patch size {coarsest_patch_size}"
        assert textcontext is not None, "textcontext must be provided"
        
        # Start with coarsest patch embedding
        patches = self.patch_embed(x)  # Shape: [B, num_patches, emb_features[-1]]
        H_patches = H // coarsest_patch_size
        W_patches = W // coarsest_patch_size
        num_patches = H_patches * W_patches
        
        # Optional Hilbert reorder at coarsest level
        hilbert_inv_idx = None
        if self.use_hilbert:
            idx = hilbert_indices(H_patches, W_patches)
            hilbert_inv_idx = inverse_permutation(idx)
            patches = patches[:, idx, :]
            
        x_seq = patches  # Start sequence at coarsest level
        
        # Prepare conditioning signals
        t_emb = self.time_embed(temb)  # Shape: [B, emb_features[-1]]
        text_emb = self.text_proj(textcontext)  # Shape: [B, emb_features[-1]]
        
        # Project conditioning to each stage's dimension if needed
        t_embs = []
        text_embs = []
        for stage in range(num_stages):
            if stage == num_stages - 1:  # Coarsest stage, use original embeddings
                t_embs.append(t_emb)
                text_embs.append(text_emb)
            else:
                # Project to appropriate dimension for this stage
                t_embs.append(
                    nn.Dense(
                        features=self.emb_features[stage],
                        dtype=self.dtype,
                        precision=self.precision,
                        name=f"t_emb_proj_stage{stage}"
                    )(t_emb)
                )
                text_embs.append(
                    nn.Dense(
                        features=self.emb_features[stage],
                        dtype=self.dtype,
                        precision=self.precision,
                        name=f"text_emb_proj_stage{stage}"
                    )(text_emb)
                )
                
        # --- Encoder Path (coarse to fine) ---
        skip_features = []
        current_stage = num_stages - 1  # Start at coarsest stage
        
        # For each stage in encoder (coarse to fine)
        for stage in range(num_stages - 1, -1, -1):
            # Get RoPE frequencies for current sequence length
            seq_len = x_seq.shape[1]
            freqs_cos, freqs_sin = self.ropes[stage](seq_len)
            
            # Apply transformer blocks for this stage
            for block in self.encoder_blocks[stage]:
                x_seq = block(
                    x_seq, 
                    t_embs[stage], 
                    text_embs[stage], 
                    freqs_cis=(freqs_cos, freqs_sin)
                )
                
            # Store features for skip connection
            skip_features.append(x_seq)
            
            # Early exit on finest level
            if stage == 0:
                break
                
        # --- Decoder Path (fine to coarse) ---
        current_stage = 0  # Start at finest stage
        
        # For each stage in decoder
        for stage_idx, stage in enumerate(range(0, num_stages)):
            # First block - use the encoder output directly
            if stage_idx == 0:
                decoder_input = x_seq
            else:
                # For subsequent stages, we need to expand patches from previous stage
                x_seq, H_patches, W_patches = self.patch_expanders[stage_idx-1](
                    x_seq, H_patches, W_patches
                )
                
                # Fusion with skip connection
                skip = skip_features[num_stages - stage - 1]
                x_seq = jnp.concatenate([x_seq, skip], axis=-1)
                x_seq = self.fusion_layers[stage_idx-1](x_seq)
            
            # Get RoPE frequencies for current sequence length
            seq_len = x_seq.shape[1]
            freqs_cos, freqs_sin = self.ropes[stage](seq_len)
            
            # Apply transformer blocks for this stage
            for block in self.decoder_blocks[stage_idx]:
                x_seq = block(
                    x_seq, 
                    t_embs[stage], 
                    text_embs[stage], 
                    freqs_cis=(freqs_cos, freqs_sin)
                )
                
        # Final processing - should now be at finest resolution
        x_seq = self.final_norm(x_seq)
        x_seq = self.final_proj(x_seq)
        
        # Undo Hilbert ordering if used
        if self.use_hilbert and hilbert_inv_idx is not None:
            x_seq = x_seq[:, hilbert_inv_idx, :]
            
        # Determine output channels for unpatchify
        final_out_channels = self.output_channels * (2 if self.learn_sigma else 1)
        
        # Reshape back to image space
        out = unpatchify(x_seq, channels=final_out_channels)
        
        return out
