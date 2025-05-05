import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Any, Optional, Tuple, Sequence, Union
import einops
from functools import partial

# Re-use existing components if they are suitable
from .vit_common import PatchEmbedding, unpatchify, RotaryEmbedding, RoPEAttention, AdaLNParams
from .common import kernel_init, FourierEmbedding, TimeProjection
# Using NormalAttention for RoPE integration
from .attention import NormalAttention
from flax.typing import Dtype, PrecisionLike

# Use our improved Hilbert implementation
from .hilbert import hilbert_indices, inverse_permutation, hilbert_patchify, hilbert_unpatchify

# --- DiT Block ---
class DiTBlock(nn.Module):
    features: int
    num_heads: int
    rope_emb: RotaryEmbedding
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_flash_attention: bool = False # Keep placeholder
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    use_gating: bool = True # Add flag to easily disable gating

    def setup(self):
        hidden_features = int(self.features * self.mlp_ratio)
        # Get modulation parameters (scale, shift, gates)
        self.ada_params_module = AdaLNParams( # Use the modified module
            self.features, dtype=self.dtype, precision=self.precision)

        # Layer Norms - one before Attn, one before MLP
        self.norm1 = nn.LayerNorm(epsilon=self.norm_epsilon, use_scale=False, use_bias=False, dtype=self.dtype, name="norm1")
        self.norm2 = nn.LayerNorm(epsilon=self.norm_epsilon, use_scale=False, use_bias=False, dtype=self.dtype, name="norm2")

        self.attention = RoPEAttention(
            query_dim=self.features,
            heads=self.num_heads,
            dim_head=self.features // self.num_heads,
            dtype=self.dtype,
            precision=self.precision,
            use_bias=True,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            rope_emb=self.rope_emb
        )

        self.mlp = nn.Sequential([
            nn.Dense(features=hidden_features, dtype=self.dtype, precision=self.precision),
            nn.gelu, # Or swish as specified in SimpleDiT? Consider consistency.
            nn.Dense(features=self.features, dtype=self.dtype, precision=self.precision)
        ])

    @nn.compact
    def __call__(self, x, conditioning, freqs_cis):
        # Get scale/shift/gate parameters
        # Shape: [B, 1, 6*F] -> split into 6 of [B, 1, F]
        scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn = jnp.split(
             self.ada_params_module(conditioning), 6, axis=-1
        )

        # --- Attention Path ---
        residual = x
        norm_x_attn = self.norm1(x)
        # Modulate after norm
        x_attn_modulated = norm_x_attn * (1 + scale_attn) + shift_attn
        attn_output = self.attention(x_attn_modulated, context=None, freqs_cis=freqs_cis)

        if self.use_gating:
             x = residual + gate_attn * attn_output
        else:
             x = residual + attn_output # Original DiT style without gate

        # --- MLP Path ---
        residual = x
        norm_x_mlp = self.norm2(x) # Apply second LayerNorm
        # Modulate after norm
        x_mlp_modulated = norm_x_mlp * (1 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(x_mlp_modulated)

        if self.use_gating:
             x = residual + gate_mlp * mlp_output
        else:
             x = residual + mlp_output # Original DiT style without gate

        return x


# --- Patch Embedding (reuse or define if needed) ---
# Assuming PatchEmbedding exists in simple_vit.py and is suitable

# --- DiT ---

class SimpleDiT(nn.Module):
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

        # Add projection layer for Hilbert patches
        if self.use_hilbert:
            self.hilbert_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection"
            )

        # Time embedding projection
        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features),
            TimeProjection(features=self.emb_features *
                           self.mlp_ratio),  # Project to MLP dim
            nn.Dense(features=self.emb_features, dtype=self.dtype,
                     precision=self.precision)  # Final projection
        ])

        # Text context projection (if used)
        # Assuming textcontext is already projected to some dimension, project it to match emb_features
        # This might need adjustment based on how text context is provided
        self.text_proj = nn.Dense(features=self.emb_features, dtype=self.dtype,
                                  precision=self.precision, name="text_context_proj")

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
        self.rope = RotaryEmbedding(
            dim=self.emb_features // self.num_heads, max_seq_len=4096, dtype=self.dtype)  # Dim per head

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
                rope_emb=self.rope,  # Pass RoPE instance
                name=f"dit_block_{i}"
            ) for i in range(self.num_layers)
        ]

        # Final Layer (Normalization + Linear Projection)
        self.final_norm = nn.LayerNorm(
            epsilon=self.norm_epsilon, dtype=self.dtype, name="final_norm")
        # self.final_norm = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="final_norm")

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
    def __call__(self, x, temb, textcontext=None):
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        # Compute dimensions in terms of patches
        H_P = H // self.patch_size
        W_P = W // self.patch_size

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

        # 2. Prepare Conditioning Signal (Time + Text Context)
        t_emb = self.time_embed(temb)  # Shape: [B, emb_features]

        cond_emb = t_emb
        if textcontext is not None:
            # Shape: [B, num_text_tokens, emb_features]
            text_emb = self.text_proj(textcontext)
            # Pool or select text embedding (e.g., mean pool or use CLS token)
            # Assuming mean pooling for simplicity
            # Shape: [B, emb_features]
            text_emb_pooled = jnp.mean(text_emb, axis=1)
            cond_emb = cond_emb + text_emb_pooled  # Combine time and text embeddings

        # 3. Apply RoPE
        # Get RoPE frequencies for the sequence length (number of patches)
        # Shape [num_patches, D_head/2]
        freqs_cos, freqs_sin = self.rope(seq_len=num_patches)

        # 4. Apply Transformer Blocks with adaLN-Zero conditioning
        for block in self.blocks:
            x_seq = block(x_seq, conditioning=cond_emb, freqs_cis=(freqs_cos, freqs_sin))

        # 5. Final Layer
        x_out = self.final_norm(x_seq)
        # Shape: [B, num_patches, patch_pixels (*2 if learn_sigma)]
        x_out = self.final_proj(x_out)

        # 6. Unpatchify
        if self.use_hilbert:
            # For Hilbert mode, we need to use the specialized unpatchify function
            if self.learn_sigma:
                # Split into mean and variance predictions
                x_mean, x_logvar = jnp.split(x_out, 2, axis=-1)
                x_image = hilbert_unpatchify(x_mean, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
                # If needed, also unpack the logvar
                # logvar_image = hilbert_unpatchify(x_logvar, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
                # return x_image, logvar_image
                return x_image
            else:
                x_image = hilbert_unpatchify(x_out, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
                return x_image
        else:
            # Standard patch ordering - use the existing unpatchify function
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
                # Shape: [B, H, W, C]
                x = unpatchify(x_out, channels=self.output_channels)
                return x
