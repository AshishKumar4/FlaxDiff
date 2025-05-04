# simple_vit.py

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Any, Optional, Tuple
from .simple_unet import FourierEmbedding, TimeProjection, ConvLayer, kernel_init
from .attention import TransformerBlock
from flaxdiff.models.simple_unet import FourierEmbedding, TimeProjection, ConvLayer, kernel_init, ResidualBlock
import einops
from flax.typing import Dtype, PrecisionLike
from functools import partial
from .hilbert import hilbert_indices, inverse_permutation, hilbert_patchify, hilbert_unpatchify


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


class UViT(nn.Module):
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12  # Should be even for U-Net structure
    num_heads: int = 12
    dropout_rate: float = 0.1  # Dropout is often 0 in diffusion models
    use_projection: bool = False  # In TransformerBlock MLP
    use_flash_attention: bool = False  # Passed to TransformerBlock
    # Passed to TransformerBlock (likely False for UViT)
    use_self_and_cross: bool = False
    force_fp32_for_softmax: bool = True  # Passed to TransformerBlock
    # Used in final convs if add_residualblock_output
    activation: Callable = jax.nn.swish
    norm_groups: int = 8
    dtype: Optional[Dtype] = None  # e.g., jnp.float32 or jnp.bfloat16
    precision: PrecisionLike = None
    add_residualblock_output: bool = False
    norm_inputs: bool = False  # Passed to TransformerBlock
    explicitly_add_residual: bool = True  # Passed to TransformerBlock
    norm_epsilon: float = 1e-5  # Adjusted default
    use_hilbert: bool = False  # Toggle Hilbert patch reorder
    use_remat: bool = False  # Add flag to use remat

    def setup(self):
        assert self.num_layers % 2 == 0, "num_layers must be even for U-Net structure"
        half_layers = self.num_layers // 2

        # --- Norm Layer ---
        if self.norm_groups > 0:
            # GroupNorm needs features arg, which varies. Define partial here, apply in __call__?
            # Or maybe use LayerNorm/RMSNorm consistently? Let's use LayerNorm for simplicity here.
            # If GroupNorm is essential, it needs careful handling with changing feature sizes.
            # self.norm_factory = partial(nn.GroupNorm, self.norm_groups, epsilon=self.norm_epsilon, dtype=self.dtype)
            print(f"Warning: norm_groups > 0 not fully supported with standard LayerNorm fallback in UViT setup. Using LayerNorm.")
            self.norm_factory = partial(
                nn.LayerNorm, epsilon=self.norm_epsilon, dtype=self.dtype)
        else:
            # Use LayerNorm or RMSNorm for sequence normalization
            # self.norm_factory = partial(nn.RMSNorm, epsilon=self.norm_epsilon, dtype=self.dtype)
            self.norm_factory = partial(
                nn.LayerNorm, epsilon=self.norm_epsilon, dtype=self.dtype)

        # --- Input Path ---
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            embedding_dim=self.emb_features,
            dtype=self.dtype,
            precision=self.precision,
            name="patch_embed"
        )
        if self.use_hilbert:
            # Projection layer needed after raw Hilbert patches
            self.hilbert_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection"
            )

        # Positional encoding (learned) - applied only to patch tokens
        # Max length needs to accommodate max possible patches
        # Example: 512x512 image, patch 16 -> (512/16)^2 = 1024 patches
        # Estimate max patches, adjust if needed
        max_patches = (512 // self.patch_size)**2
        self.pos_encoding = self.param('pos_encoding',
                                       # Standard init for ViT pos embeds
                                       jax.nn.initializers.normal(stddev=0.02),
                                       (1, max_patches, self.emb_features))

        # --- Conditioning ---
        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features),
            TimeProjection(features=self.emb_features,
                           dtype=self.dtype, precision=self.precision)
        ], name="time_embed")

        # Text projection (assuming textcontext input needs projection)
        self.text_proj = nn.DenseGeneral(
            features=self.emb_features,
            dtype=self.dtype,
            precision=self.precision,
            name="text_proj"
        )

        # --- Transformer Blocks ---
        BlockClass = TransformerBlock  # Use TransformerBlockRemat if self.use_remat

        self.down_blocks = [
            BlockClass(
                heads=self.num_heads,
                dim_head=self.emb_features // self.num_heads,
                dtype=self.dtype, precision=self.precision, use_projection=self.use_projection,
                use_flash_attention=self.use_flash_attention, use_self_and_cross=self.use_self_and_cross,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                only_pure_attention=False, norm_inputs=self.norm_inputs,
                explicitly_add_residual=self.explicitly_add_residual,
                norm_epsilon=self.norm_epsilon,
                name=f"down_block_{i}"
            ) for i in range(half_layers)
        ]

        self.mid_block = BlockClass(
            heads=self.num_heads,
            dim_head=self.emb_features // self.num_heads,
            dtype=self.dtype, precision=self.precision, use_projection=self.use_projection,
            use_flash_attention=self.use_flash_attention, use_self_and_cross=self.use_self_and_cross,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            only_pure_attention=False, norm_inputs=self.norm_inputs,
            explicitly_add_residual=self.explicitly_add_residual,
            norm_epsilon=self.norm_epsilon,
            name="mid_block"
        )

        self.up_dense = [
            nn.DenseGeneral(  # Project concatenated skip + up_path features back to emb_features
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name=f"up_dense_{i}"
            ) for i in range(half_layers)
        ]
        self.up_blocks = [
            BlockClass(
                heads=self.num_heads,
                dim_head=self.emb_features // self.num_heads,
                dtype=self.dtype, precision=self.precision, use_projection=self.use_projection,
                use_flash_attention=self.use_flash_attention, use_self_and_cross=self.use_self_and_cross,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                only_pure_attention=False, norm_inputs=self.norm_inputs,
                explicitly_add_residual=self.explicitly_add_residual,
                norm_epsilon=self.norm_epsilon,
                name=f"up_block_{i}"
            ) for i in range(half_layers)
        ]

        # --- Output Path ---
        self.final_norm = self.norm_factory(name="final_norm")  # Use factory

        patch_dim = self.patch_size ** 2 * self.output_channels
        self.final_proj = nn.Dense(
            features=patch_dim,
            dtype=self.dtype,  # Keep model dtype for projection
            precision=self.precision,
            kernel_init=nn.initializers.zeros,  # Zero init final layer
            name="final_proj"
        )

        if self.add_residualblock_output:
            # Define these layers only if needed
            self.final_conv1 = ConvLayer(
                features=64, kernel_size=(3, 3), strides=(1, 1),
                dtype=self.dtype, precision=self.precision, name="final_conv1"
            )
            self.final_norm_conv = self.norm_factory(
                name="final_norm_conv")  # Use factory
            self.final_conv2 = ConvLayer(
                features=self.output_channels, kernel_size=(3, 3), strides=(1, 1),
                dtype=jnp.float32,  # Often good to have final conv output float32
                precision=self.precision, name="final_conv2"
            )
        else:
            # Final conv to map features to output channels directly after unpatchify
            self.final_conv_direct = ConvLayer(
                # Use 1x1 conv
                features=self.output_channels, kernel_size=(1, 1), strides=(1, 1),
                dtype=jnp.float32,  # Output float32
                precision=self.precision, name="final_conv_direct"
            )

    @nn.compact
    def __call__(self, x, temb, textcontext=None):
        original_img = x  # Keep original for potential residual connection
        B, H, W, C = original_img.shape
        H_P = H // self.patch_size
        W_P = W // self.patch_size
        num_patches = H_P * W_P
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        # Ensure input dtype matches model dtype
        x = x.astype(self.dtype)

        # --- Patch Embedding ---
        hilbert_inv_idx = None
        if self.use_hilbert:
            # Use hilbert_patchify to get raw patches and inverse index
            patches_raw, hilbert_inv_idx_calc = hilbert_patchify(
                x, self.patch_size)  # Shape [B, S, P*P*C]
            # Project raw patches
            # Shape [B, S, emb_features]
            x_patches = self.hilbert_proj(patches_raw)
            # Calculate inverse permutation (needs total_size)
            idx = hilbert_indices(H_P, W_P)
            hilbert_inv_idx = inverse_permutation(
                idx, total_size=num_patches)  # Corrected call
            # Apply Hilbert reordering *after* projection
            x_patches = x_patches[:, idx, :]
        else:
            # Standard patch embedding
            # Shape: [B, num_patches, emb_features]
            x_patches = self.patch_embed(x)

        # --- Positional Encoding ---
        # Add positional encoding only to patch tokens
        assert num_patches <= self.pos_encoding.shape[
            1], f"Number of patches {num_patches} exceeds max_len {self.pos_encoding.shape[1]} in positional encoding"
        x_patches = x_patches + self.pos_encoding[:, :num_patches, :]

        # --- Conditioning Tokens ---
        # Time embedding: [B, D] -> [B, 1, D]
        time_token = self.time_embed(temb.astype(
            jnp.float32))  # Ensure input is float32
        time_token = jnp.expand_dims(time_token.astype(
            self.dtype), axis=1)  # Cast back and add seq dim

        # Text embedding: [B, S_text, D_in] -> [B, S_text, D]
        if textcontext is not None:
            text_tokens = self.text_proj(
                textcontext.astype(self.dtype))  # Cast context
            num_text_tokens = text_tokens.shape[1]
            # Concatenate: [Patches+Pos, Time, Text]
            x = jnp.concatenate([x_patches, time_token, text_tokens], axis=1)
        else:
            # Concatenate: [Patches+Pos, Time]
            num_text_tokens = 0
            x = jnp.concatenate([x_patches, time_token], axis=1)

        # --- U-Net Transformer ---
        skips = []
        # Down blocks (Encoder)
        for i in range(self.num_layers // 2):
            x = self.down_blocks[i](x)  # Pass full sequence (patches+cond)
            skips.append(x)  # Store output for skip connection

        # Middle block
        x = self.mid_block(x)

        # Up blocks (Decoder)
        for i in range(self.num_layers // 2):
            skip_conn = skips.pop()
            # Concatenate along feature dimension
            x = jnp.concatenate([x, skip_conn], axis=-1)
            # Project back to emb_features
            x = self.up_dense[i](x)
            # Apply transformer block
            x = self.up_blocks[i](x)

        # --- Output Processing ---
        # Normalize before final projection
        x = self.final_norm(x)  # Apply norm factory instance

        # Extract only the image patch tokens (first num_patches tokens)
        # Conditioning tokens (time, text) are discarded here
        x_patches_out = x[:, :num_patches, :]

        # Project to patch pixel dimensions
        # Shape: [B, num_patches, patch_dim]
        x_patches_out = self.final_proj(x_patches_out)

        # --- Unpatchify ---
        if self.use_hilbert:
            # Restore Hilbert order to row-major order and then to image
            assert hilbert_inv_idx is not None, "Hilbert inverse index missing"
            x_image = hilbert_unpatchify(
                x_patches_out, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
        else:
            # Standard unpatchify
            # Shape: [B, H, W, C_out]
            x_image = unpatchify(x_patches_out, channels=self.output_channels)

        # --- Final Convolutions ---
        if self.add_residualblock_output:
            # Concatenate the original image (ensure dtype matches)
            x_image = jnp.concatenate(
                [original_img.astype(self.dtype), x_image], axis=-1)

            x_image = self.final_conv1(x_image)
            # Apply norm factory instance
            x_image = self.final_norm_conv(x_image)
            x_image = self.activation(x_image)
            x_image = self.final_conv2(x_image)  # Outputs float32
        else:
            # Apply a simple 1x1 conv to map features if needed (unpatchify already gives C_out channels)
            # Or just return x_image if channels match output_channels
            # If unpatchify output channels == self.output_channels, this might be redundant
            # Let's assume unpatchify gives correct channels, but ensure float32
            # x_image = self.final_conv_direct(x_image) # Use 1x1 conv if needed
            pass  # Assuming unpatchify output is correct

        # Ensure final output is float32
        return x_image.astype(jnp.float32)
