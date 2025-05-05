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
from .vit_common import _rotate_half, unpatchify, PatchEmbedding, apply_rotary_embedding, RotaryEmbedding, RoPEAttention, AdaLNZero, AdaLNParams
from .simple_dit import DiTBlock


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
            print(f"Warning: norm_groups > 0 not fully supported with standard LayerNorm fallback in UViT setup. Using LayerNorm.")
            self.norm_factory = partial(
                nn.LayerNorm, epsilon=self.norm_epsilon, dtype=self.dtype)
        else:
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
            self.hilbert_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection"
            )

        max_patches = (512 // self.patch_size)**2
        self.pos_encoding = self.param('pos_encoding',
                                       jax.nn.initializers.normal(stddev=0.02),
                                       (1, max_patches, self.emb_features))

        # --- Conditioning ---
        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features),
            TimeProjection(features=self.emb_features)
        ], name="time_embed")

        self.text_proj = nn.DenseGeneral(
            features=self.emb_features,
            dtype=self.dtype,
            precision=self.precision,
            name="text_proj"
        )

        # --- Transformer Blocks ---
        BlockClass = TransformerBlock 

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
            nn.DenseGeneral(
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
        self.final_norm = self.norm_factory(name="final_norm")

        patch_dim = self.patch_size ** 2 * self.output_channels
        self.final_proj = nn.Dense(
            features=patch_dim,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,
            name="final_proj"
        )

        if self.add_residualblock_output:
            self.final_conv1 = ConvLayer(
                "conv",
                features=64, kernel_size=(3, 3), strides=(1, 1),
                dtype=self.dtype, precision=self.precision, name="final_conv1"
            )
            self.final_norm_conv = self.norm_factory(
                name="final_norm_conv")
            self.final_conv2 = ConvLayer(
                "conv",
                features=self.output_channels, kernel_size=(3, 3), strides=(1, 1),
                dtype=jnp.float32,
                precision=self.precision, name="final_conv2"
            )
        else:
            self.final_conv_direct = ConvLayer(
                "conv",
                features=self.output_channels, kernel_size=(1, 1), strides=(1, 1),
                dtype=jnp.float32,
                precision=self.precision, name="final_conv_direct"
            )

    @nn.compact
    def __call__(self, x, temb, textcontext=None):
        original_img = x
        B, H, W, C = original_img.shape
        H_P = H // self.patch_size
        W_P = W // self.patch_size
        num_patches = H_P * W_P
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        hilbert_inv_idx = None
        if self.use_hilbert:
            patches_raw, hilbert_inv_idx_calc = hilbert_patchify(
                x, self.patch_size)
            x_patches = self.hilbert_proj(patches_raw)
            idx = hilbert_indices(H_P, W_P)
            hilbert_inv_idx = inverse_permutation(
                idx, total_size=num_patches)
            x_patches = x_patches[:, idx, :]
        else:
            x_patches = self.patch_embed(x)

        assert num_patches <= self.pos_encoding.shape[
            1], f"Number of patches {num_patches} exceeds max_len {self.pos_encoding.shape[1]} in positional encoding"
        x_patches = x_patches + self.pos_encoding[:, :num_patches, :]

        time_token = self.time_embed(temb.astype(
            jnp.float32))
        time_token = jnp.expand_dims(time_token.astype(
            self.dtype), axis=1)

        if textcontext is not None:
            text_tokens = self.text_proj(
                textcontext.astype(self.dtype))
            num_text_tokens = text_tokens.shape[1]
            x = jnp.concatenate([x_patches, time_token, text_tokens], axis=1)
        else:
            num_text_tokens = 0
            x = jnp.concatenate([x_patches, time_token], axis=1)

        skips = []
        for i in range(self.num_layers // 2):
            x = self.down_blocks[i](x)
            skips.append(x)

        x = self.mid_block(x)

        for i in range(self.num_layers // 2):
            skip_conn = skips.pop()
            x = jnp.concatenate([x, skip_conn], axis=-1)
            x = self.up_dense[i](x)
            x = self.up_blocks[i](x)

        x = self.final_norm(x)

        x_patches_out = x[:, :num_patches, :]

        x_patches_out = self.final_proj(x_patches_out)

        if self.use_hilbert:
            assert hilbert_inv_idx is not None, "Hilbert inverse index missing"
            x_image = hilbert_unpatchify(
                x_patches_out, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
        else:
            x_image = unpatchify(x_patches_out, channels=self.output_channels)

        if self.add_residualblock_output:
            x_image = jnp.concatenate(
                [original_img.astype(self.dtype), x_image], axis=-1)

            x_image = self.final_conv1(x_image)
            x_image = self.final_norm_conv(x_image)
            x_image = self.activation(x_image)
            x_image = self.final_conv2(x_image)
        else:
            pass

        return x_image


# --- Simple U-DiT ---

class SimpleUDiT(nn.Module):
    """
    A Simple U-Net Diffusion Transformer (U-DiT) implementation.
    Combines the U-Net structure with DiT blocks using RoPE and AdaLN-Zero conditioning.
    Based on SimpleDiT and standard U-Net principles.
    """
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12 # Should be even for U-Net structure
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout_rate: float = 0.0  # Typically 0 for diffusion
    dtype: Optional[Dtype] = None # e.g., jnp.float32 or jnp.bfloat16
    precision: PrecisionLike = None
    use_flash_attention: bool = False # Passed to DiTBlock -> RoPEAttention
    force_fp32_for_softmax: bool = True # Passed to DiTBlock -> RoPEAttention
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False
    use_hilbert: bool = False
    norm_groups: int = 0
    activation: Callable = jax.nn.swish

    def setup(self):
        assert self.num_layers % 2 == 0, "num_layers must be even for U-Net structure"
        half_layers = self.num_layers // 2

        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            embedding_dim=self.emb_features,
            dtype=self.dtype,
            precision=self.precision,
            name="patch_embed"
        )
        if self.use_hilbert:
            self.hilbert_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection"
            )

        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features, dtype=jnp.float32),
            TimeProjection(features=self.emb_features * self.mlp_ratio, dtype=self.dtype, precision=self.precision),
            nn.Dense(features=self.emb_features, dtype=self.dtype, precision=self.precision)
        ], name="time_embed")

        self.text_proj = nn.Dense(
            features=self.emb_features,
            dtype=self.dtype,
            precision=self.precision,
            name="text_proj"
        )

        max_patches = (512 // self.patch_size)**2
        self.rope = RotaryEmbedding(
            dim=self.emb_features // self.num_heads,
            max_seq_len=max_patches,
            dtype=self.dtype,
            name="rope_emb"
        )

        self.down_blocks = [
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
                rope_emb=self.rope,
                name=f"down_block_{i}"
            ) for i in range(half_layers)
        ]

        self.mid_block = DiTBlock(
            features=self.emb_features,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            precision=self.precision,
            use_flash_attention=self.use_flash_attention,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            norm_epsilon=self.norm_epsilon,
            rope_emb=self.rope,
            name="mid_block"
        )

        self.up_dense = [
             nn.DenseGeneral(
                 features=self.emb_features,
                 dtype=self.dtype,
                 precision=self.precision,
                 name=f"up_dense_{i}"
             ) for i in range(half_layers)
        ]
        self.up_blocks = [
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
                rope_emb=self.rope,
                name=f"up_block_{i}"
            ) for i in range(half_layers)
        ]

        self.final_norm = nn.LayerNorm(
            epsilon=self.norm_epsilon, dtype=self.dtype, name="final_norm")

        output_dim = self.patch_size * self.patch_size * self.output_channels
        if self.learn_sigma:
            output_dim *= 2

        self.final_proj = nn.Dense(
            features=output_dim,
            dtype=jnp.float32,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,
            name="final_proj"
        )

    @nn.compact
    def __call__(self, x, temb, textcontext=None):
        B, H, W, C = x.shape
        H_P = H // self.patch_size
        W_P = W // self.patch_size
        num_patches = H_P * W_P
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by patch size"

        x = x.astype(self.dtype)

        hilbert_inv_idx = None
        if self.use_hilbert:
            patches_raw, _ = hilbert_patchify(x, self.patch_size)
            x_seq = self.hilbert_proj(patches_raw)
            idx = hilbert_indices(H_P, W_P)
            hilbert_inv_idx = inverse_permutation(idx, total_size=num_patches)
        else:
            x_seq = self.patch_embed(x)

        t_emb = self.time_embed(temb.astype(jnp.float32))
        t_emb = t_emb.astype(self.dtype)

        cond_emb = t_emb
        if textcontext is not None:
            text_emb = self.text_proj(textcontext.astype(self.dtype))
            if text_emb.ndim == 3:
                text_emb = jnp.mean(text_emb, axis=1)
            cond_emb = cond_emb + text_emb

        skips = []
        for i in range(self.num_layers // 2):
            x_seq = self.down_blocks[i](x_seq, conditioning=cond_emb, freqs_cis=None)
            skips.append(x_seq)

        x_seq = self.mid_block(x_seq, conditioning=cond_emb, freqs_cis=None)

        for i in range(self.num_layers // 2):
            skip_conn = skips.pop()
            x_seq = jnp.concatenate([x_seq, skip_conn], axis=-1)
            x_seq = self.up_dense[i](x_seq)
            x_seq = self.up_blocks[i](x_seq, conditioning=cond_emb, freqs_cis=None)

        x_out = self.final_norm(x_seq)
        x_out = self.final_proj(x_out)

        if self.use_hilbert:
            assert hilbert_inv_idx is not None, "Hilbert inverse index missing"
            if self.learn_sigma:
                x_mean, x_logvar = jnp.split(x_out, 2, axis=-1)
                x_image = hilbert_unpatchify(x_mean, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
            else:
                x_image = hilbert_unpatchify(x_out, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
        else:
            if self.learn_sigma:
                x_mean, x_logvar = jnp.split(x_out, 2, axis=-1)
                x_image = unpatchify(x_mean, channels=self.output_channels)
            else:
                x_image = unpatchify(x_out, channels=self.output_channels)

        return x_image.astype(jnp.float32)
