"""
UNet3D: the 2D UNet inflated for video, AnimateDiff-style.

The forward mirrors Unet exactly - same blocks, same explicit module names,
same auto-name ordering - processing every frame through the spatial path,
with zero-initialized temporal attention blocks interleaved at each
resolution level. Zero init means a freshly inflated model reproduces the 2D
UNet frame by frame, so a pretrained image checkpoint (inflate_unet_params)
is the starting point and training only has to learn motion.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from typing import Callable, Optional
from functools import partial

from .common import ConvLayer, Downsample, Upsample, FourierEmbedding, TimeProjection, ResidualBlock
from .attention import TransformerBlock
from .vit_common import RotaryEmbedding, RoPEAttention


class TemporalBlock(nn.Module):
    """Temporal self-attention over the frame axis at every spatial position.

    The output projection is zero-initialized, so the block is an exact
    identity at init and an inflated model starts as the per-frame 2D model.
    """
    features: int
    heads: int = 8
    norm_epsilon: float = 1e-5
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x, frames: int):
        # x: [B*T, H, W, C] -> tokens over T per spatial position
        BT, H, W, C = x.shape
        B = BT // frames
        h = x.reshape(B, frames, H * W, C)
        h = h.transpose(0, 2, 1, 3).reshape(B * H * W, frames, C)

        h = nn.RMSNorm(epsilon=self.norm_epsilon, dtype=self.dtype)(h)
        rope = RotaryEmbedding(
            dim=C // self.heads, max_seq_len=1024, dtype=self.dtype, name="temporal_rope")
        h = RoPEAttention(
            query_dim=C,
            heads=self.heads,
            dim_head=C // self.heads,
            dtype=self.dtype,
            precision=self.precision,
            use_bias=True,
            rope_emb=rope,
            name="temporal_attention",
        )(h, freqs_cis=rope(frames))
        # zero-init gate: identity at init, so inflation preserves the 2D model
        h = nn.Dense(
            features=C, dtype=self.dtype, precision=self.precision,
            kernel_init=nn.initializers.zeros, name="temporal_out")(h)

        h = h.reshape(B, H * W, frames, C).transpose(0, 2, 1, 3).reshape(BT, H, W, C)
        return x + h


class UNet3D(nn.Module):
    """Video UNet over (B, T, H, W, C): the 2D Unet forward per frame, with a
    TemporalBlock at every resolution level. Spatial param paths are
    identical to Unet, so 2D checkpoints inflate directly."""
    output_channels:int=3
    emb_features:int=64*4
    feature_depths:list=(64, 128, 256, 512)
    attention_configs:list=({"heads":8}, {"heads":8}, {"heads":8}, {"heads":8})
    num_res_blocks:int=2
    num_middle_res_blocks:int=1
    activation:Callable = jax.nn.swish
    norm_groups:int=8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    named_norms: bool = False
    attention_impl: Optional[str] = None
    temporal_heads: int = 8

    def setup(self):
        if self.norm_groups > 0:
            norm = partial(nn.GroupNorm, self.norm_groups)
            self.conv_out_norm = norm(name="GroupNorm_0") if self.named_norms else norm()
        else:
            norm = partial(nn.RMSNorm, 1e-5)
            self.conv_out_norm = norm()

    @nn.compact
    def __call__(self, x, temb, textcontext, train: bool = False):
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C)
        temb = jnp.repeat(temb, T, axis=0)
        if textcontext is not None:
            textcontext = jnp.repeat(textcontext, T, axis=0)

        temb = FourierEmbedding(features=self.emb_features)(temb)
        temb = TimeProjection(features=self.emb_features)(temb)

        feature_depths = self.feature_depths
        attention_configs = self.attention_configs

        conv_type = up_conv_type = down_conv_type = middle_conv_type = "conv"

        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        downs = [x]

        # Downscaling blocks
        for i, (dim_out, attention_config) in enumerate(zip(feature_depths, attention_configs)):
            dim_in = x.shape[-1]
            for j in range(self.num_res_blocks):
                x = ResidualBlock(
                    down_conv_type,
                    name=f"down_{i}_residual_{j}",
                    features=dim_in,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                    named_norms=self.named_norms
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:
                    x = TransformerBlock(heads=attention_config['heads'], dtype=attention_config.get('dtype', jnp.float32), attention_impl=self.attention_impl,
                                        dim_head=dim_in // attention_config['heads'],
                                        use_projection=attention_config.get("use_projection", False),
                                        use_self_and_cross=attention_config.get("use_self_and_cross", True),
                                        precision=attention_config.get("precision", self.precision),
                                        only_pure_attention=attention_config.get("only_pure_attention", True),
                                        force_fp32_for_softmax=attention_config.get("force_fp32_for_softmax", False),
                                        norm_inputs=attention_config.get("norm_inputs", True),
                                        explicitly_add_residual=attention_config.get("explicitly_add_residual", True),
                                        name=f"down_{i}_attention_{j}")(x, textcontext)
                downs.append(x)
            x = TemporalBlock(
                features=x.shape[-1],
                heads=self.temporal_heads,
                dtype=self.dtype,
                precision=self.precision,
                name=f"down_{i}_temporal"
            )(x, T)
            if i != len(feature_depths) - 1:
                x = Downsample(
                    features=dim_out,
                    scale=2,
                    activation=self.activation,
                    name=f"down_{i}_downsample",
                    dtype=self.dtype,
                    precision=self.precision
                )(x)

        # Middle Blocks
        middle_dim_out = self.feature_depths[-1]
        middle_attention = self.attention_configs[-1]
        for j in range(self.num_middle_res_blocks):
            x = ResidualBlock(
                middle_conv_type,
                name=f"middle_res1_{j}",
                features=middle_dim_out,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision,
                named_norms=self.named_norms
            )(x, temb)
            if middle_attention is not None and j == self.num_middle_res_blocks - 1:
                x = TransformerBlock(heads=middle_attention['heads'], dtype=middle_attention.get('dtype', jnp.float32), attention_impl=self.attention_impl,
                                    dim_head=middle_dim_out // middle_attention['heads'],
                                    use_linear_attention=False,
                                    use_projection=middle_attention.get("use_projection", False),
                                    use_self_and_cross=False,
                                    precision=middle_attention.get("precision", self.precision),
                                    only_pure_attention=middle_attention.get("only_pure_attention", True),
                                    force_fp32_for_softmax=middle_attention.get("force_fp32_for_softmax", False),
                                    norm_inputs=middle_attention.get("norm_inputs", True),
                                    explicitly_add_residual=middle_attention.get("explicitly_add_residual", True),
                                    name=f"middle_attention_{j}")(x, textcontext)
            x = TemporalBlock(
                features=x.shape[-1],
                heads=self.temporal_heads,
                dtype=self.dtype,
                precision=self.precision,
                name=f"middle_temporal_{j}"
            )(x, T)
            x = ResidualBlock(
                middle_conv_type,
                name=f"middle_res2_{j}",
                features=middle_dim_out,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision,
                named_norms=self.named_norms
            )(x, temb)

        # Upscaling Blocks
        for i, (dim_out, attention_config) in enumerate(zip(reversed(feature_depths), reversed(attention_configs))):
            for j in range(self.num_res_blocks):
                x = jnp.concatenate([x, downs.pop()], axis=-1)
                kernel_size = (3, 3)
                x = ResidualBlock(
                    up_conv_type,
                    name=f"up_{i}_residual_{j}",
                    features=dim_out,
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                    named_norms=self.named_norms
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:
                    x = TransformerBlock(heads=attention_config['heads'], dtype=attention_config.get('dtype', jnp.float32), attention_impl=self.attention_impl,
                                        dim_head=dim_out // attention_config['heads'],
                                        use_projection=attention_config.get("use_projection", False),
                                        use_self_and_cross=attention_config.get("use_self_and_cross", True),
                                        precision=attention_config.get("precision", self.precision),
                                        only_pure_attention=attention_config.get("only_pure_attention", True),
                                        force_fp32_for_softmax=middle_attention.get("force_fp32_for_softmax", False),
                                        norm_inputs=attention_config.get("norm_inputs", True),
                                        explicitly_add_residual=attention_config.get("explicitly_add_residual", True),
                                        name=f"up_{i}_attention_{j}")(x, textcontext)
            x = TemporalBlock(
                features=x.shape[-1],
                heads=self.temporal_heads,
                dtype=self.dtype,
                precision=self.precision,
                name=f"up_{i}_temporal"
            )(x, T)
            if i != len(feature_depths) - 1:
                x = Upsample(
                    features=feature_depths[-i],
                    scale=2,
                    activation=self.activation,
                    name=f"up_{i}_upsample",
                    dtype=self.dtype,
                    precision=self.precision
                )(x)

        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            precision=self.precision
        )(x)

        x = jnp.concatenate([x, downs.pop()], axis=-1)

        x = ResidualBlock(
            conv_type,
            name="final_residual",
            features=self.feature_depths[0],
            kernel_size=(3,3),
            strides=(1, 1),
            activation=self.activation,
            norm_groups=self.norm_groups,
            dtype=self.dtype,
            precision=self.precision,
            named_norms=self.named_norms
        )(x, temb)

        x = self.conv_out_norm(x)
        x = self.activation(x)

        noise_out = ConvLayer(
            conv_type,
            features=self.output_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        return noise_out.reshape(B, T, H, W, self.output_channels)


def inflate_unet_params(params_2d, params_3d):
    """Copy a trained 2D Unet param tree into a UNet3D init.

    Spatial module names are identical between the two models, so every 2D
    leaf lands on its 3D counterpart; the temporal blocks keep their
    (zero-init) fresh params. The result reproduces the 2D model frame by
    frame until training moves the temporal weights.
    """
    def merge(dst, src):
        out = dict(dst)
        for key, value in src.items():
            if isinstance(value, dict):
                assert key in dst, f"2D module '{key}' has no counterpart in the 3D tree"
                out[key] = merge(dst[key], value)
            else:
                out[key] = value
        return out
    return merge(params_3d, params_2d)
