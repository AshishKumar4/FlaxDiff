import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from typing import Dict, Callable, Sequence, Any, Union, Optional
import einops
from .common import kernel_init, ConvLayer, Downsample, Upsample, FourierEmbedding, TimeProjection, ResidualBlock
from .attention import TransformerBlock
from functools import partial

class Unet(nn.Module):
    output_channels:int=3
    emb_features:int=64*4,
    feature_depths:list=[64, 128, 256, 512],
    attention_configs:list=[{"heads":8}, {"heads":8}, {"heads":8}, {"heads":8}],
    num_res_blocks:int=2,
    num_middle_res_blocks:int=1,
    activation:Callable = jax.nn.swish
    norm_groups:int=8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    named_norms: bool = False # This is for backward compatibility reasons; older checkpoints have named norms
    kernel_init: Callable = partial(kernel_init, dtype=jnp.float32)

    def setup(self):
        if self.norm_groups > 0:
            norm = partial(nn.GroupNorm, self.norm_groups)
            self.conv_out_norm = norm(name="GroupNorm_0") if self.named_norms else norm()
        else:
            norm = partial(nn.RMSNorm, 1e-5)
            self.conv_out_norm = norm()
        
    @nn.compact
    def __call__(self, x, temb, textcontext):
        # print("embedding features", self.emb_features)
        temb = FourierEmbedding(features=self.emb_features)(temb)
        temb = TimeProjection(features=self.emb_features)(temb)
        
        _, TS, TC = textcontext.shape
        
        # print("time embedding", temb.shape)
        feature_depths = self.feature_depths
        attention_configs = self.attention_configs

        conv_type = up_conv_type = down_conv_type = middle_conv_type = "conv"
        # middle_conv_type = "separable"
        
        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=self.kernel_init(1.0),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        downs = [x]

        # Downscaling blocks
        for i, (dim_out, attention_config) in enumerate(zip(feature_depths, attention_configs)):
            dim_in = x.shape[-1]
            # dim_in = dim_out
            for j in range(self.num_res_blocks):
                x = ResidualBlock(
                    down_conv_type,
                    name=f"down_{i}_residual_{j}",
                    features=dim_in,
                    kernel_init=self.kernel_init(1.0),
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                    named_norms=self.named_norms
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:   # Apply attention only on the last block
                    x = TransformerBlock(heads=attention_config['heads'], dtype=attention_config.get('dtype', jnp.float32),
                                        dim_head=dim_in // attention_config['heads'],
                                        use_flash_attention=attention_config.get("flash_attention", False),
                                        use_projection=attention_config.get("use_projection", False),
                                        use_self_and_cross=attention_config.get("use_self_and_cross", True),
                                        precision=attention_config.get("precision", self.precision),
                                        only_pure_attention=attention_config.get("only_pure_attention", True),
                                        force_fp32_for_softmax=attention_config.get("force_fp32_for_softmax", False),
                                        kernel_init=self.kernel_init(1.0),
                                        name=f"down_{i}_attention_{j}")(x, textcontext)
                # print("down residual for feature level", i, "is of shape", x.shape, "features", dim_in)
                downs.append(x)
            if i != len(feature_depths) - 1:
                # print("Downsample", i, x.shape)
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
                kernel_init=self.kernel_init(1.0),
                kernel_size=(3, 3),
                strides=(1, 1),
                activation=self.activation,
                norm_groups=self.norm_groups,
                dtype=self.dtype,
                precision=self.precision,
                named_norms=self.named_norms
            )(x, temb)
            if middle_attention is not None and j == self.num_middle_res_blocks - 1:   # Apply attention only on the last block
                x = TransformerBlock(heads=middle_attention['heads'], dtype=middle_attention.get('dtype', jnp.float32), 
                                    dim_head=middle_dim_out // middle_attention['heads'],
                                    use_flash_attention=middle_attention.get("flash_attention", False),
                                    use_linear_attention=False,
                                    use_projection=middle_attention.get("use_projection", False),
                                    use_self_and_cross=False,
                                    precision=middle_attention.get("precision", self.precision),
                                    only_pure_attention=middle_attention.get("only_pure_attention", True),
                                    force_fp32_for_softmax=middle_attention.get("force_fp32_for_softmax", False),
                                    kernel_init=self.kernel_init(1.0),
                                    name=f"middle_attention_{j}")(x, textcontext)
            x = ResidualBlock(
                middle_conv_type,
                name=f"middle_res2_{j}",
                features=middle_dim_out,
                kernel_init=self.kernel_init(1.0),
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
            # print("Upscaling", i, "features", dim_out)
            for j in range(self.num_res_blocks):
                x = jnp.concatenate([x, downs.pop()], axis=-1)
                # print("concat==> ", i, "concat", x.shape)
                # kernel_size = (1 + 2 * (j + 1), 1 + 2 * (j + 1))
                kernel_size = (3, 3)
                x = ResidualBlock(
                    up_conv_type,# if j == 0 else "separable",
                    name=f"up_{i}_residual_{j}",
                    features=dim_out,
                    kernel_init=self.kernel_init(1.0),
                    kernel_size=kernel_size,
                    strides=(1, 1),
                    activation=self.activation,
                    norm_groups=self.norm_groups,
                    dtype=self.dtype,
                    precision=self.precision,
                    named_norms=self.named_norms
                )(x, temb)
                if attention_config is not None and j == self.num_res_blocks - 1:   # Apply attention only on the last block
                    x = TransformerBlock(heads=attention_config['heads'], dtype=attention_config.get('dtype', jnp.float32), 
                                        dim_head=dim_out // attention_config['heads'],
                                        use_flash_attention=attention_config.get("flash_attention", False),
                                        use_projection=attention_config.get("use_projection", False),
                                        use_self_and_cross=attention_config.get("use_self_and_cross", True),
                                        precision=attention_config.get("precision", self.precision),
                                        only_pure_attention=attention_config.get("only_pure_attention", True),
                                        force_fp32_for_softmax=middle_attention.get("force_fp32_for_softmax", False),
                                        kernel_init=self.kernel_init(1.0),
                                        name=f"up_{i}_attention_{j}")(x, textcontext)
            # print("Upscaling ", i, x.shape)
            if i != len(feature_depths) - 1:
                x = Upsample(
                    features=feature_depths[-i],
                    scale=2,
                    activation=self.activation,
                    name=f"up_{i}_upsample",
                    dtype=self.dtype,
                    precision=self.precision
                )(x)

        # x = self.last_up_norm(x)
        x = ConvLayer(
            conv_type,
            features=self.feature_depths[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=self.kernel_init(1.0),
            dtype=self.dtype,
            precision=self.precision
        )(x)
    
        x = jnp.concatenate([x, downs.pop()], axis=-1)

        x = ResidualBlock(
            conv_type,
            name="final_residual",
            features=self.feature_depths[0],
            kernel_init=self.kernel_init(1.0),
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
            # activation=jax.nn.mish
            kernel_init=self.kernel_init(0.0),
            dtype=self.dtype,
            precision=self.precision
        )(x)
        return noise_out#, attentions