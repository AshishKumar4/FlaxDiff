# simple_vit.py

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Any, Optional, Tuple
from .simple_unet import FourierEmbedding, TimeProjection, ConvLayer, kernel_init
from .attention import TransformerBlock
from flaxdiff.models.simple_unet import FourierEmbedding, TimeProjection, ConvLayer, kernel_init
import einops
from flax.typing import Dtype, PrecisionLike
from functools import partial

def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2], f"Invalid shape: {x.shape}, should be {h*w}, {patch_size**2*channels}"
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B (h p1) (w p2) C', h=h, p1=patch_size, p2=patch_size)
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
    output_channels:int=3
    patch_size: int = 16
    emb_features:int=768,
    num_layers: int = 12
    num_heads: int = 12
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGH
    use_projection: bool = False
    use_flash_attention: bool = False
    use_self_and_cross: bool = False
    force_fp32_for_softmax: bool = True
    activation:Callable = jax.nn.swish
    norm_groups:int=8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    kernel_init: Callable = partial(kernel_init, 1.0)

    def setup(self):
        if self.norm_groups > 0:
            self.norm = partial(nn.GroupNorm, self.norm_groups)
        else:
            self.norm = partial(nn.RMSNorm, 1e-5)
            
    @nn.compact
    def __call__(self, x, temb, textcontext=None):
        # Time embedding
        temb = FourierEmbedding(features=self.emb_features)(temb)
        temb = TimeProjection(features=self.emb_features)(temb)

        # Patch embedding
        x = PatchEmbedding(patch_size=self.patch_size, embedding_dim=self.emb_features, 
                           dtype=self.dtype, precision=self.precision)(x)
        num_patches = x.shape[1]
        
        context_emb = nn.DenseGeneral(features=self.emb_features, kernel_init=self.kernel_init(), 
                               dtype=self.dtype, precision=self.precision)(textcontext)
        num_text_tokens = textcontext.shape[1]
        
        # print(f'Shape of x after patch embedding: {x.shape}, numPatches: {num_patches}, temb: {temb.shape}, context_emb: {context_emb.shape}')
        
        # Add time embedding
        temb = jnp.expand_dims(temb, axis=1)
        x = jnp.concatenate([x, temb, context_emb], axis=1)
        # print(f'Shape of x after time embedding: {x.shape}')
        
        # Add positional encoding
        x = PositionalEncoding(max_len=x.shape[1], embedding_dim=self.emb_features)(x)
        
        # print(f'Shape of x after positional encoding: {x.shape}')
        
        skips = []
        # In blocks
        for i in range(self.num_layers // 2):
            x = TransformerBlock(heads=self.num_heads, dim_head=self.emb_features // self.num_heads, 
                                 dtype=self.dtype, precision=self.precision, use_projection=self.use_projection, 
                                 use_flash_attention=self.use_flash_attention, use_self_and_cross=self.use_self_and_cross, force_fp32_for_softmax=self.force_fp32_for_softmax, 
                                 only_pure_attention=False,
                                 kernel_init=self.kernel_init())(x)
            skips.append(x)
            
        # Middle block
        x = TransformerBlock(heads=self.num_heads, dim_head=self.emb_features // self.num_heads, 
                             dtype=self.dtype, precision=self.precision, use_projection=self.use_projection, 
                             use_flash_attention=self.use_flash_attention, use_self_and_cross=self.use_self_and_cross, force_fp32_for_softmax=self.force_fp32_for_softmax, 
                             only_pure_attention=False,
                             kernel_init=self.kernel_init())(x)
        
        # # Out blocks
        for i in range(self.num_layers // 2):
            skip = jnp.concatenate([x, skips.pop()], axis=-1)
            skip = nn.DenseGeneral(features=self.emb_features, kernel_init=self.kernel_init(), 
                                   dtype=self.dtype, precision=self.precision)(skip)
            x = TransformerBlock(heads=self.num_heads, dim_head=self.emb_features // self.num_heads, 
                                 dtype=self.dtype, precision=self.precision, use_projection=self.use_projection, 
                                 use_flash_attention=self.use_flash_attention, use_self_and_cross=self.use_self_and_cross, force_fp32_for_softmax=self.force_fp32_for_softmax, 
                                 only_pure_attention=False,
                                 kernel_init=self.kernel_init())(skip)
        
        # print(f'Shape of x after transformer blocks: {x.shape}')
        x = self.norm()(x)
        
        # print(f'Shape of x after norm: {x.shape}')
        
        patch_dim = self.patch_size ** 2 * self.output_channels
        x = nn.Dense(features=patch_dim, dtype=self.dtype, precision=self.precision, kernel_init=self.kernel_init())(x)
        # print(f'Shape of x after patch dense layer: {x.shape}, patch_dim: {patch_dim}')
        x = x[:, 1 + num_text_tokens:, :]
        x = unpatchify(x, channels=self.output_channels)
        # print(f'Shape of x after final dense layer: {x.shape}')
        x = nn.Dense(features=self.output_channels, dtype=self.dtype, precision=self.precision, kernel_init=self.kernel_init())(x)
        
        return x