# simple_vit.py

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Any
from .simply_unet import FourierEmbedding, TimeProjection, ConvLayer, kernel_init
from .attention import TransformerBlock

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

class TransformerEncoder(nn.Module):
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGH

    @nn.compact
    def __call__(self, x, training=True):
        for _ in range(self.num_layers):
            x = TransformerBlock(
                heads=self.num_heads,
                dim_head=x.shape[-1] // self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision
            )(x)
        return x

class VisionTransformer(nn.Module):
    patch_size: int = 16
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    emb_features: int = 256
    dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGH

    @nn.compact
    def __call__(self, x, temb, textcontext=None):
        # Time embedding
        temb = FourierEmbedding(features=self.emb_features)(temb)
        temb = TimeProjection(features=self.emb_features)(temb)

        # Patch embedding
        x = PatchEmbedding(patch_size=self.patch_size, embedding_dim=self.embedding_dim, 
                           dtype=self.dtype, precision=self.precision)(x)
        
        # Add positional encoding
        x = PositionalEncoding(max_len=x.shape[1], embedding_dim=self.embedding_dim)(x)

        # Add time embedding
        temb = jnp.expand_dims(temb, axis=1)
        x = jnp.concatenate([x, temb], axis=1)

        # Add text context
        if textcontext is not None:
            x = jnp.concatenate([x, textcontext], axis=1)

        # Transformer encoder
        x = TransformerEncoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            precision=self.precision
        )(x)

        # Extract the image tokens (exclude time and text embeddings)
        num_patches = (x.shape[1] - 1 - (0 if textcontext is None else textcontext.shape[1]))
        x = x[:, :num_patches, :]

        # Reshape to image dimensions
        batch, _, _ = x.shape
        height = width = int((num_patches) ** 0.5)
        x = jnp.reshape(x, (batch, height, width, self.embedding_dim))

        # Final convolution to get the desired output channels
        x = ConvLayer(
            conv_type="conv",
            features=3,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=kernel_init(0.0),
            dtype=self.dtype,
            precision=self.precision
        )(x)

        return x