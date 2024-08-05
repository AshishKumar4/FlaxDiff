"""
Some Code ported from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_flax.py
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Callable, Sequence, Any, Union
import einops
import functools
import math
from .common import kernel_init

class EfficientAttention(nn.Module):
    """
    Based on the pallas attention implementation.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    use_bias: bool = True
    kernel_init: Callable = lambda : kernel_init(1.0)

    def setup(self):
        inner_dim = self.dim_head * self.heads
        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        dense = functools.partial(
            nn.Dense,
            self.heads * self.dim_head,
            precision=self.precision, 
            use_bias=self.use_bias, 
            kernel_init=self.kernel_init(), 
            dtype=self.dtype
        )
        self.query = dense(name="to_q")
        self.key = dense(name="to_k")
        self.value = dense(name="to_v")
        
        self.proj_attn = nn.DenseGeneral(self.query_dim, use_bias=False, precision=self.precision, 
                                     kernel_init=self.kernel_init(), dtype=self.dtype, name="to_out_0")
        # self.attnfn = make_fast_generalized_attention(qkv_dim=inner_dim, lax_scan_unroll=16)
    
    def _reshape_tensor_to_head_dim(self, tensor):
        batch_size, _, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        return tensor
    
    def _reshape_tensor_from_head_dim(self, tensor):
        batch_size, _, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size, 1, seq_len, dim * head_size)
        return tensor

    @nn.compact
    def __call__(self, x:jax.Array, context=None):
        # print(x.shape)
        # x has shape [B, H * W, C]
        context = x if context is None else context
        
        B, H, W, C = x.shape
        x = x.reshape((B, 1, H * W, C))
        
        if len(context.shape) == 4:
            B, _H, _W, _C = context.shape
            context = context.reshape((B, 1, _H * _W, _C))
        else:
            B, SEQ, _C = context.shape
            context = context.reshape((B, 1, SEQ, _C))
        
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        
        query = self._reshape_tensor_to_head_dim(query)
        key = self._reshape_tensor_to_head_dim(key)
        value = self._reshape_tensor_to_head_dim(value)
        
        hidden_states = jax.experimental.pallas.ops.tpu.flash_attention.flash_attention(
            query, key, value, None
        )
        
        hidden_states = self._reshape_tensor_from_head_dim(hidden_states)
        
        
        # hidden_states = nn.dot_product_attention(
        #     query, key, value, dtype=self.dtype, broadcast_dropout=False, dropout_rng=None, precision=self.precision
        # )
        
        proj = self.proj_attn(hidden_states)
        
        proj = proj.reshape((B, H, W, C))
        
        return proj

class NormalAttention(nn.Module):
    """
    Simple implementation of the normal attention.
    """
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    use_bias: bool = True
    kernel_init: Callable = lambda : kernel_init(1.0)

    def setup(self):
        inner_dim = self.dim_head * self.heads
        dense = functools.partial(
            nn.DenseGeneral,
            features=[self.heads, self.dim_head], 
            axis=-1, 
            precision=self.precision, 
            use_bias=self.use_bias, 
            kernel_init=self.kernel_init(), 
            dtype=self.dtype
        )
        self.query = dense(name="to_q")
        self.key = dense(name="to_k")
        self.value = dense(name="to_v")

        self.proj_attn = nn.DenseGeneral(
            self.query_dim, 
            axis=(-2, -1), 
            precision=self.precision, 
            use_bias=self.use_bias, 
            dtype=self.dtype, 
            name="to_out_0",
            kernel_init=self.kernel_init()
            # kernel_init=jax.nn.initializers.xavier_uniform()
        )

    @nn.compact
    def __call__(self, x, context=None):
        # x has shape [B, H, W, C]
        B, H, W, C = x.shape
        x = x.reshape((B, H*W, C))
        context = x if context is None else context
        if len(context.shape) == 4:
            context = context.reshape((B, H*W, C))
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        
        hidden_states = nn.dot_product_attention(
            query, key, value, dtype=self.dtype, broadcast_dropout=False, dropout_rng=None, precision=self.precision
        )
        proj = self.proj_attn(hidden_states)
        proj = proj.reshape((B, H, W, C))
        return proj
    
class AttentionBlock(nn.Module):
    # Has self and cross attention
    query_dim: int
    heads: int = 4
    dim_head: int = 64
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGHEST
    use_bias: bool = True
    kernel_init: Callable = lambda : kernel_init(1.0)
    use_flash_attention:bool = False
    use_cross_only:bool = False
    
    def setup(self):
        if self.use_flash_attention:
            attenBlock = EfficientAttention
        else:
            attenBlock = NormalAttention
            
        self.attention1 = attenBlock(
         query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention1',
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=self.kernel_init
        )
        self.attention2 = attenBlock(
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            name=f'Attention2',
            precision=self.precision,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=self.kernel_init
        )
        
        self.ff = nn.DenseGeneral(
            features=self.query_dim,
            use_bias=self.use_bias,
            precision=self.precision,
            dtype=self.dtype,
            kernel_init=self.kernel_init(),
            name="ff"
        )
        self.norm1 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm4 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        
    @nn.compact
    def __call__(self, hidden_states, context=None):
        # self attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        if self.use_cross_only:
            hidden_states = self.attention1(hidden_states, context)
        else:
            hidden_states = self.attention1(hidden_states)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.attention2(hidden_states, context)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states

class TransformerBlock(nn.Module):
    heads: int = 4
    dim_head: int = 32
    use_linear_attention: bool = True
    dtype: Any = jnp.float32
    precision: Any = jax.lax.Precision.HIGH
    use_projection: bool = False
    use_flash_attention:bool = True
    use_self_and_cross:bool = False

    @nn.compact
    def __call__(self, x, context=None):
        inner_dim = self.heads * self.dim_head
        B, H, W, C = x.shape
        normed_x = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)(x)
        if self.use_projection == True:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=inner_dim, 
                                       use_bias=False, precision=self.precision, 
                                       kernel_init=kernel_init(1.0),
                                       dtype=self.dtype, name=f'project_in')(normed_x)
            else:
                projected_x = nn.Conv(
                    features=inner_dim, kernel_size=(1, 1),
                    kernel_init=kernel_init(1.0),
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name=f'project_in_conv',
                )(normed_x)
        else:
            projected_x = normed_x
            inner_dim = C
            
        context = projected_x if context is None else context

        if self.use_self_and_cross:
            projected_x = AttentionBlock(
                query_dim=inner_dim,
                heads=self.heads,
                dim_head=self.dim_head,
                name=f'Attention',
                precision=self.precision,
                use_bias=False,
                dtype=self.dtype,
                use_flash_attention=self.use_flash_attention,
                use_cross_only=False
            )(projected_x, context)
        elif self.use_flash_attention == True:
            projected_x = EfficientAttention(
                query_dim=inner_dim,
                heads=self.heads,
                dim_head=self.dim_head,
                name=f'Attention',
                precision=self.precision,
                use_bias=False,
                dtype=self.dtype,
            )(projected_x, context)
        else:
            projected_x = NormalAttention(
                query_dim=inner_dim,
                heads=self.heads,
                dim_head=self.dim_head,
                name=f'Attention',
                precision=self.precision,
                use_bias=False,
            )(projected_x, context)
        

        if self.use_projection == True:
            if self.use_linear_attention:
                projected_x = nn.Dense(features=C, precision=self.precision, 
                                       dtype=self.dtype, use_bias=False, 
                                       kernel_init=kernel_init(1.0),
                                       name=f'project_out')(projected_x)
            else:
                projected_x = nn.Conv(
                    features=C, kernel_size=(1, 1),
                    kernel_init=kernel_init(1.0),
                    strides=(1, 1), padding='VALID', use_bias=False, dtype=self.dtype,
                    precision=self.precision, name=f'project_out_conv',
                )(projected_x)

        out = x + projected_x
        return out

class FlaxGEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype, precision=jax.lax.Precision.DEFAULT)

    def __call__(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=3)
        return hidden_linear * nn.gelu(hidden_gelu)
    
class FlaxFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
    [`FeedForward`] class, with the following simplifications:
    - The activation function is currently hardcoded to a gated linear unit from:
    https://arxiv.org/abs/2002.05202
    - `dim_out` is equal to `dim`.
    - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = FlaxGEGLU(self.dim, self.dtype)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype, precision=jax.lax.Precision.DEFAULT)

    def __call__(self, hidden_states):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states

class BasicTransformerBlock(nn.Module):
    query_dim: int
    heads: int
    dim_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    precision: Any = jax.lax.Precision.DEFAULT

    def setup(self):
        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = NormalAttention(
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dtype=self.dtype,
            precision=self.precision,
        )
        # cross attention
        self.attn2 = NormalAttention(
            query_dim=self.query_dim,
            heads=self.heads,
            dim_head=self.dim_head,
            dtype=self.dtype,
            precision=self.precision,
        )
        self.ff = FlaxFeedForward(dim=self.query_dim, dropout=self.dropout, dtype=self.dtype)
        self.norm1 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.RMSNorm(epsilon=1e-5, dtype=self.dtype)

    def __call__(self, hidden_states, context, deterministic=True):
        # self attention
        residual = hidden_states
        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm1(hidden_states), context)
        else:
            hidden_states = self.attn1(self.norm1(hidden_states))
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states))
        hidden_states = hidden_states + residual

        return hidden_states