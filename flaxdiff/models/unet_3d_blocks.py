from typing import Tuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from diffusers.models.attention_flax import (
    FlaxBasicTransformerBlock,
    FlaxTransformer2DModel,
)

from diffusers.models.resnet_flax import (
    FlaxResnetBlock2D,
    FlaxUpsample2D,
    FlaxDownsample2D,
)

from diffusers.models.unets.unet_2d_blocks_flax import (
    FlaxCrossAttnDownBlock2D,
    FlaxDownBlock2D,
    FlaxUNetMidBlock2DCrossAttn,
    FlaxUpBlock2D,
    FlaxCrossAttnUpBlock2D,
)

class FlaxTransformerTemporalModel(nn.Module):
    """
    Transformer for temporal attention in 3D UNet.
    """
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    
    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
        # Use existing FlaxBasicTransformerBlock from diffusers
        self.transformer_blocks = [
            FlaxBasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
            )
            for _ in range(self.depth)
        ]
        
        self.proj_out = nn.Dense(inner_dim, dtype=self.dtype)

        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, hidden_states: jnp.ndarray, context: jnp.ndarray, num_frames: int, deterministic=True):
        # Save original shape for later reshaping
        batch_depth, height, width, channels = hidden_states.shape
        batch = batch_depth // num_frames
        
        # Reshape to (batch, depth, height, width, channels)
        hidden_states = hidden_states.reshape(batch, num_frames, height, width, channels)
        residual = hidden_states
        
        # Apply normalization
        hidden_states = self.norm(hidden_states)
        
        # Reshape for temporal attention: (batch, depth, height, width, channels) -> 
        # (batch*height*width, depth, channels)
        hidden_states = hidden_states.transpose(0, 2, 3, 1, 4)
        hidden_states = hidden_states.reshape(batch * height * width, num_frames, channels)
        
        # Project input
        hidden_states = self.proj_in(hidden_states)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context, deterministic=deterministic)
        
        # Project output
        hidden_states = self.proj_out(hidden_states)
        
        # Reshape back to original shape
        hidden_states = hidden_states.reshape(batch, height, width, num_frames, channels)
        hidden_states = hidden_states.transpose(0, 3, 1, 2, 4)
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        # Reshape back to (batch*depth, height, width, channels)
        hidden_states = hidden_states.reshape(batch_depth, height, width, channels)
        
        return hidden_states

class TemporalConvLayer(nn.Module):
    in_channels: int
    out_channels: Optional[int] = None
    dropout: float = 0.0
    norm_num_groups: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, num_frames: int, deterministic=True) -> jnp.ndarray:
        """
        Args:
          x: shape (B*F, H, W, C)
          num_frames: number of frames F per batch element

        Returns:
          A jnp.ndarray of shape (B*F, H, W, C)
        """
        out_channels = self.out_channels or self.in_channels
        bf, h, w, c = x.shape
        b = bf // num_frames

        # Reshape to [B, F, H, W, C], interpret F as "depth" for 3D conv
        x = x.reshape(b, num_frames, h, w, c)
        identity = x

        # conv1: in_channels -> out_channels
        x = nn.GroupNorm(num_groups=self.norm_num_groups)(x)
        x = nn.silu(x)
        x = nn.Conv(features=out_channels, kernel_size=(3, 1, 1),
                    dtype=self.dtype,
                    padding=((1,1), (0,0), (0,0)))(x)

        # conv2: out_channels -> in_channels
        x = nn.GroupNorm(num_groups=self.norm_num_groups)(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Conv(features=self.in_channels, kernel_size=(3, 1, 1),
                    dtype=self.dtype,
                    padding=((1,1), (0,0), (0,0)))(x)

        # conv3: in_channels -> in_channels
        x = nn.GroupNorm(num_groups=self.norm_num_groups)(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Conv(features=self.in_channels, kernel_size=(3, 1, 1),
                    dtype=self.dtype,
                    padding=((1,1), (0,0), (0,0)))(x)

        # conv4 (zero-init): in_channels -> in_channels
        x = nn.GroupNorm(num_groups=self.norm_num_groups)(x)
        x = nn.silu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.Conv(
            features=self.in_channels,
            kernel_size=(3, 1, 1),
            padding=((1,1), (0,0), (0,0)),
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
        )(x)

        # Residual connection and reshape back to (B*F, H, W, C)
        x = identity + x
        x = x.reshape(bf, h, w, c)
        return x


class FlaxCrossAttnDownBlock3D(FlaxCrossAttnDownBlock2D):
    """
    Cross attention 3D downsampling block.
    """

    def setup(self):
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []

        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)
            temp_conv = TemporalConvLayer(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                dtype=self.dtype,
            )
            temp_convs.append(temp_conv)
            attn_block = FlaxTransformer2DModel(
                in_channels=self.out_channels,
                n_heads=self.num_attention_heads,
                d_head=self.out_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=self.only_cross_attention,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
                dtype=self.dtype,
            )
            attentions.append(attn_block)
            temp_attn_block = FlaxTransformerTemporalModel(
                in_channels=self.out_channels,
                n_heads=self.num_attention_heads,
                d_head=self.out_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block,
                dropout=self.dropout,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
                dtype=self.dtype,
            )
            temp_attentions.append(temp_attn_block)
        
        self.temp_convs = temp_convs
        self.temp_attentions = temp_attentions
        self.resnets = resnets
        self.attentions = attentions

        if self.add_downsample:
            # self.downsamplers_0 = FlaxDownsample3D(self.out_channels, dtype=self.dtype)
            self.downsamplers_0 = FlaxDownsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, temb, encoder_hidden_states, num_frames, deterministic=True):
        output_states = ()

        for resnet, temp_conv, attn, temp_attn in zip(self.resnets, self.temp_convs, self.attentions, self.temp_attentions):
            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames, deterministic=deterministic)
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
            hidden_states = temp_attn(hidden_states, None, num_frames=num_frames, deterministic=deterministic)
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class FlaxDownBlock3D(FlaxDownBlock2D):
    """
    Basic downsampling block without attention.
    """
    def setup(self):
        resnets = []
        temp_convs = []

        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)
            temp_conv = TemporalConvLayer(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                dtype=self.dtype,
            )
            temp_convs.append(temp_conv)
        self.temp_convs = temp_convs
        self.resnets = resnets

        if self.add_downsample:
            self.downsamplers_0 = FlaxDownsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, temb, num_frames, deterministic=True):
        output_states = ()

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames, deterministic=deterministic)
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class FlaxCrossAttnUpBlock3D(FlaxCrossAttnUpBlock2D):
    """
    Cross attention 3D upsampling block.
    """

    def setup(self):
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []

        for i in range(self.num_layers):
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)
            temp_conv = TemporalConvLayer(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                dtype=self.dtype,
            )
            temp_convs.append(temp_conv)
            attn_block = FlaxTransformer2DModel(
                in_channels=self.out_channels,
                n_heads=self.num_attention_heads,
                d_head=self.out_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block,
                use_linear_projection=self.use_linear_projection,
                only_cross_attention=self.only_cross_attention,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
                dtype=self.dtype,
            )
            attentions.append(attn_block)
            temp_attn_block = FlaxTransformerTemporalModel(
                in_channels=self.out_channels,
                n_heads=self.num_attention_heads,
                d_head=self.out_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block,
                dropout=self.dropout,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
                dtype=self.dtype,
            )
            temp_attentions.append(temp_attn_block)

        self.resnets = resnets
        self.attentions = attentions
        self.temp_convs = temp_convs
        self.temp_attentions = temp_attentions

        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, num_frames, deterministic=True):
        for resnet, temp_conv, attn, temp_attn in zip(self.resnets, self.temp_convs, self.attentions, self.temp_attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)

            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames, deterministic=deterministic)
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
            hidden_states = temp_attn(hidden_states, None, num_frames=num_frames, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)

        return hidden_states


class FlaxUpBlock3D(FlaxUpBlock2D):
    """
    Basic upsampling block without attention.
    """
    def setup(self):
        resnets = []
        temp_convs = []

        for i in range(self.num_layers):
            res_skip_channels = self.in_channels if (i == self.num_layers - 1) else self.out_channels
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=self.out_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)
            temp_conv = TemporalConvLayer(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                dropout=self.dropout,
                dtype=self.dtype,
            )
            temp_convs.append(temp_conv)

        self.resnets = resnets
        self.temp_convs = temp_convs

        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)

    def __call__(self, hidden_states, res_hidden_states_tuple, temb, num_frames, deterministic=True):
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)

            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames, deterministic=deterministic)

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)

        return hidden_states


class FlaxUNetMidBlock3DCrossAttn(FlaxUNetMidBlock2DCrossAttn):
    """
    Middle block with cross-attention for 3D UNet.
    """
    def setup(self):
        # there is always at least one resnet
        resnets = [
            FlaxResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
        ]
        temp_convs = [
            TemporalConvLayer(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout=self.dropout,
                dtype=self.dtype,
            )
        ]

        attentions = []
        temp_attentions = []

        for _ in range(self.num_layers):
            attn_block = FlaxTransformer2DModel(
                in_channels=self.in_channels,
                n_heads=self.num_attention_heads,
                d_head=self.in_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block,
                use_linear_projection=self.use_linear_projection,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
                dtype=self.dtype,
            )
            attentions.append(attn_block)
            
            temp_block = FlaxTransformerTemporalModel(
                in_channels=self.in_channels,
                n_heads=self.num_attention_heads,
                d_head=self.in_channels // self.num_attention_heads,
                depth=self.transformer_layers_per_block,
                dropout=self.dropout,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
                dtype=self.dtype,
            )
            temp_attentions.append(temp_block)

            res_block = FlaxResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)
            temp_conv = TemporalConvLayer(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout=self.dropout,
                dtype=self.dtype,
            )
            temp_convs.append(temp_conv)
            
        self.temp_convs = temp_convs
        self.temp_attentions = temp_attentions
        self.resnets = resnets
        self.attentions = attentions
        
    def __call__(self, hidden_states, temb, encoder_hidden_states, num_frames, deterministic=True):
        hidden_states = self.resnets[0](hidden_states, temb, deterministic=deterministic)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames, deterministic=deterministic)
            
        for attn, temp_attn, resnet, temp_conv in zip(
            self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
            hidden_states = temp_attn(hidden_states, None, num_frames=num_frames, deterministic=deterministic)
            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames, deterministic=deterministic)

        return hidden_states
