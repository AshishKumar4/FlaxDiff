from typing import Dict, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from diffusers.configuration_utils import ConfigMixin, flax_register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from diffusers.models.modeling_flax_utils import FlaxModelMixin

from .unet_3d_blocks import (
    FlaxCrossAttnDownBlock3D,
    FlaxCrossAttnUpBlock3D,
    FlaxDownBlock3D,
    FlaxUNetMidBlock3DCrossAttn,
    FlaxUpBlock3D,
)


@flax_register_to_config
class FlaxUNet3DConditionModel(nn.Module, FlaxModelMixin, ConfigMixin):
    r"""
    A conditional 3D UNet model for video diffusion.

    Parameters:
        sample_size (`int` or `Tuple[int, int, int]`, *optional*, defaults to (16, 32, 32)):
            The spatial and temporal size of the input sample. Can be provided as a single integer for square spatial size and fixed temporal size.
        in_channels (`int`, *optional*, defaults to 4):
            The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4):
            The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to ("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D")):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D")):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to (320, 640, 1280, 1280)):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        attention_head_dim (`int`, *optional*, defaults to 8):
            The dimension of the attention heads.
        cross_attention_dim (`int`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        dropout (`float`, *optional*, defaults to 0):
            Dropout probability for down, up and bottleneck blocks.
        use_linear_projection (`bool`, *optional*, defaults to False):
            Whether to use linear projection in attention blocks.
        dtype (`jnp.dtype`, *optional*, defaults to jnp.float32):
            The dtype of the model weights.
        flip_sin_to_cos (`bool`, *optional*, defaults to True):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): 
            The frequency shift to apply to the time embedding.
        use_memory_efficient_attention (`bool`, *optional*, defaults to False):
            Whether to use memory-efficient attention.
        split_head_dim (`bool`, *optional*, defaults to False):
            Whether to split the head dimension into a new axis for the self-attention computation.
    """

    sample_size: Union[int, Tuple[int, int, int]] = (16, 32, 32)
    in_channels: int = 4
    out_channels: int = 4
    down_block_types: Tuple[str, ...] = (
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "DownBlock3D",
    )
    up_block_types: Tuple[str, ...] = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D")
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    layers_per_block: int = 2
    attention_head_dim: Union[int, Tuple[int, ...]] = 8
    num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None
    cross_attention_dim: int = 1280
    dropout: float = 0.0
    use_linear_projection: bool = False
    dtype: jnp.dtype = jnp.float32
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1
    addition_embed_type: Optional[str] = None
    addition_time_embed_dim: Optional[int] = None

    def init_weights(self, rng: jax.Array) -> FrozenDict:
        # init input tensors
        if isinstance(self.sample_size, int):
            sample_size = (self.sample_size, self.sample_size, self.sample_size)
        else:
            sample_size = self.sample_size
            
        # Shape: [batch, frames, height, width, channels]
        sample_shape = (1, sample_size[0], sample_size[1], sample_size[2], self.in_channels)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        added_cond_kwargs = None
        if self.addition_embed_type == "text_time":
            # For text-time conditioning for video diffusion
            text_embeds_dim = self.cross_attention_dim
            time_ids_dims = 6  # Default value for video models
            added_cond_kwargs = {
                "text_embeds": jnp.zeros((1, text_embeds_dim), dtype=jnp.float32),
                "time_ids": jnp.zeros((1, time_ids_dims), dtype=jnp.float32),
            }
            
        return self.init(rngs, sample, timesteps, encoder_hidden_states, added_cond_kwargs)["params"]

    def setup(self) -> None:
        block_out_channels = self.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        if self.num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue. "
                "Use `attention_head_dim` instead."
            )

        # Default behavior: if num_attention_heads is not set, use attention_head_dim
        num_attention_heads = self.num_attention_heads or self.attention_head_dim

        # input
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding=((1, 1), (1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # time
        self.time_proj = FlaxTimesteps(
            block_out_channels[0], flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
        )
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)

        # Handle attention head configurations
        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(self.down_block_types)

        # transformer layers per block
        transformer_layers_per_block = self.transformer_layers_per_block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(self.down_block_types)

        # addition embed types
        if self.addition_embed_type == "text_time":
            if self.addition_time_embed_dim is None:
                raise ValueError(
                    f"addition_embed_type {self.addition_embed_type} requires `addition_time_embed_dim` to not be None"
                )
            self.add_time_proj = FlaxTimesteps(self.addition_time_embed_dim, self.flip_sin_to_cos, self.freq_shift)
            self.add_embedding = FlaxTimestepEmbedding(time_embed_dim, dtype=self.dtype)
        else:
            self.add_embedding = None

        # down blocks
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock3D":
                down_block = FlaxCrossAttnDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    num_attention_heads=num_attention_heads[i],
                    add_downsample=not is_final_block,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=False,  # We don't use only cross attention in 3D UNet
                    use_memory_efficient_attention=self.use_memory_efficient_attention,
                    split_head_dim=self.split_head_dim,
                    dtype=self.dtype,
                )
            elif down_block_type == "DownBlock3D":
                down_block = FlaxDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    dropout=self.dropout,
                    num_layers=self.layers_per_block,
                    add_downsample=not is_final_block,
                    dtype=self.dtype,
                )
            else:
                raise ValueError(f"Unknown down block type: {down_block_type}")

            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # mid block
        self.mid_block = FlaxUNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            dropout=self.dropout,
            num_attention_heads=num_attention_heads[-1],
            transformer_layers_per_block=transformer_layers_per_block[-1],
            use_linear_projection=self.use_linear_projection,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
            split_head_dim=self.split_head_dim,
            dtype=self.dtype,
        )

        # up blocks
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        output_channel = reversed_block_out_channels[0]
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
        
        for i, up_block_type in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "CrossAttnUpBlock3D":
                up_block = FlaxCrossAttnUpBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    num_attention_heads=reversed_num_attention_heads[i],
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    use_linear_projection=self.use_linear_projection,
                    only_cross_attention=False,  # We don't use only cross attention in 3D UNet
                    use_memory_efficient_attention=self.use_memory_efficient_attention,
                    split_head_dim=self.split_head_dim,
                    dtype=self.dtype,
                )
            elif up_block_type == "UpBlock3D":
                up_block = FlaxUpBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    num_layers=self.layers_per_block + 1,
                    add_upsample=not is_final_block,
                    dropout=self.dropout,
                    dtype=self.dtype,
                )
            else:
                raise ValueError(f"Unknown up block type: {up_block_type}")

            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = up_blocks

        # out
        self.conv_norm_out = nn.GroupNorm(num_groups=32, epsilon=1e-5)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding=((1, 1), (1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(
        self,
        sample: jnp.ndarray,
        timesteps: Union[jnp.ndarray, float, int],
        encoder_hidden_states: jnp.ndarray,
        frame_encoder_hidden_states: Optional[jnp.ndarray] = None,
        added_cond_kwargs: Optional[Union[Dict, FrozenDict]] = None,
        down_block_additional_residuals: Optional[Tuple[jnp.ndarray, ...]] = None,
        mid_block_additional_residual: Optional[jnp.ndarray] = None,
        return_dict: bool = True,
        train: bool = False,
    ) -> Union[jnp.ndarray]:
        r"""
        Args:
            sample (`jnp.ndarray`): (batch, frames, height, width, channels) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (batch_size, sequence_length, hidden_size) encoder hidden states
            frame_encoder_hidden_states (`jnp.ndarray`, *optional*): 
                (batch_size, frames, sequence_length, hidden_size) per-frame encoder hidden states
            added_cond_kwargs: (`dict`, *optional*):
                Additional embeddings to add to the time embeddings
            down_block_additional_residuals: (`tuple` of `jnp.ndarray`, *optional*):
                Additional residual connections for down blocks
            mid_block_additional_residual: (`jnp.ndarray`, *optional*):
                Additional residual connection for mid block
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a dict or tuple
            train (`bool`, *optional*, defaults to `False`):
                Training mode flag for dropout
        """
        # Extract the number of frames from the input
        batch, num_frames, height, width, channels = sample.shape

        # 1. Time embedding
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)
        
        # Repeat time embedding for each frame
        t_emb = jnp.repeat(t_emb, repeats=num_frames, axis=0)
        

        # additional embeddings
        if self.add_embedding is not None and added_cond_kwargs is not None:
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    "text_embeds must be provided for text_time addition_embed_type"
                )
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    "time_ids must be provided for text_time addition_embed_type"
                )
                
            text_embeds = added_cond_kwargs["text_embeds"]
            time_ids = added_cond_kwargs["time_ids"]
            
            # Compute time embeds
            time_embeds = self.add_time_proj(jnp.ravel(time_ids))
            time_embeds = jnp.reshape(time_embeds, (text_embeds.shape[0], -1))
            
            # Concatenate text and time embeds
            add_embeds = jnp.concatenate([text_embeds, time_embeds], axis=-1)
            
            # Project to time embedding dimension
            aug_emb = self.add_embedding(add_embeds)
            t_emb = t_emb + aug_emb

        # 2. Pre-process input - reshape from [B, F, H, W, C] to [B*F, H, W, C] for 2D operations
        sample = sample.reshape(batch * num_frames, height, width, channels)
        sample = self.conv_in(sample)

        # Process encoder hidden states - repeat for each frame and combine with frame-specific conditioning if provided
        if encoder_hidden_states is not None:
            # Repeat video-wide conditioning for each frame: (B, S, X) -> (B*F, S, X)
            encoder_hidden_states_expanded = jnp.repeat(
                encoder_hidden_states, repeats=num_frames, axis=0
            )
            
            # If we have frame-specific conditioning, reshape and concatenate with video conditioning
            if frame_encoder_hidden_states is not None:
                # Reshape from (B, F, S, X) to (B*F, S, X)
                frame_encoder_hidden_states = frame_encoder_hidden_states.reshape(
                    batch * num_frames, *frame_encoder_hidden_states.shape[2:]
                )
                
                # Concatenate along the sequence dimension
                encoder_hidden_states_combined = jnp.concatenate(
                    [encoder_hidden_states_expanded, frame_encoder_hidden_states],
                    axis=1
                )
            else:
                encoder_hidden_states_combined = encoder_hidden_states_expanded
        else:
            encoder_hidden_states_combined = None

        # 3. Down blocks
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if isinstance(down_block, FlaxCrossAttnDownBlock3D):
                sample, res_samples = down_block(
                    sample, 
                    t_emb, 
                    encoder_hidden_states_combined, 
                    num_frames=num_frames, 
                    deterministic=not train
                )
            else:
                sample, res_samples = down_block(
                    sample, 
                    t_emb, 
                    num_frames=num_frames, 
                    deterministic=not train
                )
            down_block_res_samples += res_samples

        # Add additional residuals if provided
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample += down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. Mid block
        sample = self.mid_block(
            sample, 
            t_emb, 
            encoder_hidden_states_combined, 
            num_frames=num_frames, 
            deterministic=not train
        )

        # Add mid block residual if provided
        if mid_block_additional_residual is not None:
            sample += mid_block_additional_residual

        # 5. Up blocks
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-(self.layers_per_block + 1) :]
            down_block_res_samples = down_block_res_samples[: -(self.layers_per_block + 1)]
            if isinstance(up_block, FlaxCrossAttnUpBlock3D):
                sample = up_block(
                    sample,
                    res_hidden_states_tuple=res_samples,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states_combined,
                    num_frames=num_frames,
                    deterministic=not train,
                )
            else:
                sample = up_block(
                    sample, 
                    res_hidden_states_tuple=res_samples, 
                    temb=t_emb, 
                    num_frames=num_frames, 
                    deterministic=not train
                )

        # 6. Post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        
        # Reshape back to [B, F, H, W, C]
        sample = sample.reshape(batch, num_frames, height, width, self.out_channels)
        return sample