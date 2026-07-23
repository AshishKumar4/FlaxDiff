"""
Video DiT with factorized spatial-temporal attention, built from the shared
DiT machinery. This replaces the old diffusers-derived UNet3D, which was
never wired into the registry and whose upstream flax blocks no longer exist.

Each layer is a spatial ModulatedBlock over the patch tokens of every frame
followed by a temporal ModulatedBlock over the frame axis of every patch
position - the standard factorized design, so compute stays linear in T for
the spatial half and linear in S for the temporal half.
"""

import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from flax.typing import Dtype, PrecisionLike

from .dit_common import (
    PatchSequenceEmbed, ConditioningEmbed, PatchSequenceOutput,
    ModulatedBlock, neutralized_rope_freqs,
)
from .vit_common import RotaryEmbedding


class VideoDiT(nn.Module):
    """Factorized spatial-temporal DiT over (B, T, H, W, C) inputs."""
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12  # Each layer is one spatial + one temporal block
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False
    qk_norm: bool = False
    attention_impl: Optional[str] = None
    use_hilbert: bool = False
    use_zigzag: bool = False

    @property
    def scan_order(self):
        assert not (self.use_hilbert and self.use_zigzag), \
            "use_hilbert and use_zigzag are mutually exclusive"
        return 'hilbert' if self.use_hilbert else 'zigzag' if self.use_zigzag else 'raster'

    def setup(self):
        self.embed = PatchSequenceEmbed(
            patch_size=self.patch_size,
            emb_features=self.emb_features,
            scan_order=self.scan_order,
            dtype=self.dtype,
            precision=self.precision,
        )
        self.conditioning = ConditioningEmbed(
            emb_features=self.emb_features,
            mlp_ratio=self.mlp_ratio,
            dtype=self.dtype,
            precision=self.precision,
        )
        dim_head = self.emb_features // self.num_heads
        self.spatial_rope = RotaryEmbedding(
            dim=dim_head, max_seq_len=4096, dtype=self.dtype, name="spatial_rope")
        self.temporal_rope = RotaryEmbedding(
            dim=dim_head, max_seq_len=1024, dtype=self.dtype, name="temporal_rope")

        def block(name):
            return ModulatedBlock(
                features=self.emb_features,
                num_heads=self.num_heads,
                mixer='attention',
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                norm_epsilon=self.norm_epsilon,
                qk_norm=self.qk_norm,
                attention_impl=self.attention_impl,
                name=name,
            )

        self.spatial_blocks = [block(f"spatial_block_{i}") for i in range(self.num_layers)]
        self.temporal_blocks = [block(f"temporal_block_{i}") for i in range(self.num_layers)]

        self.output = PatchSequenceOutput(
            patch_size=self.patch_size,
            output_channels=self.output_channels,
            learn_sigma=self.learn_sigma,
            norm_epsilon=self.norm_epsilon,
            dtype=self.dtype,
            precision=self.precision,
        )

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train: bool = False):
        B, T, H, W, C = x.shape

        # Per-frame patchify; the permutation is identical for every frame
        frames = x.reshape(B * T, H, W, C)
        tokens, inv_idx = self.embed(frames)  # [B*T, S, F]
        S = tokens.shape[1]

        cond_emb = self.conditioning(temb, textcontext)  # [B, F]
        cond_spatial = jnp.repeat(cond_emb, T, axis=0)   # [B*T, F]
        cond_temporal = jnp.repeat(cond_emb, S, axis=0)  # [B*S, F]

        freqs_spatial = neutralized_rope_freqs(self.spatial_rope, S, self.scan_order)
        # Time is a genuine 1D axis, RoPE applies directly
        freqs_temporal = self.temporal_rope(seq_len=T)

        for spatial, temporal in zip(self.spatial_blocks, self.temporal_blocks):
            tokens = spatial(tokens, conditioning=cond_spatial, freqs_cis=freqs_spatial, train=train)
            # [B*T, S, F] -> [B*S, T, F]
            tokens = tokens.reshape(B, T, S, -1).transpose(0, 2, 1, 3).reshape(B * S, T, -1)
            tokens = temporal(tokens, conditioning=cond_temporal, freqs_cis=freqs_temporal, train=train)
            # back to [B*T, S, F]
            tokens = tokens.reshape(B, S, T, -1).transpose(0, 2, 1, 3).reshape(B * T, S, -1)

        out_frames = self.output(tokens, inv_idx, H, W)  # [B*T, H, W, C]
        return out_frames.reshape(B, T, H, W, self.output_channels)
