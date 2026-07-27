"""
Hybrid SSM-attention DiT: interleaves linear-time S5 blocks with attention
blocks in a configurable ratio. The S5 layers themselves live in s5.py; the
block and the patchify/conditioning/output machinery live in dit_common.py.
"""

import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence
from flax.typing import Dtype, PrecisionLike

from .dit_common import (
    PatchSequenceEmbed, ConditioningEmbed, PatchSequenceOutput,
    ModulatedBlock, remat_block, neutralized_rope_freqs,
)
from .vit_common import RotaryEmbedding


class HybridSSMAttentionDiT(nn.Module):
    """DiT that interleaves SSM blocks with attention blocks in a configurable ratio.
    block_pattern (e.g. ['ssm','ssm','ssm','attn']) overrides ssm_attention_ratio ('3:1').
    """
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    ssm_state_dim: int = 64
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False
    qk_norm: bool = False
    attention_impl: Optional[str] = None
    remat: bool = False
    use_hilbert: bool = False
    use_zigzag: bool = False  # ZigMa-style serpentine scan
    block_pattern: Optional[Sequence[str]] = None  # e.g., ['ssm','ssm','ssm','attn']
    ssm_attention_ratio: str = "3:1"  # e.g., "3:1", "1:1", "all-ssm", "all-attn"
    bidirectional_ssm: bool = True
    use_2d_fusion: bool = False  # 2D state fusion in SSM blocks (see SpatialFusionConv)

    @property
    def scan_order(self):
        assert not (self.use_hilbert and self.use_zigzag), \
            "use_hilbert and use_zigzag are mutually exclusive"
        return 'hilbert' if self.use_hilbert else 'zigzag' if self.use_zigzag else 'raster'

    def _build_block_pattern(self):
        """Generate block pattern from ratio string."""
        if self.block_pattern is not None:
            pattern = list(self.block_pattern)
        elif self.ssm_attention_ratio == "all-ssm":
            pattern = ['ssm'] * self.num_layers
        elif self.ssm_attention_ratio == "all-attn":
            pattern = ['attn'] * self.num_layers
        else:
            parts = self.ssm_attention_ratio.split(':')
            n_ssm, n_attn = int(parts[0]), int(parts[1])
            unit = ['ssm'] * n_ssm + ['attn'] * n_attn
            pattern = (unit * (self.num_layers // len(unit) + 1))[:self.num_layers]
        return pattern

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
        self.rope = RotaryEmbedding(
            dim=self.emb_features // self.num_heads,
            max_seq_len=4096, dtype=self.dtype)

        pattern = self._build_block_pattern()
        blocks = []
        for i, block_type in enumerate(pattern):
            if block_type == 'ssm':
                blocks.append(remat_block(ModulatedBlock, self.remat, policy=None)(
                    features=self.emb_features,
                    num_heads=self.num_heads,
                    rope_emb=self.rope,
                    mixer='ssm',
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype,
                    precision=self.precision,
                    norm_epsilon=self.norm_epsilon,
                    ssm_state_dim=self.ssm_state_dim,
                    bidirectional_ssm=self.bidirectional_ssm,
                    use_2d_fusion=self.use_2d_fusion,
                    scan_order=self.scan_order,
                    name=f"ssm_block_{i}"
                ))
            else:  # 'attn'
                blocks.append(remat_block(ModulatedBlock, self.remat)(
                    features=self.emb_features,
                    num_heads=self.num_heads,
                    rope_emb=self.rope,
                    mixer='attention',
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype,
                    precision=self.precision,
                    force_fp32_for_softmax=self.force_fp32_for_softmax,
                    norm_epsilon=self.norm_epsilon,
                    qk_norm=self.qk_norm,
                    attention_impl=self.attention_impl,
                    name=f"dit_block_{i}"
                ))
        self.blocks = blocks

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
        B, H, W, C = x.shape
        x_seq, inv_idx = self.embed(x)
        cond_emb = self.conditioning(temb, textcontext)
        freqs_cis = neutralized_rope_freqs(self.rope, x_seq.shape[1], self.scan_order)

        for block in self.blocks:
            x_seq = block(x_seq, cond_emb, freqs_cis, train)

        return self.output(x_seq, inv_idx, H, W)
