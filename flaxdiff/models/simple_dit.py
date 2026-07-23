import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from flax.typing import Dtype, PrecisionLike

from .dit_common import (
    PatchSequenceEmbed, ConditioningEmbed, PatchSequenceOutput,
    ModulatedBlock, neutralized_rope_freqs,
)
from .vit_common import RotaryEmbedding


class SimpleDiT(nn.Module):
    """Standard DiT: a plain stack of adaLN-Zero attention blocks."""
    output_channels: int = 3
    patch_size: int = 16
    emb_features: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    dropout_rate: float = 0.0  # Typically 0 for diffusion
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False
    qk_norm: bool = False
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
        self.rope = RotaryEmbedding(
            dim=self.emb_features // self.num_heads, max_seq_len=4096, dtype=self.dtype)
        self.blocks = [
            ModulatedBlock(
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
                name=f"dit_block_{i}"
            ) for i in range(self.num_layers)
        ]
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
            x_seq = block(x_seq, conditioning=cond_emb, freqs_cis=freqs_cis, train=train)

        return self.output(x_seq, inv_idx, H, W)
