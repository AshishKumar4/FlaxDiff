"""JEPA encoders and predictors, arranged from the shared DiT machinery.

A JEPA encoder is the DiT sandwich without the diffusion parts: patchify with
the 2D sincos signal, run unmodulated ModulatedBlocks over the tokens, norm.
There is no timestep to condition on, so the blocks run in their unmodulated
(plain pre-norm) mode, and the mixer is still pluggable - mixer patterns with
'ssm' give a linear-time S5 encoder.

Position never comes from RoPE here. Both the encoder and the predictor work
on a masked subset of the sequence, where a token's index in the sequence is
not its position on the grid, so RoPE is left at identity and position is
carried entirely by the 2D sincos embedding that travels with each token.
"""

import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence, Tuple
from flax.typing import Dtype, PrecisionLike

from ..models.dit_common import (
    PatchSequenceEmbed, ModulatedBlock, build_block_pattern,
    identity_rope_freqs, scan_ordered_pos_embed,
)
from ..models.vit_common import RotaryEmbedding


def gather_tokens(tokens, idx):
    """Select tokens per sample: [B, S, F] and [B, N] -> [B, N, F]."""
    return jnp.take_along_axis(tokens, idx[..., None], axis=-2)


class TokenStack(nn.Module):
    """A stack of unmodulated blocks over a token sequence."""
    features: int
    num_layers: int
    num_heads: int
    mlp_ratio: int = 4
    ssm_attention_ratio: str = "all-attn"
    block_pattern: Optional[Sequence[str]] = None
    ssm_state_dim: int = 64
    bidirectional_ssm: bool = True
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    qk_norm: bool = False
    attention_impl: Optional[str] = None

    def setup(self):
        pattern = build_block_pattern(
            self.num_layers, self.ssm_attention_ratio, self.block_pattern)
        self.blocks = [
            ModulatedBlock(
                features=self.features,
                num_heads=self.num_heads,
                mixer='ssm' if kind == 'ssm' else 'attention',
                modulated=False,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                norm_epsilon=self.norm_epsilon,
                qk_norm=self.qk_norm,
                attention_impl=self.attention_impl,
                ssm_state_dim=self.ssm_state_dim,
                bidirectional_ssm=self.bidirectional_ssm,
                name=f"block_{i}",
            ) for i, kind in enumerate(pattern)
        ]

    def __call__(self, tokens, freqs_cis=None, train: bool = False):
        if freqs_cis is None:
            freqs_cis = identity_rope_freqs(tokens.shape[-2], self.features // self.num_heads)
        for block in self.blocks:
            tokens = block(tokens, conditioning=None, freqs_cis=freqs_cis, train=train)
        return tokens


class FactorizedTokenStack(nn.Module):
    """Spatial then temporal blocks over [B, T, N, F], as in VideoDiT.

    Time is a real 1D axis that masking never touches, so the temporal half
    keeps genuine RoPE while the spatial half runs at identity.
    """
    features: int
    num_layers: int
    num_heads: int
    max_frames: int = 1024
    mlp_ratio: int = 4
    ssm_attention_ratio: str = "all-attn"
    block_pattern: Optional[Sequence[str]] = None
    ssm_state_dim: int = 64
    bidirectional_ssm: bool = True
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    qk_norm: bool = False
    attention_impl: Optional[str] = None

    def setup(self):
        def stack(name, num_layers, ratio):
            return TokenStack(
                features=self.features, num_layers=num_layers, num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio, ssm_attention_ratio=ratio,
                ssm_state_dim=self.ssm_state_dim, bidirectional_ssm=self.bidirectional_ssm,
                dropout_rate=self.dropout_rate, dtype=self.dtype, precision=self.precision,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                norm_epsilon=self.norm_epsilon, qk_norm=self.qk_norm,
                attention_impl=self.attention_impl, name=name)

        # one spatial and one temporal block per layer, built as single-block
        # stacks so the two halves can be interleaved
        pattern = build_block_pattern(
            self.num_layers, self.ssm_attention_ratio, self.block_pattern)
        self.spatial = [stack(f"spatial_{i}", 1, "all-ssm" if kind == 'ssm' else "all-attn")
                        for i, kind in enumerate(pattern)]
        self.temporal = [stack(f"temporal_{i}", 1, "all-attn") for i in range(self.num_layers)]
        self.temporal_rope = RotaryEmbedding(
            dim=self.features // self.num_heads, max_seq_len=self.max_frames,
            name="temporal_rope")

    def __call__(self, tokens, train: bool = False):
        B, T, N, F = tokens.shape
        freqs_temporal = self.temporal_rope(seq_len=T)

        tokens = tokens.reshape(B * T, N, F)
        for spatial, temporal in zip(self.spatial, self.temporal):
            tokens = spatial(tokens, train=train)
            tokens = tokens.reshape(B, T, N, F).transpose(0, 2, 1, 3).reshape(B * N, T, F)
            tokens = temporal(tokens, freqs_cis=freqs_temporal, train=train)
            tokens = tokens.reshape(B, N, T, F).transpose(0, 2, 1, 3).reshape(B * T, N, F)
        return tokens.reshape(B, T, N, F)


class JepaEncoder(nn.Module):
    """ViT over an image, optionally restricted to a subset of its patches."""
    patch_size: int = 16
    emb_features: int = 384
    num_layers: int = 12
    num_heads: int = 6
    mlp_ratio: int = 4
    ssm_attention_ratio: str = "all-attn"
    ssm_state_dim: int = 64
    bidirectional_ssm: bool = True
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
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
        self.stack = TokenStack(
            features=self.emb_features, num_layers=self.num_layers,
            num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
            ssm_attention_ratio=self.ssm_attention_ratio,
            ssm_state_dim=self.ssm_state_dim, bidirectional_ssm=self.bidirectional_ssm,
            dropout_rate=self.dropout_rate, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            norm_epsilon=self.norm_epsilon, qk_norm=self.qk_norm,
            attention_impl=self.attention_impl,
        )
        self.norm = nn.LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")

    def __call__(self, x, token_idx=None, train: bool = False):
        tokens, _ = self.embed(x)
        if token_idx is not None:
            tokens = gather_tokens(tokens, token_idx)
        return self.norm(self.stack(tokens, train=train))


class JepaVideoEncoder(nn.Module):
    """Factorized spatial-temporal encoder over (B, T, H, W, C).

    token_idx selects a tubelet: the same patch positions in every frame, so
    the factorized layout survives masking untouched.
    """
    patch_size: int = 16
    emb_features: int = 384
    num_layers: int = 12
    num_heads: int = 6
    mlp_ratio: int = 4
    ssm_attention_ratio: str = "all-attn"
    ssm_state_dim: int = 64
    bidirectional_ssm: bool = True
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
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
        self.stack = FactorizedTokenStack(
            features=self.emb_features, num_layers=self.num_layers,
            num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
            ssm_attention_ratio=self.ssm_attention_ratio,
            ssm_state_dim=self.ssm_state_dim, bidirectional_ssm=self.bidirectional_ssm,
            dropout_rate=self.dropout_rate, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            norm_epsilon=self.norm_epsilon, qk_norm=self.qk_norm,
            attention_impl=self.attention_impl,
        )
        self.norm = nn.LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")

    def __call__(self, x, token_idx=None, train: bool = False):
        B, T, H, W, C = x.shape
        tokens, _ = self.embed(x.reshape(B * T, H, W, C))
        if token_idx is not None:
            tokens = gather_tokens(tokens, jnp.repeat(token_idx, T, axis=0))
        tokens = tokens.reshape(B, T, tokens.shape[1], self.emb_features)
        return self.norm(self.stack(tokens, train=train))


class JepaPredictor(nn.Module):
    """Narrow transformer from context embeddings to target embeddings.

    Context tokens are projected down, mask tokens stand in for the targets,
    and both carry the sincos signal for the grid position they belong to.
    """
    grid: Tuple[int, int] = (14, 14)
    emb_features: int = 384      # encoder width, in and out
    predictor_features: int = 192
    num_layers: int = 6
    num_heads: int = 6
    mlp_ratio: int = 4
    ssm_attention_ratio: str = "all-attn"
    ssm_state_dim: int = 64
    bidirectional_ssm: bool = True
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    qk_norm: bool = False
    attention_impl: Optional[str] = None
    scan_order: str = 'raster'
    factorized: bool = False     # space-time blocks, for video

    def setup(self):
        self.proj_in = nn.Dense(features=self.predictor_features, dtype=self.dtype,
                                precision=self.precision, name="proj_in")
        self.mask_token = self.param(
            "mask_token", nn.initializers.normal(0.02), (1, 1, self.predictor_features))
        stack_kwargs = dict(
            features=self.predictor_features, num_layers=self.num_layers,
            num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
            ssm_attention_ratio=self.ssm_attention_ratio,
            ssm_state_dim=self.ssm_state_dim, bidirectional_ssm=self.bidirectional_ssm,
            dropout_rate=self.dropout_rate, dtype=self.dtype, precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            norm_epsilon=self.norm_epsilon, qk_norm=self.qk_norm,
            attention_impl=self.attention_impl,
        )
        self.stack = (FactorizedTokenStack(**stack_kwargs) if self.factorized
                      else TokenStack(**stack_kwargs))
        self.norm = nn.LayerNorm(epsilon=self.norm_epsilon, dtype=self.dtype, name="norm")
        self.proj_out = nn.Dense(features=self.emb_features, dtype=self.dtype,
                                 precision=self.precision, name="proj_out")

    def __call__(self, context, context_idx, target_idx, train: bool = False):
        """context: [B, (T,) N_ctx, F] -> predictions [B, (T,) N_tgt, F]."""
        pos_embed = jnp.asarray(
            scan_ordered_pos_embed(self.predictor_features, *self.grid, self.scan_order),
            dtype=self.dtype or jnp.float32)

        def positions(idx):
            pos = pos_embed[idx]                       # [B, N, P]
            return pos[:, None] if self.factorized else pos

        num_target_tokens = target_idx.shape[-1]
        context = self.proj_in(context) + positions(context_idx)
        targets = self.mask_token + positions(target_idx)
        if self.factorized:
            targets = jnp.broadcast_to(
                targets, (*context.shape[:-2], num_target_tokens, self.predictor_features))

        tokens = jnp.concatenate([context, targets], axis=-2)
        tokens = self.stack(tokens, train=train)
        return self.proj_out(self.norm(tokens[..., -num_target_tokens:, :]))
