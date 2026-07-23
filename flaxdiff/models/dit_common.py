"""
Shared machinery for the DiT family.

Every DiT-style model here is the same sandwich: patchify in some scan order,
add a 2D sincos position signal, run adaLN-Zero modulated blocks over the
token sequence, and unpatchify back. The only real differences between the
models are the token mixer inside the block (attention or S5 SSM) and how the
blocks are arranged (plain stack, U-shaped skips, hybrid patterns). This
module owns the sandwich; the model files just arrange blocks.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from flax.typing import Dtype, PrecisionLike

from .common import FourierEmbedding, TimeProjection
from .vit_common import PatchEmbedding, RotaryEmbedding, RoPEAttention, AdaLNParams, unpatchify
from .s5 import S5Layer, BidirectionalS5Layer, SpatialFusionConv
from .hilbert import (
    hilbert_indices, inverse_permutation, hilbert_patchify, hilbert_unpatchify,
    zigzag_indices, zigzag_patchify,
    build_2d_sincos_pos_embed,
)

SCAN_ORDERS = ('raster', 'hilbert', 'zigzag')


def scan_indices(scan_order: str, H_P: int, W_P: int):
    """Forward permutation for a scan order (None for raster)."""
    if scan_order == 'hilbert':
        return hilbert_indices(H_P, W_P)
    if scan_order == 'zigzag':
        return zigzag_indices(H_P, W_P)
    return None


class PatchSequenceEmbed(nn.Module):
    """Patchify in raster/hilbert/zigzag order and add the 2D sincos signal.

    Returns (tokens, inv_idx) - inv_idx restores row-major order on the way
    out and is None for raster.
    """
    patch_size: int
    emb_features: int
    scan_order: str = 'raster'
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        assert self.scan_order in SCAN_ORDERS, f"Unknown scan order {self.scan_order}"
        if self.scan_order == 'raster':
            self.patch_embed = PatchEmbedding(
                patch_size=self.patch_size,
                embedding_dim=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="patch_embed",
            )
        else:
            # Raw patch extraction + Dense instead of the Conv-based embedding
            self.scan_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection",
            )

    def __call__(self, x):
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Image dimensions must be divisible by patch size"
        H_P, W_P = H // self.patch_size, W // self.patch_size

        inv_idx = None
        if self.scan_order == 'hilbert':
            patches_raw, inv_idx = hilbert_patchify(x, self.patch_size)
            tokens = self.scan_proj(patches_raw)
        elif self.scan_order == 'zigzag':
            patches_raw, inv_idx = zigzag_patchify(x, self.patch_size)
            tokens = self.scan_proj(patches_raw)
        else:
            tokens = self.patch_embed(x)

        # Fixed 2D sincos position embedding (order-invariant spatial signal).
        # For hilbert/zigzag, reorder the row-major embedding to match the scan
        # so each token gets the sincos for its real 2D position.
        pos_embed = build_2d_sincos_pos_embed(self.emb_features, H_P, W_P)
        pos_embed = jnp.asarray(pos_embed, dtype=tokens.dtype)
        idx = scan_indices(self.scan_order, H_P, W_P)
        if idx is not None:
            pos_embed = pos_embed[idx]
        tokens = tokens + pos_embed[None, :, :]
        return tokens, inv_idx


class ConditioningEmbed(nn.Module):
    """Fourier time embedding + mean-pooled text projection, summed into the
    single conditioning vector the adaLN modulation consumes."""
    emb_features: int
    mlp_ratio: int = 4
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    def setup(self):
        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features),
            TimeProjection(features=self.emb_features * self.mlp_ratio),
            nn.Dense(features=self.emb_features, dtype=self.dtype, precision=self.precision),
        ], name="time_embed")
        self.text_proj = nn.Dense(
            features=self.emb_features, dtype=self.dtype,
            precision=self.precision, name="text_context_proj")

    def __call__(self, temb, textcontext=None):
        cond_emb = self.time_embed(temb)
        if textcontext is not None:
            text_emb = self.text_proj(textcontext)
            cond_emb = cond_emb + jnp.mean(text_emb, axis=1)
        return cond_emb


class PatchSequenceOutput(nn.Module):
    """Final norm + zero-init fp32 head + unpatchify for any scan order and
    any (non-square included) patch grid."""
    patch_size: int
    output_channels: int
    learn_sigma: bool = False
    modulated: bool = False  # adaLN shift/scale on the final norm (DiT FinalLayer)
    norm_epsilon: float = 1e-5
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, tokens, inv_idx, H, W, conditioning=None):
        features = tokens.shape[-1]
        x_out = nn.LayerNorm(
            epsilon=self.norm_epsilon, use_scale=not self.modulated,
            use_bias=not self.modulated, dtype=self.dtype, name="final_norm")(tokens)
        if self.modulated:
            assert conditioning is not None, "modulated output head needs the conditioning vector"
            if conditioning.ndim == 2:
                conditioning = jnp.expand_dims(conditioning, axis=1)
            shift, scale = jnp.split(nn.Dense(
                features=2 * features,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=nn.initializers.zeros,
                name="final_ada_proj",
            )(nn.silu(conditioning)), 2, axis=-1)
            x_out = x_out * (1 + scale) + shift

        output_dim = self.patch_size * self.patch_size * self.output_channels
        if self.learn_sigma:
            output_dim *= 2
        x_out = nn.Dense(
            features=output_dim,
            dtype=jnp.float32,  # fp32 output head - the loss is computed in fp32
            precision=self.precision,
            kernel_init=nn.initializers.zeros,
            name="final_proj",
        )(x_out)

        if self.learn_sigma:
            x_out, _x_logvar = jnp.split(x_out, 2, axis=-1)
        if inv_idx is not None:
            return hilbert_unpatchify(x_out, inv_idx, self.patch_size, H, W, self.output_channels)
        return unpatchify(x_out, channels=self.output_channels,
                          H_P=H // self.patch_size, W_P=W // self.patch_size)


class ModulatedBlock(nn.Module):
    """adaLN-Zero modulated residual block with a pluggable token mixer.

    mixer='attention' gives the standard DiT block (RoPE attention);
    mixer='ssm' replaces attention with a bidirectional S5 scan, optionally
    followed by Spatial-Mamba style 2D state fusion. freqs_cis is unused by
    the SSM mixer but kept in the interface so blocks are interchangeable.
    """
    features: int
    num_heads: int
    rope_emb: RotaryEmbedding = None
    mixer: str = 'attention'
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    use_gating: bool = True
    qk_norm: bool = False
    attention_impl: Optional[str] = None  # None (reference) | 'xla' | 'cudnn' | 'tpu'
    # ssm mixer options
    ssm_state_dim: int = 64
    bidirectional_ssm: bool = True
    use_2d_fusion: bool = False
    scan_order: str = 'raster'  # needed to un-permute for the 2D fusion conv

    def setup(self):
        assert self.mixer in ('attention', 'ssm'), f"Unknown mixer {self.mixer}"
        hidden_features = int(self.features * self.mlp_ratio)

        self.ada_params_module = AdaLNParams(
            self.features, dtype=self.dtype, precision=self.precision)
        self.norm1 = nn.LayerNorm(
            epsilon=self.norm_epsilon, use_scale=False, use_bias=False,
            dtype=self.dtype, name="norm1")
        self.norm2 = nn.LayerNorm(
            epsilon=self.norm_epsilon, use_scale=False, use_bias=False,
            dtype=self.dtype, name="norm2")

        if self.mixer == 'attention':
            self.attention = RoPEAttention(
                query_dim=self.features,
                heads=self.num_heads,
                dim_head=self.features // self.num_heads,
                dtype=self.dtype,
                precision=self.precision,
                use_bias=True,
                qk_norm=self.qk_norm,
                attention_impl=self.attention_impl,
                force_fp32_for_softmax=self.force_fp32_for_softmax,
                rope_emb=self.rope_emb,
            )
        else:
            ssm_cls = BidirectionalS5Layer if self.bidirectional_ssm else S5Layer
            self.ssm = ssm_cls(
                features=self.features,
                state_dim=self.ssm_state_dim,
                dtype=self.dtype,
                precision=self.precision,
                name="ssm",
            )
            if self.use_2d_fusion:
                assert self.scan_order in SCAN_ORDERS, f"Unknown scan_order {self.scan_order}"
                self.spatial_fusion = SpatialFusionConv(
                    features=self.features,
                    dilations=(1, 2, 3),
                    kernel_size=3,
                    dtype=self.dtype,
                    precision=self.precision,
                    name="spatial_fusion",
                )

        self.mlp = nn.Sequential([
            nn.Dense(features=hidden_features, dtype=self.dtype, precision=self.precision),
            nn.gelu,
            nn.Dense(features=self.features, dtype=self.dtype, precision=self.precision),
        ])
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def _apply_2d_fusion(self, ssm_output):
        """Un-permute scan-ordered SSM output to a 2D grid, fuse, re-permute back."""
        B, S, F = ssm_output.shape
        # square patch grid; S is a static python int at trace time
        import math
        H_P = math.isqrt(S)
        W_P = H_P
        assert H_P * W_P == S, (
            f"2D fusion requires a square patch grid; got S={S} which is not a "
            f"perfect square.")

        # Index arrays are constant at JIT time so computing both directions is free
        scan_fwd = scan_indices(self.scan_order, H_P, W_P)
        if scan_fwd is not None:
            scan_inv = inverse_permutation(scan_fwd, S)
            ssm_rm = ssm_output[:, scan_inv, :]
        else:
            ssm_rm = ssm_output

        y_2d = ssm_rm.reshape(B, H_P, W_P, F)
        y_fused_2d = self.spatial_fusion(y_2d)
        y_fused_rm = y_fused_2d.reshape(B, S, F)

        if scan_fwd is not None:
            y_fused = y_fused_rm[:, scan_fwd, :]
        else:
            y_fused = y_fused_rm
        return y_fused

    @nn.compact
    def __call__(self, x, conditioning, freqs_cis, train: bool = False):
        scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn = jnp.split(
            self.ada_params_module(conditioning), 6, axis=-1
        )

        # --- Mixer path (attention or SSM) ---
        residual = x
        x_modulated = self.norm1(x) * (1 + scale_attn) + shift_attn
        if self.mixer == 'attention':
            mixer_output = self.attention(x_modulated, context=None, freqs_cis=freqs_cis)
        else:
            mixer_output = self.ssm(x_modulated)
            if self.use_2d_fusion:
                mixer_output = self._apply_2d_fusion(mixer_output)
        mixer_output = self.dropout(mixer_output, deterministic=not train)

        if self.use_gating:
            x = residual + gate_attn * mixer_output
        else:
            x = residual + mixer_output

        # --- MLP path ---
        residual = x
        x_mlp_modulated = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(x_mlp_modulated)
        mlp_output = self.dropout(mlp_output, deterministic=not train)

        if self.use_gating:
            x = residual + gate_mlp * mlp_output
        else:
            x = residual + mlp_output
        return x


def neutralized_rope_freqs(rope: RotaryEmbedding, seq_len: int, scan_order: str):
    """RoPE frequencies for the sequence, neutralized to identity for
    hilbert/zigzag scans where the 1D index is not a 2D position (the 2D
    sincos embedding already carries position there)."""
    freqs_cos, freqs_sin = rope(seq_len=seq_len)
    if scan_order != 'raster':
        freqs_cos = jnp.ones_like(freqs_cos)
        freqs_sin = jnp.zeros_like(freqs_sin)
    return freqs_cos, freqs_sin
