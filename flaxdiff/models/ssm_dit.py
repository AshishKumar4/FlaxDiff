"""
SSM (S5) based DiT blocks and a hybrid SSM-attention DiT.
S5 uses a diagonal state space with associative_scan, so it runs well on TPUs.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Any, Optional, Tuple, Sequence, Union
import einops
from functools import partial

from .vit_common import PatchEmbedding, unpatchify, RotaryEmbedding, RoPEAttention, AdaLNParams
from .common import kernel_init, FourierEmbedding, TimeProjection
from .attention import NormalAttention
from flax.typing import Dtype, PrecisionLike

from .hilbert import (
    hilbert_indices, inverse_permutation, hilbert_patchify, hilbert_unpatchify,
    zigzag_indices, zigzag_patchify, zigzag_unpatchify,
    build_2d_sincos_pos_embed,
)
from .simple_dit import DiTBlock


# --- S5 SSM Layer ---

def hippo_log_a_real_init(key, shape, dtype=jnp.float32):
    """HiPPO-diag init: A_real_n = -(n + 0.5), stored as log of the negative."""
    state_dim = shape[0]
    n = jnp.arange(state_dim, dtype=dtype)
    return jnp.log(n + 0.5).astype(dtype)


def hippo_a_imag_init(key, shape, dtype=jnp.float32):
    """HiPPO-diag init: A_imag_n = pi * n."""
    state_dim = shape[0]
    n = jnp.arange(state_dim, dtype=dtype)
    return (jnp.pi * n).astype(dtype)


class S5Layer(nn.Module):
    """S5 layer with diagonal complex state matrix.
        x_k = A * x_{k-1} + B * u_k
        y_k = Re(C * x_k) + D * u_k
    """
    features: int
    state_dim: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, u):
        # u: [B, S, F]
        B, S, F = u.shape

        # A: diagonal complex state matrix, HiPPO init, parameterized as
        # log of the negative real part for stability
        log_A_real = self.param(
            'log_A_real',
            hippo_log_a_real_init,
            (self.state_dim,)
        )
        A_imag = self.param(
            'A_imag',
            hippo_a_imag_init,
            (self.state_dim,)
        )

        # B: input-to-state projection [state_dim, F]
        B_re = self.param(
            'B_re',
            nn.initializers.lecun_normal(),
            (self.state_dim, F)
        )
        B_im = self.param(
            'B_im',
            nn.initializers.lecun_normal(),
            (self.state_dim, F)
        )

        # C: state-to-output projection [F, state_dim], lecun_normal as in S5
        C_re = self.param(
            'C_re',
            nn.initializers.lecun_normal(),
            (F, self.state_dim)
        )
        C_im = self.param(
            'C_im',
            nn.initializers.lecun_normal(),
            (F, self.state_dim)
        )

        # D: skip connection, N(0,1) per channel as in S5
        D = self.param('D', nn.initializers.normal(stddev=1.0), (F,))

        # dt: discretization timestep, learned per state dim so each state
        # channel can model its own time scale
        log_dt = self.param(
            'log_dt',
            lambda key, shape: jax.random.uniform(
                key, shape,
                minval=jnp.log(self.dt_min),
                maxval=jnp.log(self.dt_max)
            ),
            (self.state_dim,)
        )
        dt = jnp.exp(log_dt)  # [state_dim]

        # Construct complex A and discretize
        A_real = -jnp.exp(log_A_real)  # negative real part for stability
        A_diag = A_real + 1j * A_imag  # [state_dim]

        # ZOH discretization: A_bar = exp(A * dt), B_bar = (A_bar - I) * A^{-1} * B
        A_bar = jnp.exp(A_diag * dt)  # [state_dim], complex

        B_complex = B_re + 1j * B_im
        B_bar = ((A_bar[:, None] - 1.0) / (A_diag[:, None] + 1e-8)) * B_complex  # [state_dim, F]

        C_complex = C_re + 1j * C_im

        # --- Parallel Scan ---
        # x_k = A_bar * x_{k-1} + B_bar @ u_k via associative scan with
        # (a1, b1) * (a2, b2) = (a1 * a2, a2 * b1 + b2)
        u_float = u.astype(jnp.float32)
        Bu = jnp.einsum('bsf,nf->bsn', u_float, B_bar)  # [B, S, state_dim]

        A_bar_expanded = jnp.broadcast_to(A_bar[None, None, :], (B, S, self.state_dim))

        def binary_operator(e1, e2):
            a1, b1 = e1
            a2, b2 = e2
            return a1 * a2, a2 * b1 + b2

        _, x_states = jax.lax.associative_scan(
            binary_operator,
            (A_bar_expanded, Bu),
            axis=1
        )
        # x_states: [B, S, state_dim] (complex)

        # y_k = Re(C @ x_k) + D * u_k
        y_complex = jnp.einsum('fn,bsn->bsf', C_complex, x_states)  # [B, S, F]
        y = y_complex.real

        # skip connection
        y = y + D[None, None, :] * u_float  # [B, S, F]

        # cast back to input dtype
        if self.dtype is not None:
            y = y.astype(self.dtype)
        else:
            y = y.astype(u.dtype)

        return y


# --- Bidirectional S5 ---

class BidirectionalS5Layer(nn.Module):
    """Runs forward and backward S5 scans, concats and projects back to features.
    Patches have no inherent direction, so scan both ways.
    """
    features: int
    state_dim: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, u):
        # u: [B, S, F]
        y_fwd = S5Layer(
            features=self.features,
            state_dim=self.state_dim,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dtype=self.dtype,
            precision=self.precision,
            name="s5_forward"
        )(u)

        # backward scan: reverse input, scan, reverse output
        u_rev = jnp.flip(u, axis=1)
        y_bwd_rev = S5Layer(
            features=self.features,
            state_dim=self.state_dim,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dtype=self.dtype,
            precision=self.precision,
            name="s5_backward"
        )(u_rev)
        y_bwd = jnp.flip(y_bwd_rev, axis=1)

        y_cat = jnp.concatenate([y_fwd, y_bwd], axis=-1)  # [B, S, 2F]

        y = nn.Dense(
            features=self.features,
            dtype=self.dtype,
            precision=self.precision,
            name="out_proj"
        )(y_cat)

        return y


# --- 2D state fusion (Spatial-Mamba style) ---

class SpatialFusionConv(nn.Module):
    """Multi-dilation depthwise 2D convs summed as a residual over the SSM output grid.
    The 1D scan scrambles 2D locality; this recovers a direction-balanced local
    receptive field. Kernels are zero-init so the fusion starts as a pass-through.
    """
    features: int
    dilations: Tuple[int, ...] = (1, 2, 3)
    kernel_size: int = 3
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, y_2d):
        # y_2d: [B, H_P, W_P, F], SSM output reshaped to a row-major grid
        out = y_2d
        for dil in self.dilations:
            dw = nn.Conv(
                features=self.features,
                kernel_size=(self.kernel_size, self.kernel_size),
                strides=(1, 1),
                padding='SAME',
                kernel_dilation=(dil, dil),
                feature_group_count=self.features,  # depthwise
                use_bias=False,
                kernel_init=nn.initializers.zeros,
                dtype=self.dtype,
                precision=self.precision,
                name=f"dwconv_dil{dil}",
            )(y_2d)
            out = out + dw
        return out


# --- SSM DiT Block ---

class SSMDiTBlock(nn.Module):
    """Same interface as DiTBlock, but attention replaced with bidirectional S5.
    freqs_cis is accepted for interface compat but unused by the SSM.
    """
    features: int
    num_heads: int  # Not used by SSM, kept for interface compat
    rope_emb: RotaryEmbedding  # Not used by SSM, kept for interface compat
    state_dim: int = 64
    mlp_ratio: int = 4
    dropout_rate: float = 0.0
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None
    use_flash_attention: bool = False  # Ignored, interface compat
    force_fp32_for_softmax: bool = True  # Ignored, interface compat
    norm_epsilon: float = 1e-5
    use_gating: bool = True
    bidirectional: bool = True
    use_2d_fusion: bool = False  # 2D state fusion (SpatialFusionConv) after the scan
    scan_order: str = 'raster'  # parent model's scan order, needed to un-permute for the conv

    def setup(self):
        hidden_features = int(self.features * self.mlp_ratio)

        # AdaLN modulation, same as DiTBlock
        self.ada_params_module = AdaLNParams(
            self.features, dtype=self.dtype, precision=self.precision)

        self.norm1 = nn.LayerNorm(
            epsilon=self.norm_epsilon, use_scale=False, use_bias=False,
            dtype=self.dtype, name="norm1")
        self.norm2 = nn.LayerNorm(
            epsilon=self.norm_epsilon, use_scale=False, use_bias=False,
            dtype=self.dtype, name="norm2")

        # S5 SSM layer (replaces attention)
        ssm_cls = BidirectionalS5Layer if self.bidirectional else S5Layer
        self.ssm = ssm_cls(
            features=self.features,
            state_dim=self.state_dim,
            dtype=self.dtype,
            precision=self.precision,
            name="ssm"
        )

        # optional 2D state fusion after the SSM scan
        if self.use_2d_fusion:
            assert self.scan_order in ('raster', 'hilbert', 'zigzag'), \
                f"Unknown scan_order {self.scan_order}"
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
            nn.Dense(features=self.features, dtype=self.dtype, precision=self.precision)
        ])

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

        # hilbert_indices/zigzag_indices give the forward perm scan_idx[h] = k
        # (row-major index of the h-th scan token). Index arrays are constant
        # at JIT time so computing both directions is free.
        if self.scan_order == 'hilbert':
            scan_fwd = hilbert_indices(H_P, W_P)          # [S], scan→rowmajor
            scan_inv = inverse_permutation(scan_fwd, S)   # [S], rowmajor→scan
        elif self.scan_order == 'zigzag':
            scan_fwd = zigzag_indices(H_P, W_P)
            scan_inv = inverse_permutation(scan_fwd, S)
        else:  # raster
            scan_fwd = None
            scan_inv = None

        if scan_fwd is not None:
            # to row-major: rowmajor_tokens[k] = scan_tokens[scan_inv[k]]
            ssm_rm = ssm_output[:, scan_inv, :]
        else:
            ssm_rm = ssm_output

        y_2d = ssm_rm.reshape(B, H_P, W_P, F)
        y_fused_2d = self.spatial_fusion(y_2d)
        y_fused_rm = y_fused_2d.reshape(B, S, F)

        if scan_fwd is not None:
            # back to scan order: scan_tokens[h] = rowmajor_tokens[scan_fwd[h]]
            y_fused = y_fused_rm[:, scan_fwd, :]
        else:
            y_fused = y_fused_rm

        return y_fused

    @nn.compact
    def __call__(self, x, conditioning, freqs_cis):
        # Get scale/shift/gate parameters
        scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn = jnp.split(
            self.ada_params_module(conditioning), 6, axis=-1
        )

        # --- SSM Path (replaces Attention Path) ---
        residual = x
        norm_x = self.norm1(x)
        x_modulated = norm_x * (1 + scale_attn) + shift_attn
        ssm_output = self.ssm(x_modulated)

        if self.use_2d_fusion:
            ssm_output = self._apply_2d_fusion(ssm_output)

        if self.use_gating:
            x = residual + gate_attn * ssm_output
        else:
            x = residual + ssm_output

        # --- MLP Path ---
        residual = x
        norm_x_mlp = self.norm2(x)
        x_mlp_modulated = norm_x_mlp * (1 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(x_mlp_modulated)

        if self.use_gating:
            x = residual + gate_mlp * mlp_output
        else:
            x = residual + mlp_output

        return x


# --- Hybrid SSM-Attention DiT ---

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
    use_flash_attention: bool = False
    force_fp32_for_softmax: bool = True
    norm_epsilon: float = 1e-5
    learn_sigma: bool = False
    use_hilbert: bool = False
    use_zigzag: bool = False  # ZigMa-style serpentine scan
    norm_groups: int = 0
    activation: Callable = jax.nn.swish
    block_pattern: Optional[Sequence[str]] = None  # e.g., ['ssm','ssm','ssm','attn']
    ssm_attention_ratio: str = "3:1"  # e.g., "3:1", "1:1", "all-ssm", "all-attn"
    bidirectional_ssm: bool = True
    use_2d_fusion: bool = False  # 2D state fusion in SSM blocks (see SpatialFusionConv)

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
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            embedding_dim=self.emb_features,
            dtype=self.dtype,
            precision=self.precision
        )

        assert not (self.use_hilbert and self.use_zigzag), \
            "use_hilbert and use_zigzag are mutually exclusive"

        if self.use_hilbert or self.use_zigzag:
            self.hilbert_proj = nn.Dense(
                features=self.emb_features,
                dtype=self.dtype,
                precision=self.precision,
                name="hilbert_projection"
            )

        # Time embedding
        self.time_embed = nn.Sequential([
            FourierEmbedding(features=self.emb_features),
            TimeProjection(features=self.emb_features * self.mlp_ratio),
            nn.Dense(features=self.emb_features, dtype=self.dtype, precision=self.precision)
        ])

        # Text context projection
        self.text_proj = nn.Dense(
            features=self.emb_features, dtype=self.dtype,
            precision=self.precision, name="text_context_proj")

        # RoPE (used by attention blocks, passed through SSM blocks)
        self.rope = RotaryEmbedding(
            dim=self.emb_features // self.num_heads,
            max_seq_len=4096, dtype=self.dtype)

        # Build hybrid block sequence
        pattern = self._build_block_pattern()
        blocks = []
        for i, block_type in enumerate(pattern):
            if block_type == 'ssm':
                if self.use_hilbert:
                    scan_order = 'hilbert'
                elif self.use_zigzag:
                    scan_order = 'zigzag'
                else:
                    scan_order = 'raster'
                blocks.append(SSMDiTBlock(
                    features=self.emb_features,
                    num_heads=self.num_heads,
                    rope_emb=self.rope,
                    state_dim=self.ssm_state_dim,
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype,
                    precision=self.precision,
                    norm_epsilon=self.norm_epsilon,
                    bidirectional=self.bidirectional_ssm,
                    use_2d_fusion=self.use_2d_fusion,
                    scan_order=scan_order,
                    name=f"ssm_block_{i}"
                ))
            else:  # 'attn'
                blocks.append(DiTBlock(
                    features=self.emb_features,
                    num_heads=self.num_heads,
                    rope_emb=self.rope,
                    mlp_ratio=self.mlp_ratio,
                    dropout_rate=self.dropout_rate,
                    dtype=self.dtype,
                    precision=self.precision,
                    use_flash_attention=self.use_flash_attention,
                    force_fp32_for_softmax=self.force_fp32_for_softmax,
                    norm_epsilon=self.norm_epsilon,
                    name=f"dit_block_{i}"
                ))
        self.blocks = blocks

        # Final layer
        self.final_norm = nn.LayerNorm(
            epsilon=self.norm_epsilon, dtype=self.dtype, name="final_norm")

        output_dim = self.patch_size * self.patch_size * self.output_channels
        if self.learn_sigma:
            output_dim *= 2

        self.final_proj = nn.Dense(
            features=output_dim,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=nn.initializers.zeros,
            name="final_proj"
        )

    @nn.compact
    def __call__(self, x, temb, textcontext=None):
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        H_P = H // self.patch_size
        W_P = W // self.patch_size

        # 1. Patch Embedding
        hilbert_inv_idx = None
        if self.use_hilbert:
            patches_raw, hilbert_inv_idx = hilbert_patchify(x, self.patch_size)
            patches = self.hilbert_proj(patches_raw)
        elif self.use_zigzag:
            patches_raw, hilbert_inv_idx = zigzag_patchify(x, self.patch_size)
            patches = self.hilbert_proj(patches_raw)
        else:
            patches = self.patch_embed(x)

        num_patches = patches.shape[1]

        # 2D sincos position embedding - the SSM blocks ignore RoPE so they need
        # an explicit positional signal. For hilbert/zigzag, reorder the row-major
        # embedding to match the scan so each patch gets its real 2D position.
        pos_embed_2d_rm = build_2d_sincos_pos_embed(self.emb_features, H_P, W_P)
        pos_embed_2d_rm = jnp.asarray(pos_embed_2d_rm, dtype=patches.dtype)
        if self.use_hilbert:
            scan_idx = hilbert_indices(H_P, W_P)
            pos_embed_2d = pos_embed_2d_rm[scan_idx]
        elif self.use_zigzag:
            scan_idx = zigzag_indices(H_P, W_P)
            pos_embed_2d = pos_embed_2d_rm[scan_idx]
        else:
            pos_embed_2d = pos_embed_2d_rm
        patches = patches + pos_embed_2d[None, :, :]

        x_seq = patches

        # 2. Conditioning
        t_emb = self.time_embed(temb)
        cond_emb = t_emb
        if textcontext is not None:
            text_emb = self.text_proj(textcontext)
            text_emb_pooled = jnp.mean(text_emb, axis=1)
            cond_emb = cond_emb + text_emb_pooled

        # 3. RoPE frequencies for the attention blocks. With hilbert/zigzag the
        # sequence index is not a 2D position and RoPE distances would be wrong,
        # so use identity freqs (cos=1, sin=0) - the 2D sincos above carries position.
        freqs_cos, freqs_sin = self.rope(seq_len=num_patches)
        if self.use_hilbert or self.use_zigzag:
            freqs_cos = jnp.ones_like(freqs_cos)
            freqs_sin = jnp.zeros_like(freqs_sin)

        # 4. Hybrid blocks (SSM and attention interleaved)
        for block in self.blocks:
            x_seq = block(x_seq, conditioning=cond_emb, freqs_cis=(freqs_cos, freqs_sin))

        # 5. Final output
        x_out = self.final_norm(x_seq)
        x_out = self.final_proj(x_out)

        # 6. Unpatchify
        if self.use_hilbert or self.use_zigzag:
            if self.learn_sigma:
                x_mean, _x_logvar = jnp.split(x_out, 2, axis=-1)
                return hilbert_unpatchify(x_mean, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
            return hilbert_unpatchify(x_out, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
        if self.learn_sigma:
            x_mean, _x_logvar = jnp.split(x_out, 2, axis=-1)
            return unpatchify(x_mean, channels=self.output_channels)
        return unpatchify(x_out, channels=self.output_channels)
