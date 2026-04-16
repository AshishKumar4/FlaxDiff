"""
SSM-based DiT Block and Hybrid SSM-Attention DiT for diffusion models.

Implements S5 (Simplified State Space) layers as a drop-in replacement for
attention in DiT blocks. The S5 layer uses diagonal state spaces with
JAX associative_scan for efficient parallel computation on TPUs.

References:
    - Smith et al., "Simplified State Space Layers for Sequence Modeling", ICLR 2023
    - Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023
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


# =============================================================================
# S5 SSM Layer - Diagonal State Space with Parallel Scan
# =============================================================================

def hippo_log_a_real_init(key, shape, dtype=jnp.float32):
    """HiPPO-diag init for log(|A_real|).

    Standard S5 / HiPPO-diag init: A_real_n = -(n + 0.5).
    We store as log of the negative, so A_real = -exp(log_A_real_init) = -(n + 0.5).
    """
    state_dim = shape[0]
    n = jnp.arange(state_dim, dtype=dtype)
    return jnp.log(n + 0.5).astype(dtype)


def hippo_a_imag_init(key, shape, dtype=jnp.float32):
    """HiPPO-diag init for A_imag.

    Standard S5 / HiPPO-diag init: A_imag_n = pi * n.
    """
    state_dim = shape[0]
    n = jnp.arange(state_dim, dtype=dtype)
    return (jnp.pi * n).astype(dtype)


class S5Layer(nn.Module):
    """S5 (Simplified State Space) layer with diagonal state matrix.

    Processes sequences using state space recurrence:
        x_k = A * x_{k-1} + B * u_k
        y_k = Re(C * x_k) + D * u_k

    Where A is diagonal (complex), enabling efficient parallel scan via
    jax.lax.associative_scan. This is TPU-friendly as it maps to
    prefix-sum operations on the systolic array.

    Args:
        features: Output dimension (must match input for residual)
        state_dim: Dimension of the hidden state (per feature)
        dt_min: Minimum discretization step (log scale)
        dt_max: Maximum discretization step (log scale)
        dtype: Computation dtype
        precision: JAX precision level
    """
    features: int
    state_dim: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, u):
        """
        Args:
            u: Input sequence [B, S, F]
        Returns:
            y: Output sequence [B, S, F]
        """
        B, S, F = u.shape

        # --- Learnable SSM Parameters ---
        # A: Diagonal state matrix (complex) - initialized with HiPPO-LegS diagonal scheme.
        # We parameterize as log of negative real part for stability:
        #   A_real_n = -exp(log_A_real_n) = -(n + 0.5)  (after HiPPO init)
        #   A_imag_n =  pi * n                          (after HiPPO init)
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

        # B: Input-to-state projection [state_dim, F]
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

        # C: State-to-output projection [F, state_dim].
        # lecun_normal matches the canonical S5 / S4D initialization (Smith et al. 2022
        # and Gu et al. 2022). An earlier experiment used zeros-init to bias the block
        # toward a pure-identity start, but this actively hurt convergence in our
        # ablation (pfkggns2 vs mq05643r), so we reverted to the canonical scheme.
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

        # D: Skip connection (direct input-to-output). Canonical S5 samples D per-channel
        # from N(0, 1); an earlier ones-init shortcut was non-canonical and reverted.
        D = self.param('D', nn.initializers.normal(stddev=1.0), (F,))

        # dt: Discretization timestep, learned PER STATE DIMENSION (standard S5).
        # Per-state-dim dt allows different state channels to model different time scales,
        # which is the core inductive bias of S4/S5 over a generic linear RNN.
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

        # --- Construct complex A and discretize ---
        # A_diag: complex diagonal [state_dim]
        A_real = -jnp.exp(log_A_real)  # Ensure negative real part for stability
        A_diag = A_real + 1j * A_imag  # [state_dim]

        # ZOH discretization: A_bar = exp(A * dt), B_bar = (A_bar - I) * A^{-1} * B
        # Both A_diag and dt are [state_dim], so this is a proper element-wise per-state
        # discretization (no averaging).
        A_bar = jnp.exp(A_diag * dt)  # [state_dim], complex

        # B as complex: [state_dim, F]
        B_complex = B_re + 1j * B_im
        # Discretized B: element-wise per state dimension
        B_bar = ((A_bar[:, None] - 1.0) / (A_diag[:, None] + 1e-8)) * B_complex  # [state_dim, F]

        # C as complex: [F, state_dim]
        C_complex = C_re + 1j * C_im

        # --- Parallel Scan (associative scan for TPU efficiency) ---
        # For diagonal SSM: x_k = A_bar * x_{k-1} + B_bar @ u_k
        # This can be computed via associative scan with the binary operator:
        #   (a1, b1) * (a2, b2) = (a1 * a2, a2 * b1 + b2)

        # Compute per-step inputs: B_bar @ u_k for each timestep
        # u: [B, S, F], B_bar: [state_dim, F]
        # Bu: [B, S, state_dim] (complex)
        u_float = u.astype(jnp.float32)
        Bu = jnp.einsum('bsf,nf->bsn', u_float, B_bar)  # [B, S, state_dim]

        # A_bar is the same for all timesteps: [state_dim]
        # Expand for scan: [B, S, state_dim]
        A_bar_expanded = jnp.broadcast_to(A_bar[None, None, :], (B, S, self.state_dim))

        # Associative scan elements: (a, b) where a=A_bar, b=Bu
        def binary_operator(e1, e2):
            a1, b1 = e1
            a2, b2 = e2
            return a1 * a2, a2 * b1 + b2

        # Run parallel scan along sequence dimension (axis=1)
        _, x_states = jax.lax.associative_scan(
            binary_operator,
            (A_bar_expanded, Bu),
            axis=1
        )
        # x_states: [B, S, state_dim] (complex) - hidden states at each step

        # --- Output computation ---
        # y_k = Re(C @ x_k) + D * u_k
        # C: [F, state_dim], x_states: [B, S, state_dim]
        y_complex = jnp.einsum('fn,bsn->bsf', C_complex, x_states)  # [B, S, F]
        y = y_complex.real  # Take real part: [B, S, F]

        # Add skip connection
        y = y + D[None, None, :] * u_float  # [B, S, F]

        # Cast back to input dtype
        if self.dtype is not None:
            y = y.astype(self.dtype)
        else:
            y = y.astype(u.dtype)

        return y


# =============================================================================
# Bidirectional S5 - processes sequence in both directions
# =============================================================================

class BidirectionalS5Layer(nn.Module):
    """Bidirectional S5 layer - runs forward and backward scans.

    For diffusion models, spatial patches have no inherent directionality,
    so bidirectional processing captures dependencies in both directions
    along the serialization curve (Hilbert, raster, etc.).

    Forward and backward outputs are concatenated then linearly projected
    back to `features`. Concat-then-project preserves the directional
    information that a sum would discard.
    """
    features: int
    state_dim: int = 64
    dt_min: float = 0.001
    dt_max: float = 0.1
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, u):
        """
        Args:
            u: [B, S, F]
        Returns:
            y: [B, S, F]
        """
        # Forward scan
        y_fwd = S5Layer(
            features=self.features,
            state_dim=self.state_dim,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dtype=self.dtype,
            precision=self.precision,
            name="s5_forward"
        )(u)

        # Backward scan (reverse input, scan, reverse output)
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

        # Concatenate forward and backward (preserves directional information).
        y_cat = jnp.concatenate([y_fwd, y_bwd], axis=-1)  # [B, S, 2F]

        # Project back to features
        y = nn.Dense(
            features=self.features,
            dtype=self.dtype,
            precision=self.precision,
            name="out_proj"
        )(y_cat)

        return y


# =============================================================================
# SpatialFusionConv - 2D state fusion for Direction α
# =============================================================================

class SpatialFusionConv(nn.Module):
    """Spatial-Mamba-style 2D state fusion via multi-dilation depthwise conv.

    Given an SSM's 1D output [B, H_P, W_P, F] already reshaped to a 2D grid,
    apply parallel depthwise 2D convolutions with different dilation rates and
    sum them as a residual against the input. The output is [B, H_P, W_P, F].

    Motivation: An SSM with exponential state decay has an anisotropic 1D
    receptive field. Under naive 1D linearization (raster/hilbert/zigzag) the
    effective 2D neighborhood of a token depends on scan order; Hilbert in
    particular has high variance in 2D-distance-between-sequentially-adjacent
    tokens, which by Jensen's inequality hurts gradient flow through the
    exponential decay. A multi-dilation depthwise 2D conv provides an
    isotropic, direction-balanced short-range receptive field that complements
    the SSM's long-range-but-anisotropic information. The sum acts as a
    residual: the SSM does long-range global mixing, the conv recovers local
    2D structure that the scan order destroyed.

    Depthwise conv kernels are zero-initialized so at step 0 the fusion is a
    pure pass-through (y_out = y_in), ensuring stable warmup. AdaLN gates in
    the outer SSMDiTBlock are also zero-initialized, so the block starts as
    near-identity and the fusion learns its contribution gradually.

    Reference: Xiao, Zhang, Dou, Chen, "Spatial-Mamba: Effective Visual State
    Space Models via Structure-Aware State Fusion," ICLR 2025
    (arxiv:2410.15091).
    """
    features: int
    dilations: Tuple[int, ...] = (1, 2, 3)
    kernel_size: int = 3
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, y_2d):
        """
        Args:
            y_2d: [B, H_P, W_P, F] — SSM output reshaped to 2D row-major grid.
        Returns:
            fused: [B, H_P, W_P, F] — y_2d + sum of K dilated depthwise convs.
        """
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


# =============================================================================
# SSMDiTBlock - Drop-in replacement for DiTBlock
# =============================================================================

class SSMDiTBlock(nn.Module):
    """SSM-based DiT Block that replaces attention with bidirectional S5.

    Maintains the EXACT same interface as DiTBlock:
        __call__(self, x, conditioning, freqs_cis) -> x

    The AdaLN modulation, gating, residual connections, and MLP path
    are IDENTICAL to DiTBlock. Only the attention is replaced with S5.

    Note: freqs_cis (RoPE frequencies) is accepted for interface compatibility
    but not used by the SSM - positional information is implicitly captured
    by the sequential scan along the serialization order (Hilbert/raster/etc).
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

    # --- Direction α additions ---
    # Spatial-Mamba-style 2D state fusion after the SSM scan. When enabled the
    # block reshapes the scan output to a 2D patch grid, applies multi-dilation
    # depthwise convolutions, and sums as a residual. See SpatialFusionConv
    # docstring for full motivation.
    use_2d_fusion: bool = False
    # Scan order the parent model uses to linearize 2D patches. One of
    # 'raster' (row-major, the default), 'hilbert', or 'zigzag'. We need this
    # inside the block so we can un-permute scan-ordered tokens to a 2D
    # row-major grid before the conv, then re-permute back to scan order for
    # the residual path. Defaulting to 'raster' keeps existing callers working.
    scan_order: str = 'raster'

    def setup(self):
        hidden_features = int(self.features * self.mlp_ratio)

        # AdaLN modulation - IDENTICAL to DiTBlock
        self.ada_params_module = AdaLNParams(
            self.features, dtype=self.dtype, precision=self.precision)

        # Layer Norms - IDENTICAL to DiTBlock
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

        # Direction α: optional 2D state fusion after the SSM scan.
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

        # MLP - IDENTICAL to DiTBlock
        self.mlp = nn.Sequential([
            nn.Dense(features=hidden_features, dtype=self.dtype, precision=self.precision),
            nn.gelu,
            nn.Dense(features=self.features, dtype=self.dtype, precision=self.precision)
        ])

    def _apply_2d_fusion(self, ssm_output):
        """Reshape scan-ordered SSM output to 2D row-major grid, apply multi-
        dilation depthwise conv fusion, reshape back to scan order.

        This is a no-op (equivalent to an all-zero residual) at step 0 because
        the conv kernels are zero-initialized. Gradient flow into the conv
        weights from the denoising loss drives them away from zero so the
        fusion gradually learns to contribute short-range 2D structure that
        the SSM's anisotropic 1D scan cannot provide.
        """
        B, S, F = ssm_output.shape
        # Assume square patch grid. Every DiT-family model in this project
        # uses square images with square patch sizes, so H_P == W_P. S is
        # a Python int (static shape) at JIT trace time, so math.isqrt works.
        import math
        H_P = math.isqrt(S)
        W_P = H_P
        assert H_P * W_P == S, (
            f"2D fusion requires a square patch grid; got S={S} which is not a "
            f"perfect square.")

        # Un-permute scan-ordered tokens to row-major 2D grid order.
        # hilbert_indices / zigzag_indices return the forward permutation
        # `scan_idx[h] = k` (row-major index of the h-th scan-order token).
        # We have scan-ordered tokens `ssm_output[:, h, :]` and want them at
        # row-major slot k. That's a SCATTER, or equivalently a GATHER with
        # the inverse permutation `inv[k] = h`. We compute both once at the
        # start of this method — cheap because index arrays are constant
        # at JIT time.
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
            # Gather to row-major: rowmajor_tokens[k] = scan_tokens[scan_fwd[k]]?
            # Wait — scan_fwd[h] = k means the h-th scan-position is at rowmajor k.
            # Equivalently, rowmajor_tokens[k] = scan_tokens[scan_inv[k]].
            # We want row-major, so gather using scan_inv:
            ssm_rm = ssm_output[:, scan_inv, :]
        else:
            ssm_rm = ssm_output

        # Reshape to 2D grid, fuse, reshape back to flat sequence.
        y_2d = ssm_rm.reshape(B, H_P, W_P, F)
        y_fused_2d = self.spatial_fusion(y_2d)
        y_fused_rm = y_fused_2d.reshape(B, S, F)

        # Re-permute to scan order for the residual path.
        if scan_fwd is not None:
            # row-major → scan: scan_tokens[h] = rowmajor_tokens[scan_fwd[h]]
            y_fused = y_fused_rm[:, scan_fwd, :]
        else:
            y_fused = y_fused_rm

        return y_fused

    @nn.compact
    def __call__(self, x, conditioning, freqs_cis):
        """Exact same signature as DiTBlock.__call__"""
        # Get scale/shift/gate parameters - IDENTICAL to DiTBlock
        scale_mlp, shift_mlp, gate_mlp, scale_attn, shift_attn, gate_attn = jnp.split(
            self.ada_params_module(conditioning), 6, axis=-1
        )

        # --- SSM Path (replaces Attention Path) ---
        residual = x
        norm_x = self.norm1(x)
        x_modulated = norm_x * (1 + scale_attn) + shift_attn
        ssm_output = self.ssm(x_modulated)

        # Direction α: 2D state fusion.
        if self.use_2d_fusion:
            ssm_output = self._apply_2d_fusion(ssm_output)

        if self.use_gating:
            x = residual + gate_attn * ssm_output
        else:
            x = residual + ssm_output

        # --- MLP Path --- IDENTICAL to DiTBlock
        residual = x
        norm_x_mlp = self.norm2(x)
        x_mlp_modulated = norm_x_mlp * (1 + scale_mlp) + shift_mlp
        mlp_output = self.mlp(x_mlp_modulated)

        if self.use_gating:
            x = residual + gate_mlp * mlp_output
        else:
            x = residual + mlp_output

        return x


# =============================================================================
# HybridSSMAttentionDiT - The proposed novel architecture
# =============================================================================

class HybridSSMAttentionDiT(nn.Module):
    """Hybrid SSM-Attention Diffusion Transformer.

    Interleaves SSM blocks (for O(n) local processing along Hilbert curve)
    with attention blocks (for O(n^2) global composition) in a configurable
    ratio. Both block types share the same AdaLN conditioning interface.

    Args:
        block_pattern: List of 'ssm' or 'attn' strings defining the block sequence.
            Example: ['ssm','ssm','ssm','attn'] for 3:1 ratio repeated.
            If None, uses ssm_attention_ratio to auto-generate pattern.
        ssm_attention_ratio: Shorthand ratio like '3:1' (3 SSM per 1 attention).
            Only used if block_pattern is None.
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

    # --- Direction α: 2D state fusion ---
    # When True, each SSM block reshapes its scan output to a 2D patch grid
    # and applies multi-dilation depthwise conv fusion before the residual.
    # See SpatialFusionConv docstring for motivation. Zero architectural
    # overhead when False (default).
    use_2d_fusion: bool = False

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
                # Determine scan_order string for the block to use when
                # un-permuting before 2D fusion. Mirrors the use_hilbert/
                # use_zigzag flags on the outer model.
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
        """Exact same signature as SimpleDiT.__call__"""
        B, H, W, C = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0

        H_P = H // self.patch_size
        W_P = W // self.patch_size

        # 1. Patch Embedding (identical to SimpleDiT).
        # Scan order (raster / hilbert / zigzag) is mutually exclusive.
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

        # 2D positional embedding — encodes (row, col) of each patch.
        # The SSM blocks have no implicit positional signal (they ignore RoPE), so we
        # add a 2D sin-cos position embedding to every patch. For any non-raster
        # scan mode, we reorder the row-major position embedding to match the scan
        # sequence so each patch is paired with its TRUE 2D position.
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

        # 2. Conditioning (identical to SimpleDiT)
        t_emb = self.time_embed(temb)
        cond_emb = t_emb
        if textcontext is not None:
            text_emb = self.text_proj(textcontext)
            text_emb_pooled = jnp.mean(text_emb, axis=1)
            cond_emb = cond_emb + text_emb_pooled

        # 3. RoPE frequencies for the attention blocks. In any non-raster scan
        # mode (hilbert / zigzag) the sequence index != 2D position and RoPE
        # would encode wrong relative distances. The additive 2D sin-cos above
        # already supplies the correct 2D position signal, so we override RoPE
        # with identity (cos=1, sin=0) — makes apply_rotary_embedding a no-op
        # without changing any interface.
        freqs_cos, freqs_sin = self.rope(seq_len=num_patches)
        if self.use_hilbert or self.use_zigzag:
            freqs_cos = jnp.ones_like(freqs_cos)
            freqs_sin = jnp.zeros_like(freqs_sin)

        # 4. Hybrid blocks (SSM and attention interleaved)
        for block in self.blocks:
            x_seq = block(x_seq, conditioning=cond_emb, freqs_cis=(freqs_cos, freqs_sin))

        # 5. Final output (identical to SimpleDiT)
        x_out = self.final_norm(x_seq)
        x_out = self.final_proj(x_out)

        # 6. Unpatchify. hilbert_unpatchify is scatter-by-inv_idx so both
        # hilbert and zigzag go through it; raster uses the einops unpatchify.
        if self.use_hilbert or self.use_zigzag:
            if self.learn_sigma:
                x_mean, _x_logvar = jnp.split(x_out, 2, axis=-1)
                return hilbert_unpatchify(x_mean, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
            return hilbert_unpatchify(x_out, hilbert_inv_idx, self.patch_size, H, W, self.output_channels)
        if self.learn_sigma:
            x_mean, _x_logvar = jnp.split(x_out, 2, axis=-1)
            return unpatchify(x_mean, channels=self.output_channels)
        return unpatchify(x_out, channels=self.output_channels)
