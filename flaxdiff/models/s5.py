"""
S5 state-space layers (diagonal SSM with associative_scan, HiPPO init) and the
Spatial-Mamba style 2D state fusion conv. Extracted from ssm_dit.py so the
shared DiT block can use them without importing the full hybrid model.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
from flax.typing import Dtype, PrecisionLike

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
        assert F == self.features, f"S5Layer built for {self.features} features, got {F}"

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
