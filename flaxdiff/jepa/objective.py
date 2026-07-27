"""The I-JEPA / V-JEPA objective.

Predict the representation of masked target blocks from the representation of
the visible context, in latent space. Three moving parts:

  - the context encoder sees only the context tokens and is trained;
  - the target encoder sees the whole image and is not: it is the EMA of the
    context encoder, so it lives in the trainer's ema_params rather than in a
    parameter subtree of its own, and its branch is stop_gradient'd;
  - the predictor maps context embeddings plus target positions to the target
    representations.

Targets are layer-normalized (no learned affine) before the L2 loss, which is
what keeps the scale of the prediction problem fixed as the encoder drifts.
The paper's L2 is used rather than the reference implementation's smooth-L1:
the LN already bounds the target scale, so there are no outliers for smooth-L1
to protect against, and L2 keeps the loss directly comparable to the paper.

The characteristic failure is silent: both encoders quietly agree on a
constant and the loss goes to zero. representation_health is reported on every
step so that collapse is visible in the training curves rather than at the end
of a probe run.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from ..trainer.objectives import Objective, EMASpec
from .masking import MultiBlockMask

CONTEXT_ENCODER = "context_encoder"
PREDICTOR = "predictor"


def representation_health(z) -> Dict[str, jax.Array]:
    """Collapse telemetry for pooled embeddings [B, D].

    repr_std is the per-dimension standard deviation across the batch: it goes
    to zero exactly when the encoder stops distinguishing inputs. repr_cov_offdiag
    is the RMS magnitude of the off-diagonal covariance, which rises when the
    dimensions become redundant (dimensional collapse) even while repr_std holds.
    """
    batch_size, dim = z.shape
    centered = z - jnp.mean(z, axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(batch_size - 1, 1)
    off_diagonal = cov * (1.0 - jnp.eye(dim, dtype=cov.dtype))
    return {
        "repr_std": jnp.mean(jnp.std(z, axis=0)),
        "repr_cov_offdiag": jnp.sqrt(jnp.sum(off_diagonal ** 2) / max(dim * (dim - 1), 1)),
    }


def normalize_targets(x, epsilon: float = 1e-6):
    """Feature-wise layer norm with no learned affine.

    Applied to the target encoder's output so the prediction problem keeps a
    fixed scale as the encoder drifts, and so shrinking the representation is
    not a way to lower the loss.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon)


class JepaObjective(Objective):
    """Joint-embedding prediction over images (B,H,W,C) or video (B,T,H,W,C)."""

    tag = "jepa"

    def __init__(
        self,
        encoder,
        predictor,
        mask: MultiBlockMask,
        sample_data_key: str,
        sample_data_shape: Tuple[int, ...],
        momentum: Tuple[float, float] = (0.996, 1.0),
        momentum_steps: int = 100_000,
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.mask = mask
        self.sample_data_key = sample_data_key
        self.sample_data_shape = tuple(sample_data_shape)
        self.is_video = len(self.sample_data_shape) == 4
        self.ema = EMASpec(
            decay=optax.linear_schedule(momentum[0], momentum[1], momentum_steps),
            path=("params", CONTEXT_ENCODER),
        )

    def init_params(self, rng):
        encoder_rng, predictor_rng = jax.random.split(rng)
        sample = jnp.ones((1, *self.sample_data_shape))
        context_idx = jnp.arange(self.mask.num_context, dtype=jnp.int32)[None]
        target_idx = jnp.arange(self.mask.block_area, dtype=jnp.int32)[None]

        encoder = self.encoder.init(encoder_rng, sample, context_idx)
        context = self.encoder.apply(encoder, sample, context_idx)
        predictor = self.predictor.init(predictor_rng, context, context_idx, target_idx)
        return {"params": {CONTEXT_ENCODER: encoder["params"],
                           PREDICTOR: predictor["params"]}}

    def encode(self, encoder_params, data, token_idx=None, train=False, rngs=None):
        return self.encoder.apply({"params": encoder_params}, data, token_idx,
                                  train=train, rngs=rngs)

    def loss(self, params, ema_params, batch, rng, step):
        data = (jnp.asarray(batch[self.sample_data_key], dtype=jnp.float32) - 127.5) / 127.5
        batch_size = data.shape[0]
        mask_rng, dropout_rng = jax.random.split(rng)
        context_idx, target_idx = self.mask.sample(mask_rng, batch_size)
        num_targets = self.mask.num_targets

        # Target branch: the whole view through the EMA encoder, no gradient
        full = normalize_targets(self.encode(ema_params["params"][CONTEXT_ENCODER], data))
        # [B, (T,) S, F] -> [B, M, (T,) n_tgt, F]
        frame_axis = (1,) if self.is_video else ()
        gather_idx = target_idx.reshape(batch_size, num_targets, *frame_axis, -1, 1)
        targets = jax.lax.stop_gradient(
            jnp.take_along_axis(full[:, None], gather_idx, axis=-2))

        context = self.encode(
            params["params"][CONTEXT_ENCODER], data, context_idx,
            train=True, rngs={"dropout": dropout_rng})

        # Each target block is predicted from the same context: fold the block
        # axis into the batch so one predictor call covers all M of them
        repeated = jnp.repeat(context, num_targets, axis=0)
        predictions = self.predictor.apply(
            {"params": params["params"][PREDICTOR]},
            repeated,
            jnp.repeat(context_idx, num_targets, axis=0),
            target_idx.reshape(batch_size * num_targets, -1),
            train=True, rngs={"dropout": dropout_rng},
        ).reshape(targets.shape)

        loss = jnp.mean((predictions - targets) ** 2)
        pooled = jnp.mean(full, axis=tuple(range(1, full.ndim - 1)))
        return loss, representation_health(pooled)

    def make_validation_step(self, **_):
        """Frozen target-encoder embeddings, which the probes score."""
        def embed(val_state, batch):
            data = (jnp.asarray(batch[self.sample_data_key], dtype=jnp.float32) - 127.5) / 127.5
            features = self.encode(
                val_state.ema_params["params"][CONTEXT_ENCODER], data)
            return jnp.mean(features, axis=tuple(range(1, features.ndim - 1)))

        return embed
