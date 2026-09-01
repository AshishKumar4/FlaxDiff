"""PSNR in pure jax, batched over images and video.

Both functions accept (B, H, W, C) or (B, T, H, W, C): video is flattened
to (B*T, H, W, C) and every frame scores independently.
"""

import jax.numpy as jnp

from .common import EvaluationMetric


def _as_frame_batch(images: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
    """Return (frames, was_video); frames always has shape (N, H, W, C)."""
    images = jnp.asarray(images)
    if images.ndim == 5:
        B, T, H, W, C = images.shape
        return images.reshape((B * T, H, W, C)), True
    assert images.ndim == 4, (
        f"expected (B, H, W, C) or (B, T, H, W, C), got shape {images.shape}"
    )
    return images, False


def psnr(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    data_range: float,
    per_example: bool = False,
) -> jnp.ndarray:
    """Peak signal-to-reference ratio, PSNR = 10 * log10(data_range^2 / MSE).

    Args:
        predictions: (B, H, W, C) images or (B, T, H, W, C) video.
        targets: same shape as predictions.
        data_range: dynamic range of the signal (e.g. 2.0 for [-1, 1] inputs,
            255 for uint8), exactly as in skimage's PSNR.
        per_example: return the per-frame scores instead of the mean.

    Returns:
        Scalar score, or (B,) per-frame scores when per_example is set.
        Identical inputs give +inf.
    """
    pred, _ = _as_frame_batch(predictions)
    targ, _ = _as_frame_batch(targets)
    mse = jnp.mean((pred - targ) ** 2, axis=(1, 2, 3))
    scores = 10.0 * jnp.log10(data_range**2 / mse)
    return scores if per_example else jnp.mean(scores)


def get_psnr_metric(
    data_range: float = 2.0,
    per_example: bool = False,
) -> EvaluationMetric:
    """Mean PSNR in dB between generated and reference frames, higher is better.

    data_range must match the convention the trainer feeds the metric: 2.0
    for the library's [-1, 1] samples, 255 for uint8 batches.
    """
    def psnr_metric(generated: jnp.ndarray, batch):
        return psnr(generated, jnp.asarray(batch["image"], dtype=jnp.float32), data_range, per_example)

    return EvaluationMetric(
        function=psnr_metric,
        name="psnr",
        higher_is_better=True,
    )
