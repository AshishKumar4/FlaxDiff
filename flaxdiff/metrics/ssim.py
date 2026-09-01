"""SSIM in pure jax (Wang et al. 2004), batched over images and video.

Standard parameters: 11x11 gaussian window, sigma 1.5, means taken over
channels after per-channel SSIM. No scipy/skimage dependency.
"""

import jax
import jax.numpy as jnp

from .common import EvaluationMetric
from .psnr import _as_frame_batch

# Standard SSIM constants for the 11x11/sigma-1.5 gaussian window
_K1 = 0.01
_K2 = 0.03
_WINDOW_SIZE = 11
_SIGMA = 1.5


def _gaussian_window_1d(size: int, sigma: float) -> jnp.ndarray:
    coords = jnp.arange(size, dtype=jnp.float32) - (size - 1) / 2.0
    kernel = jnp.exp(-(coords**2) / (2.0 * sigma**2))
    return kernel / jnp.sum(kernel)


def _ssim_single_channel(
    x: jnp.ndarray, y: jnp.ndarray, data_range: float
) -> jnp.ndarray:
    """SSIM map between two single-channel images of shape (H, W).

    The window is applied as a 2D separable convolution with VALID padding,
    exactly the reference implementation's filtering; the score is the mean
    of the SSIM map.
    """
    window_1d = _gaussian_window_1d(_WINDOW_SIZE, _SIGMA)
    window_2d = jnp.outer(window_1d, window_1d)
    window_2d = window_2d[None, None]  # (1, 1, K, K) for conv_general_dilated

    def filt(img: jnp.ndarray) -> jnp.ndarray:
        img = img[None, None]  # (1, 1, H, W)
        out = jax.lax.conv_general_dilated(
            img, window_2d,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return out[0, 0]  # (H', W')

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sigma_x2 = filt(x**2) - mu_x2
    sigma_y2 = filt(y**2) - mu_y2
    sigma_xy = filt(x * y) - mu_xy

    c1 = (_K1 * data_range) ** 2
    c2 = (_K2 * data_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return jnp.mean(numerator / denominator)


def ssim(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    data_range: float,
    per_example: bool = False,
) -> jnp.ndarray:
    """Structural similarity (Wang et al. 2004) with standard parameters.

    SSIM is computed per channel with an 11x11 gaussian window (sigma 1.5)
    and averaged over channels and frames.

    Args:
        predictions: (B, H, W, C) images or (B, T, H, W, C) video.
        targets: same shape as predictions.
        data_range: dynamic range of the signal (2.0 for [-1, 1] inputs).
        per_example: return the per-frame scores instead of the mean.

    Returns:
        Scalar score, or (B,) per-frame scores when per_example is set.
        Identical inputs give 1.0.
    """
    pred, _ = _as_frame_batch(predictions)
    targ, _ = _as_frame_batch(targets)
    # vmap over frames then channels: each (H, W) plane is scored independently
    scores = jax.vmap(
        jax.vmap(_ssim_single_channel, in_axes=(2, 2, None)),
        in_axes=(0, 0, None),
    )(pred, targ, data_range)
    return scores if per_example else jnp.mean(scores)


def get_ssim_metric(
    data_range: float = 2.0,
    per_example: bool = False,
) -> EvaluationMetric:
    """Mean SSIM between generated and reference frames, higher is better.

    data_range must match the convention the trainer feeds the metric: 2.0
    for the library's [-1, 1] samples, 255 for uint8 batches.
    """
    def ssim_metric(generated: jnp.ndarray, batch):
        return ssim(generated, jnp.asarray(batch["image"], dtype=jnp.float32), data_range, per_example)

    return EvaluationMetric(
        function=ssim_metric,
        name="ssim",
        higher_is_better=True,
    )
