"""Evaluation metric tests.

PSNR and SSIM are checked against their closed forms and against the
properties they exist to report (degradation ordering, shape handling).
The Frechet distance itself is checked against closed forms that need no
weights; the end-to-end InceptionV3 path downloads the FID checkpoint and is
network-marked.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flaxdiff.metrics as metrics
from flaxdiff.metrics import (
    EvaluationMetric,
    get_psnr_metric,
    get_ssim_metric,
    psnr,
    ssim,
)
from flaxdiff.metrics.fid import frechet_distance, get_fid_metric


def test_frechet_distance_of_a_distribution_with_itself_is_zero(rng):
    features = np.asarray(jax.random.normal(rng, (256, 16)))
    mu, sigma = features.mean(axis=0), np.cov(features, rowvar=False)
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_of_shifted_gaussians_is_the_squared_mean_gap():
    sigma = np.eye(8)
    mu_a = np.zeros(8)
    mu_b = np.full(8, 0.5)
    assert frechet_distance(mu_a, sigma, mu_b, sigma) == pytest.approx(8 * 0.25, abs=1e-6)


def test_frechet_distance_grows_with_covariance_mismatch():
    mu = np.zeros(8)
    identity = np.eye(8)
    near = frechet_distance(mu, identity, mu, identity * 1.5)
    far = frechet_distance(mu, identity, mu, identity * 4.0)
    assert 0 < near < far


@pytest.mark.network
def test_fid_metric_scores_real_images_better_than_noise(rng):
    metric = get_fid_metric()
    assert isinstance(metric, EvaluationMetric)
    assert metric.name == 'fid' and metric.higher_is_better is False

    key_real, key_noise = jax.random.split(rng)
    real = jax.random.randint(key_real, (8, 64, 64, 3), 0, 256, dtype=jnp.int32).astype(jnp.uint8)
    batch = {'image': real}

    # Generated samples live in [-1, 1]; the same images should score far
    # closer to the batch than unrelated noise does
    matching = metric.function((jnp.asarray(real, jnp.float32) - 127.5) / 127.5, batch)
    unrelated = metric.function(jax.random.normal(key_noise, (8, 64, 64, 3)), batch)
    assert np.isfinite(matching) and np.isfinite(unrelated)
    assert matching < unrelated


############################################################################################################
# PSNR / SSIM
############################################################################################################


def _ramp_image(shape, key):
    """Smooth-ish image in [-1, 1]: SSIM on pure noise is degenerate."""
    ramp = jnp.linspace(-1.0, 1.0, shape[-3])
    base = jnp.broadcast_to(ramp[None, :, None, None], (shape[0], shape[-3], shape[-2], shape[-1]))
    return base + 0.1 * jax.random.normal(key, base.shape)


def _blur(images, width=5):
    """Box blur over H and W, so 'degraded but structured' is a real case."""
    kernel = jnp.ones((width, width)) / (width * width)
    channels = images.shape[-1]
    weights = jnp.zeros((channels, channels, width, width)).at[
        jnp.arange(channels), jnp.arange(channels)
    ].set(kernel)
    return jax.lax.conv_general_dilated(
        images.transpose(0, 3, 1, 2), weights, (1, 1), 'SAME',
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
    ).transpose(0, 2, 3, 1)


def test_psnr_of_identical_images_is_infinite(rng):
    x = _ramp_image((2, 32, 32, 3), rng)
    assert jnp.isinf(psnr(x, x, data_range=2.0))


def test_ssim_of_identical_images_is_one(rng):
    x = _ramp_image((2, 32, 32, 3), rng)
    assert float(ssim(x, x, data_range=2.0)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("offset,data_range", [(0.1, 1.0), (0.1, 2.0), (0.5, 2.0)])
def test_psnr_matches_the_closed_form_for_a_known_error(offset, data_range):
    """A constant offset makes MSE exact, so PSNR is exact too."""
    x = jnp.zeros((2, 8, 8, 3))
    got = psnr(x, x + offset, data_range=data_range)
    expected = 10.0 * np.log10(data_range**2 / offset**2)
    assert float(got) == pytest.approx(expected, rel=1e-5)


def test_psnr_falls_as_noise_grows(rng):
    x = _ramp_image((2, 32, 32, 3), rng)
    key_a, key_b = jax.random.split(rng)
    small = psnr(x, x + 0.05 * jax.random.normal(key_a, x.shape), data_range=2.0)
    large = psnr(x, x + 0.20 * jax.random.normal(key_b, x.shape), data_range=2.0)
    assert float(small) > float(large)


def test_ssim_falls_under_blur_and_noise(rng):
    """The point of SSIM: both degradations score below a perfect match."""
    x = _ramp_image((2, 32, 32, 3), rng)
    perfect = float(ssim(x, x, data_range=2.0))
    blurred = float(ssim(x, _blur(x), data_range=2.0))
    noisy = float(ssim(x, x + 0.2 * jax.random.normal(rng, x.shape), data_range=2.0))
    assert perfect > blurred
    assert perfect > noisy


def test_ssim_falls_as_noise_grows(rng):
    x = _ramp_image((2, 32, 32, 3), rng)
    key_a, key_b = jax.random.split(rng)
    small = ssim(x, x + 0.05 * jax.random.normal(key_a, x.shape), data_range=2.0)
    large = ssim(x, x + 0.20 * jax.random.normal(key_b, x.shape), data_range=2.0)
    assert float(small) > float(large)


def test_ssim_matches_the_closed_form_on_constant_images():
    """Constant images have zero local variance, which collapses SSIM to
    (2 mu_x mu_y + C1) / (mu_x^2 + mu_y^2 + C1)."""
    data_range = 1.0
    x = jnp.full((1, 16, 16, 1), 0.5)
    y = jnp.full((1, 16, 16, 1), 0.7)
    c1 = (0.01 * data_range) ** 2
    expected = (2 * 0.5 * 0.7 + c1) / (0.5**2 + 0.7**2 + c1)
    assert float(ssim(x, y, data_range=data_range)) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("metric_fn", [psnr, ssim], ids=['psnr', 'ssim'])
def test_video_scores_equal_the_flattened_frame_batch(rng, metric_fn):
    """(B, T, H, W, C) must score exactly as the (B*T, H, W, C) frames do."""
    key_x, key_noise = jax.random.split(rng)
    video = _ramp_image((2 * 3, 32, 32, 3), key_x).reshape(2, 3, 32, 32, 3)
    degraded = video + 0.1 * jax.random.normal(key_noise, video.shape)
    frames = video.reshape(6, 32, 32, 3)
    assert float(metric_fn(video, degraded, data_range=2.0)) == pytest.approx(
        float(metric_fn(frames, degraded.reshape(6, 32, 32, 3), data_range=2.0)), rel=1e-5
    )


@pytest.mark.parametrize("metric_fn", [psnr, ssim], ids=['psnr', 'ssim'])
@pytest.mark.parametrize("shape", [(2, 32, 32, 3), (2, 3, 32, 32, 3), (2, 32, 32, 1)])
def test_per_example_scores_have_one_entry_per_frame(rng, metric_fn, shape):
    key_x, key_noise = jax.random.split(rng)
    x = _ramp_image((int(np.prod(shape[:-3])),) + shape[-3:], key_x).reshape(shape)
    y = x + 0.1 * jax.random.normal(key_noise, x.shape)
    scores = metric_fn(x, y, data_range=2.0, per_example=True)
    assert scores.shape == (int(np.prod(shape[:-3])),)
    assert float(jnp.mean(scores)) == pytest.approx(
        float(metric_fn(x, y, data_range=2.0)), rel=1e-5
    )


@pytest.mark.parametrize("metric_fn", [psnr, ssim], ids=['psnr', 'ssim'])
def test_metrics_reject_unbatched_inputs(metric_fn):
    with pytest.raises(AssertionError):
        metric_fn(jnp.zeros((32, 32, 3)), jnp.zeros((32, 32, 3)), data_range=2.0)


def test_psnr_metric_factory_wires_up_the_trainer_contract(rng):
    metric = get_psnr_metric()
    assert isinstance(metric, EvaluationMetric)
    assert metric.name == 'psnr' and metric.higher_is_better is True

    x = _ramp_image((2, 32, 32, 3), rng)
    degraded = x + 0.1
    assert float(metric.function(degraded, {'image': x})) == pytest.approx(
        float(psnr(degraded, x, data_range=2.0)), rel=1e-5
    )


def test_ssim_metric_factory_wires_up_the_trainer_contract(rng):
    metric = get_ssim_metric()
    assert isinstance(metric, EvaluationMetric)
    assert metric.name == 'ssim' and metric.higher_is_better is True

    x = _ramp_image((2, 32, 32, 3), rng)
    degraded = _blur(x)
    assert float(metric.function(degraded, {'image': x})) == pytest.approx(
        float(ssim(degraded, x, data_range=2.0)), rel=1e-5
    )


def test_metrics_package_exports_resolve():
    """The package surface the trainer configures metrics from."""
    assert set(metrics.__all__) == {
        'EvaluationMetric',
        'get_clip_metric',
        'get_clip_score_metric',
        'get_fid_metric',
        'frechet_distance',
        'psnr',
        'get_psnr_metric',
        'ssim',
        'get_ssim_metric',
    }
    for name in metrics.__all__:
        assert getattr(metrics, name) is not None
