"""Evaluation metric tests.

The Frechet distance itself is checked against closed forms that need no
weights; the end-to-end InceptionV3 path downloads the FID checkpoint and is
network-marked.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flaxdiff.metrics.common import EvaluationMetric
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
