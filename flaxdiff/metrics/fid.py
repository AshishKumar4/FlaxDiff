from .common import EvaluationMetric
import jax
import jax.numpy as jnp
import numpy as np


# The FID InceptionV3 is ~90MB of weights; one copy is enough for every metric
# built from this module.
_inception_cache: dict = {}


def _get_inception():
    """Cached (model, params) for the pool3 feature extractor."""
    if 'inception' not in _inception_cache:
        from .inception import InceptionV3
        print("[metrics] Loading InceptionV3 FID weights (cached for reuse)...")
        model = InceptionV3(pretrained=True)
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
        _inception_cache['inception'] = (model, params)
    return _inception_cache['inception']


def frechet_distance(mu_a, sigma_a, mu_b, sigma_b, eps=1e-6) -> float:
    """Frechet distance between two multivariate gaussians.

    Runs on the host through scipy: the matrix square root of the covariance
    product has no jax equivalent, and FID is computed once per validation
    batch so the transfer is irrelevant.
    """
    from scipy import linalg

    mu_a, mu_b = np.atleast_1d(mu_a), np.atleast_1d(mu_b)
    sigma_a, sigma_b = np.atleast_2d(sigma_a), np.atleast_2d(sigma_b)

    covmean, _ = linalg.sqrtm(sigma_a.dot(sigma_b), disp=False)
    if not np.isfinite(covmean).all():
        # Singular product covariance, nudge the diagonal as in the reference
        # implementations rather than returning a nan
        offset = np.eye(sigma_a.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_a - mu_b
    return float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2 * np.trace(covmean))


def _gaussian_stats(activations):
    activations = np.asarray(activations, dtype=np.float64)
    return activations.mean(axis=0), np.cov(activations, rowvar=False)


def get_fid_metric():
    """FID between the generated batch and the real batch, lower is better.

    Per-batch FID is noisy at typical validation batch sizes and is only
    meaningful as a relative trend across checkpoints, not as a headline number
    comparable to published FID-50k.
    """
    model, params = _get_inception()

    @jax.jit
    def activations(images):
        # Inception wants [-1, 1] at 299x299; pool3 output is [B, 1, 1, 2048]
        resized = jax.image.resize(images, (images.shape[0], 299, 299, 3), method='bilinear')
        features = model.apply(params, resized, train=False)
        return features.reshape(features.shape[0], -1)

    def fid_metric(generated: jnp.ndarray, batch):
        original = (jnp.asarray(batch['image'], dtype=jnp.float32) - 127.5) / 127.5
        mu_gen, sigma_gen = _gaussian_stats(activations(generated))
        mu_real, sigma_real = _gaussian_stats(activations(original))
        return frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)

    return EvaluationMetric(function=fid_metric, name='fid')
