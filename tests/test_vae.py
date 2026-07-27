"""Vendored Stable Diffusion VAE tests, plus the latent normalization seam.

Everything touching the pretrained weights is network-marked: they download
from the HuggingFace Hub on first run. Excluded in CI (-m "not network").
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flaxdiff.models.autoencoder import AutoEncoder


class IdentityAutoEncoder(AutoEncoder):
    """Latents are the input, so only the normalization seam is under test."""
    def __encode__(self, x, **kwargs):
        return x

    def __decode__(self, z, **kwargs):
        return z

    def serialize(self):
        return {}


def test_latent_normalization_defaults_to_the_identity(rng):
    autoencoder = IdentityAutoEncoder()
    x = jax.random.normal(rng, (2, 8, 8, 4))
    assert jnp.allclose(autoencoder.encode(x), x)
    assert jnp.allclose(autoencoder.decode(x), x)


@pytest.mark.parametrize("shape", [(2, 8, 8, 4), (2, 3, 8, 8, 4)])
def test_latent_normalization_shifts_and_scales_roundtrip(rng, shape):
    """SD3-style shift+scale: latents come out centred and rescaled, and
    decoding inverts it exactly, for images and for video."""
    autoencoder = IdentityAutoEncoder(latent_shift=0.25, latent_scale=4.0)
    x = jax.random.normal(rng, shape)
    latent = autoencoder.encode(x)
    assert jnp.allclose(latent, (x - 0.25) * 4.0, atol=1e-6)
    assert jnp.allclose(autoencoder.decode(latent), x, atol=1e-5)


def test_latent_normalization_whitens_a_known_distribution(rng):
    """The point of per-dataset stats: the diffusion model sees zero mean and
    unit variance instead of whatever the encoder happens to produce."""
    x = 3.0 + 5.0 * jax.random.normal(rng, (4096, 1, 1, 4))
    autoencoder = IdentityAutoEncoder(latent_shift=float(jnp.mean(x)), latent_scale=1.0 / float(jnp.std(x)))
    latent = autoencoder.encode(x)
    assert abs(float(jnp.mean(latent))) < 1e-4
    assert abs(float(jnp.std(latent)) - 1.0) < 1e-4


@pytest.fixture(scope="module")
def vae():
    from flaxdiff.models.autoencoder.diffusers import StableDiffusionVAE
    return StableDiffusionVAE(dtype=jnp.float32)


@pytest.mark.network
def test_vae_shapes(vae):
    assert vae.downscale_factor == 8
    assert vae.latent_channels == 4


@pytest.mark.network
def test_vae_uses_the_latent_normalization_seam(vae):
    """The SD scaling factor rides on the shared seam, so there is one
    normalization path a caller can override with dataset statistics."""
    assert vae.latent_scale == pytest.approx(0.18215)
    assert vae.latent_shift == 0.0


@pytest.mark.network
def test_vae_roundtrip_reconstructs(vae, rng):
    # A smooth image should survive the encode/decode roundtrip well
    ramp = jnp.linspace(-0.8, 0.8, 64)
    x = jnp.broadcast_to(ramp[None, :, None, None], (1, 64, 64, 3)).transpose(0, 2, 1, 3)
    rec = vae.decode(vae.encode(x))
    mse = float(jnp.mean((rec - x) ** 2))
    psnr = 10 * np.log10(4.0 / mse)
    assert psnr > 20, f"reconstruction too poor: {psnr:.1f}dB"
