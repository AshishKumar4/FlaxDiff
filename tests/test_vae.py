"""Vendored Stable Diffusion VAE tests.

Marked as network tests: they download the pretrained weights from the
HuggingFace Hub on first run. Excluded in CI (-m "not network").
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def vae():
    from flaxdiff.models.autoencoder.diffusers import StableDiffusionVAE
    return StableDiffusionVAE(dtype=jnp.float32)


def test_vae_shapes(vae):
    assert vae.downscale_factor == 8
    assert vae.latent_channels == 4


def test_vae_roundtrip_reconstructs(vae, rng):
    # A smooth image should survive the encode/decode roundtrip well
    ramp = jnp.linspace(-0.8, 0.8, 64)
    x = jnp.broadcast_to(ramp[None, :, None, None], (1, 64, 64, 3)).transpose(0, 2, 1, 3)
    rec = vae.decode(vae.encode(x))
    mse = float(jnp.mean((rec - x) ** 2))
    psnr = 10 * np.log10(4.0 / mse)
    assert psnr > 20, f"reconstruction too poor: {psnr:.1f}dB"
