"""SimpleAutoEncoder tests.

The tutorial-grade AE has no pretrained weights, so nothing here asserts
reconstruction quality: what matters is that it satisfies the AutoEncoder
contract the samplers and input config depend on (shapes, downscale factor,
latent channels, video flattening, and the latent normalization seam).
"""

import jax
import jax.numpy as jnp
import pytest

from flaxdiff.models.autoencoder import AutoEncoder, SimpleAutoEncoder

# Small enough to stay quick on CPU: 3 stages -> downscale factor 8
DEPTHS = (8, 16, 32)
IMAGE_SIZE = 8


@pytest.fixture(scope="module")
def autoencoder():
    return SimpleAutoEncoder(latent_channels=4, feature_depths=DEPTHS)


@pytest.fixture(scope="module")
def image(autoencoder):
    return jax.random.uniform(
        jax.random.PRNGKey(1), (2, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0, maxval=1.0
    )


def test_it_is_an_autoencoder(autoencoder):
    assert isinstance(autoencoder, AutoEncoder)
    assert autoencoder.name == 'simple_autoencoder'


def test_downscale_factor_and_latent_channels_describe_the_bottleneck(autoencoder):
    """Samplers and the input config size latents off these two properties."""
    assert autoencoder.downscale_factor == 2 ** len(DEPTHS)
    assert autoencoder.latent_channels == 4


def test_encode_produces_the_advertised_latent_shape(autoencoder, image):
    latent = autoencoder.encode(image)
    downscaled = IMAGE_SIZE // autoencoder.downscale_factor
    assert latent.shape == (2, downscaled, downscaled, autoencoder.latent_channels)


def test_decode_restores_the_image_shape(autoencoder, image):
    reconstruction = autoencoder.decode(autoencoder.encode(image))
    assert reconstruction.shape == image.shape


def test_roundtrip_through_call_keeps_the_input_shape(autoencoder, image):
    assert autoencoder(image).shape == image.shape


def test_video_is_encoded_frame_by_frame(autoencoder):
    """5D input keeps its batch and time axes; the base class flattens frames."""
    video = jax.random.uniform(
        jax.random.PRNGKey(2), (2, 3, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0, maxval=1.0
    )
    latent = autoencoder.encode(video)
    downscaled = IMAGE_SIZE // autoencoder.downscale_factor
    assert latent.shape == (2, 3, downscaled, downscaled, autoencoder.latent_channels)
    assert autoencoder.decode(latent).shape == video.shape


def test_video_frames_match_the_same_frames_encoded_as_images(autoencoder):
    """The B*T flattening must not mix frames together."""
    frames = jax.random.uniform(
        jax.random.PRNGKey(3), (6, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0, maxval=1.0
    )
    video = frames.reshape(2, 3, IMAGE_SIZE, IMAGE_SIZE, 3)
    per_frame = autoencoder.encode(frames)
    assert jnp.allclose(
        autoencoder.encode(video), per_frame.reshape(2, 3, *per_frame.shape[1:]), atol=1e-6
    )


def test_latent_normalization_is_inverted_by_decode(autoencoder, image):
    """encode applies (z - shift) * scale and decode must undo exactly it, so
    decode(encode(x)) is the raw decoder applied to the raw latent."""
    normalized = SimpleAutoEncoder(
        latent_channels=4,
        feature_depths=DEPTHS,
        latent_shift=0.3,
        latent_scale=2.5,
        params=autoencoder.params,
    )
    raw_latent = autoencoder.encode(image)  # identity normalization by default
    latent = normalized.encode(image)
    assert jnp.allclose(latent, (raw_latent - 0.3) * 2.5, atol=1e-5)
    assert jnp.allclose(normalized.decode(latent), autoencoder.decode(raw_latent), atol=1e-5)


def test_normalized_latents_can_be_whitened(image):
    """Point of the seam: pass dataset statistics and the latents come out
    centred and unit-variance for the diffusion model."""
    plain = SimpleAutoEncoder(latent_channels=4, feature_depths=DEPTHS)
    latents = plain.encode(image)
    whitened = SimpleAutoEncoder(
        latent_channels=4,
        feature_depths=DEPTHS,
        latent_shift=float(jnp.mean(latents)),
        latent_scale=1.0 / float(jnp.std(latents)),
        params=plain.params,
    ).encode(image)
    assert abs(float(jnp.mean(whitened))) < 1e-4
    assert float(jnp.std(whitened)) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("depths", [(8,), (8, 16), (8, 16, 32)])
def test_downscale_factor_follows_the_stage_count(depths):
    autoencoder = SimpleAutoEncoder(latent_channels=2, feature_depths=depths)
    size = autoencoder.downscale_factor
    assert size == 2 ** len(depths)
    image = jnp.zeros((1, size, size, 3))
    assert autoencoder.encode(image).shape == (1, 1, 1, 2)
    assert autoencoder.decode(autoencoder.encode(image)).shape == image.shape


def test_group_norm_survives_depths_not_divisible_by_norm_groups():
    """GroupNorm needs a divisor of the channel count; the AE picks one."""
    autoencoder = SimpleAutoEncoder(latent_channels=2, feature_depths=(12,), norm_groups=8)
    image = jnp.zeros((1, 2, 2, 3))
    assert autoencoder(image).shape == image.shape


def test_params_can_be_reused_across_instances(autoencoder, image):
    """Checkpointed weights load by construction, no re-init."""
    reloaded = SimpleAutoEncoder(
        latent_channels=4, feature_depths=DEPTHS, params=autoencoder.params
    )
    assert jnp.allclose(reloaded.encode(image), autoencoder.encode(image), atol=1e-6)


def test_fresh_instances_get_different_random_weights(image):
    """No pretrained weights: two seeds must not agree."""
    a = SimpleAutoEncoder(latent_channels=4, feature_depths=DEPTHS, key=jax.random.PRNGKey(0))
    b = SimpleAutoEncoder(latent_channels=4, feature_depths=DEPTHS, key=jax.random.PRNGKey(1))
    assert not jnp.allclose(a.encode(image), b.encode(image), atol=1e-4)


def test_serialize_records_the_config(autoencoder):
    serialized = autoencoder.serialize()
    assert serialized['name'] == 'simple_autoencoder'
    assert serialized['latent_channels'] == 4
    assert serialized['feature_depths'] == list(DEPTHS)
    assert serialized['latent_shift'] == 0.0 and serialized['latent_scale'] == 1.0
