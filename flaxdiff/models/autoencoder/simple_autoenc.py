from typing import Callable, Optional, Sequence

import jax
import flax.linen as nn
from jax import numpy as jnp
from flax.typing import Dtype, PrecisionLike

from .autoencoder import AutoEncoder


def _group_count(channels: int, requested: int) -> int:
    """Largest group count <= requested that divides `channels` (GroupNorm needs it)."""
    groups = min(requested, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SimpleEncoder(nn.Module):
    """Stride-2 conv stack: [B, H, W, C] -> [B, H/f, W/f, latent_channels]."""

    latent_channels: int
    feature_depths: Sequence[int]
    activation: Callable = jax.nn.silu
    norm_groups: int = 8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, features in enumerate(self.feature_depths):
            x = nn.Conv(
                features=features,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                dtype=self.dtype,
                precision=self.precision,
                name=f"down_{i}",
            )(x)
            if self.norm_groups > 0:
                x = nn.GroupNorm(_group_count(features, self.norm_groups), name=f"down_norm_{i}")(x)
            x = self.activation(x)
        return nn.Conv(
            features=self.latent_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            dtype=self.dtype,
            precision=self.precision,
            name="to_latent",
        )(x)


class SimpleDecoder(nn.Module):
    """Nearest-upsample + conv stack mirroring the encoder: latents -> images."""

    out_channels: int
    feature_depths: Sequence[int]
    activation: Callable = jax.nn.silu
    norm_groups: int = 8
    dtype: Optional[Dtype] = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        for i, features in enumerate(self.feature_depths):
            B, H, W, C = z.shape
            z = jax.image.resize(z, (B, H * 2, W * 2, C), method="nearest")
            z = nn.Conv(
                features=features,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                dtype=self.dtype,
                precision=self.precision,
                name=f"up_{i}",
            )(z)
            if self.norm_groups > 0:
                z = nn.GroupNorm(_group_count(features, self.norm_groups), name=f"up_norm_{i}")(z)
            z = self.activation(z)
        return nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            dtype=self.dtype,
            precision=self.precision,
            name="to_image",
        )(z)


class SimpleAutoEncoder(AutoEncoder):
    """Tutorial-grade convolutional autoencoder, no pretrained weights.

    The encoder halves the resolution once per entry of `feature_depths`
    (stride-2 3x3 conv + GroupNorm + SiLU) and projects to `latent_channels`;
    the decoder mirrors it with nearest-neighbour upsampling. So
    `downscale_factor == 2 ** len(feature_depths)` and `latent_channels` is the
    bottleneck width, the two properties the samplers and input config read off
    an autoencoder (same contract as StableDiffusionVAE).

    Like StableDiffusionVAE this owns its parameters, so `encode`/`decode` work
    on a bare instance; unlike it, the weights start random. Train them (or pass
    a trained tree as `params`) before the reconstructions mean anything. The
    latent is deterministic: there is no KL bottleneck, so the encode `key` is
    accepted and ignored. Video comes free from the AutoEncoder base class,
    which flattens [B, T, H, W, C] to frames.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        feature_depths: Sequence[int] = (32, 64, 128),
        out_channels: int = 3,
        activation: Callable = jax.nn.silu,
        norm_groups: int = 8,
        dtype: Optional[Dtype] = jnp.float32,
        precision: PrecisionLike = None,
        latent_shift: float = 0.0,
        latent_scale: float = 1.0,
        params=None,
        key: jax.Array = None,
    ):
        self.latent_shift = latent_shift
        self.latent_scale = latent_scale
        self.feature_depths = tuple(feature_depths)
        self.out_channels = out_channels
        self.dtype = dtype

        self.encoder = SimpleEncoder(
            latent_channels=latent_channels,
            feature_depths=self.feature_depths,
            activation=activation,
            norm_groups=norm_groups,
            dtype=dtype,
            precision=precision,
        )
        self.decoder = SimpleDecoder(
            out_channels=out_channels,
            feature_depths=self.feature_depths[::-1],
            activation=activation,
            norm_groups=norm_groups,
            dtype=dtype,
            precision=precision,
        )

        # One conv stage per feature depth, each halving the resolution
        self.__downscale_factor__ = 2 ** len(self.feature_depths)
        self.__latent_channels__ = latent_channels

        if params is None:
            params = self.init_params(key if key is not None else jax.random.PRNGKey(0))
        self.params = params

        def encode_single_frame(images):
            return self.encoder.apply({"params": self.params["encoder"]}, images)

        def decode_single_frame(latents):
            return self.decoder.apply({"params": self.params["decoder"]}, latents)

        self.encode_single_frame = jax.jit(encode_single_frame)
        self.decode_single_frame = jax.jit(decode_single_frame)

    def init_params(self, key: jax.Array) -> dict:
        """Freshly initialize encoder and decoder parameters.

        Convolutional, so the init resolution is irrelevant as long as it
        survives every downscale stage; the smallest such image is used.
        """
        size = self.__downscale_factor__
        encode_key, decode_key = jax.random.split(key)
        image = jnp.zeros((1, size, size, self.out_channels), dtype=self.dtype)
        encoder_params = self.encoder.init(encode_key, image)["params"]
        latent = self.encoder.apply({"params": encoder_params}, image)
        decoder_params = self.decoder.init(decode_key, latent)["params"]
        return {"encoder": encoder_params, "decoder": decoder_params}

    def __encode__(self, images: jnp.ndarray, key: jax.Array = None, **kwargs) -> jnp.ndarray:
        """Encode images [B, H, W, C] to latents [B, H/f, W/f, latent_channels].

        `key` is part of the AutoEncoder contract but unused: this encoder is
        deterministic.
        """
        return self.encode_single_frame(images)

    def __decode__(self, latents: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Decode latents [B, h, w, latent_channels] to images [B, h*f, w*f, C]."""
        return self.decode_single_frame(latents)

    @property
    def downscale_factor(self) -> int:
        """Spatial downscale factor between image and latent space."""
        return self.__downscale_factor__

    @property
    def latent_channels(self) -> int:
        """Number of channels in the latent space."""
        return self.__latent_channels__

    @property
    def name(self) -> str:
        return "simple_autoencoder"

    def serialize(self) -> dict:
        """Config only, matching StableDiffusionVAE: weights live in a checkpoint."""
        return {
            "name": self.name,
            "latent_channels": self.__latent_channels__,
            "feature_depths": list(self.feature_depths),
            "out_channels": self.out_channels,
            "dtype": str(self.dtype),
            "latent_shift": self.latent_shift,
            "latent_scale": self.latent_scale,
        }
