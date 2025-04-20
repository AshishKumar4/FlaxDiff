import jax
import jax.numpy as jnp
from flax import linen as nn
from .autoencoder import AutoEncoder

"""
This module contains an Autoencoder implementation which uses the Stable Diffusion VAE model from the HuggingFace Diffusers library.
The actual model was not trained by me, but was taken from the HuggingFace model hub.
I have only implemented the wrapper around the diffusers pipeline to make it compatible with our library
All credits for the model go to the developers of Stable Diffusion VAE and all credits for the pipeline go to the developers of the Diffusers library.
"""

class StableDiffusionVAE(AutoEncoder):
    def __init__(self, modelname = "CompVis/stable-diffusion-v1-4", revision="bf16", dtype=jnp.bfloat16):
        
        from diffusers.models.vae_flax import FlaxEncoder, FlaxDecoder
        from diffusers import FlaxStableDiffusionPipeline, FlaxAutoencoderKL
        
        vae, params = FlaxAutoencoderKL.from_pretrained(
            modelname,
            # revision=revision,
            dtype=dtype,
        )
        
        self.modelname = modelname
        self.revision = revision
        self.dtype = dtype
        
        enc = FlaxEncoder(
            in_channels=vae.config.in_channels,
            out_channels=vae.config.latent_channels,
            down_block_types=vae.config.down_block_types,
            block_out_channels=vae.config.block_out_channels,
            layers_per_block=vae.config.layers_per_block,
            act_fn=vae.config.act_fn,
            norm_num_groups=vae.config.norm_num_groups,
            double_z=True,
            dtype=vae.dtype,
        )
        
        dec = FlaxDecoder(
            in_channels=vae.config.latent_channels,
            out_channels=vae.config.out_channels,
            up_block_types=vae.config.up_block_types,
            block_out_channels=vae.config.block_out_channels,
            layers_per_block=vae.config.layers_per_block,
            norm_num_groups=vae.config.norm_num_groups,
            act_fn=vae.config.act_fn,
            dtype=vae.dtype,
        )
        
        quant_conv = nn.Conv(
            2 * vae.config.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=vae.dtype,
        )

        post_quant_conv = nn.Conv(
            vae.config.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=vae.dtype,
        )
        
        scaling_factor = vae.scaling_factor
        print(f"Scaling factor: {scaling_factor}")
        
        def encode_single_frame(images, rngkey: jax.random.PRNGKey = None):
            latents = enc.apply({"params": params['encoder']}, images, deterministic=True)
            latents = quant_conv.apply({"params": params['quant_conv']}, latents)
            if rngkey is not None:
                mean, log_std = jnp.split(latents, 2, axis=-1)
                log_std = jnp.clip(log_std, -30, 20)
                std = jnp.exp(0.5 * log_std)
                latents = mean + std * jax.random.normal(rngkey, mean.shape, dtype=mean.dtype)
            else:
                latents, _ = jnp.split(latents, 2, axis=-1)
            latents *= scaling_factor
            return latents
        
        def decode_single_frame(latents):
            latents = (1.0 / scaling_factor) * latents
            latents = post_quant_conv.apply({"params": params['post_quant_conv']}, latents)
            return dec.apply({"params": params['decoder']}, latents)
        
        self.encode_single_frame = jax.jit(encode_single_frame)
        self.decode_single_frame = jax.jit(decode_single_frame)
        
        # Calculate downscale factor by passing a dummy input through the encoder
        print("Calculating downscale factor...")
        dummy_input = jnp.ones((1, 128, 128, 3), dtype=dtype)
        dummy_latents = self.encode_single_frame(dummy_input)
        _, h, w, c = dummy_latents.shape
        _, H, W, C = dummy_input.shape
        self.__downscale_factor__ = H // h
        self.__latent_channels__ = c
        print(f"Downscale factor: {self.__downscale_factor__}")
        print(f"Latent channels: {self.__latent_channels__}")

    def __encode__(self, images, key: jax.random.PRNGKey = None, **kwargs):
        """Encode a batch of images to latent representations.
        
        Implements the abstract method from the parent class.
        
        Args:
            images: Image tensor of shape [B, H, W, C]
            key: Optional random key for stochastic encoding
            **kwargs: Additional arguments (unused)
            
        Returns:
            Latent representations of shape [B, h, w, c]
        """
        return self.encode_single_frame(images, key)
    
    def __decode__(self, latents, **kwargs):
        """Decode latent representations to images.
        
        Implements the abstract method from the parent class.
        
        Args:
            latents: Latent tensor of shape [B, h, w, c]
            **kwargs: Additional arguments (unused)
            
        Returns:
            Decoded images of shape [B, H, W, C]
        """
        return self.decode_single_frame(latents)

    @property
    def downscale_factor(self) -> int:
        """Returns the downscale factor for the encoder."""
        return self.__downscale_factor__
    
    @property
    def latent_channels(self) -> int:
        """Returns the number of channels in the latent space."""
        return self.__latent_channels__
    
    @property
    def name(self) -> str:
        """Get the name of the autoencoder model."""
        return "stable_diffusion"
    
    def serialize(self):
        """Serialize the model to a dictionary format."""
        return {
            "modelname": self.modelname,
            "revision": self.revision,
            "dtype": str(self.dtype),
        }
    