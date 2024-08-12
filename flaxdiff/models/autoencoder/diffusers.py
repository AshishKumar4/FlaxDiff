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
        from diffusers import FlaxStableDiffusionPipeline
        
        pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
            modelname,
            revision=revision,
            dtype=dtype,
        )
        
        vae = pipeline.vae
        
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
        
        self.enc = enc
        self.dec = dec
        self.post_quant_conv = post_quant_conv
        self.quant_conv = quant_conv
        self.params = params
        self.scaling_factor = vae.scaling_factor
        
    def encode(self, images, rngkey: jax.random.PRNGKey = None):
        latents = self.enc.apply({"params": self.params["vae"]['encoder']}, images, deterministic=True)
        latents = self.quant_conv.apply({"params": self.params["vae"]['quant_conv']}, latents)
        if rngkey is not None:
            mean, log_std = jnp.split(latents, 2, axis=-1)
            log_std = jnp.clip(log_std, -30, 20)
            std = jnp.exp(0.5 * log_std)
            latents = mean + std * jax.random.normal(rngkey, mean.shape, dtype=mean.dtype)
            # print("Sampled")
        else:
            # return the mean
            latents, _ = jnp.split(latents, 2, axis=-1)
        latents *= self.scaling_factor
        return latents
    
    def decode(self, latents):
        latents = (1.0 / self.scaling_factor) * latents
        latents = self.post_quant_conv.apply({"params": self.params["vae"]['post_quant_conv']}, latents)
        return self.dec.apply({"params": self.params["vae"]['decoder']}, latents)
