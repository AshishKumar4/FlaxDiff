from flax import linen as nn
import jax
import jax.numpy as jnp

class BCHWModelWrapper(nn.Module):
    model: nn.Module

    @nn.compact
    def __call__(self, x, temb, textcontext):
        # Reshape the input to BCHW format from BHWC
        x = jnp.transpose(x, (0, 3, 1, 2))
        # Pass the input through the UNet model
        out = self.model(
            sample=x,
            timesteps=temb,
            encoder_hidden_states=textcontext,
        )
        # Reshape the output back to BHWC format
        out = jnp.transpose(out.sample, (0, 2, 3, 1))
        return out
    