import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from flaxdiff.models.video_unet import FlaxUNet3DConditionModel
import matplotlib.pyplot as plt
import os

# Force CPU backend for testing
os.environ['JAX_PLATFORMS'] = 'cpu'

def test_unet3d_model():
    """
    Test the FlaxUNet3DConditionModel with a simple random input
    and visualize the output.
    """
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(42)
    
    # Define model parameters
    model = FlaxUNet3DConditionModel(
        sample_size=32,  # Small sample for testing
        in_channels=4,
        out_channels=4,
        down_block_types=(
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types=(
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        block_out_channels=(32, 64, 64, 64),  # Smaller channels for testing
        layers_per_block=1,
        cross_attention_dim=64,
        attention_head_dim=8,
        dropout=0.0,
        dtype=jnp.float32
    )
    
    # Create dummy inputs
    batch_size = 1
    num_frames = 4
    sample = jax.random.normal(
        rng, 
        shape=(batch_size, num_frames, 32, 32, 4),
        dtype=jnp.float32
    )
    
    timestep = jnp.array([0], dtype=jnp.int32)
    
    # Create dummy text embeddings
    rng, text_key = jax.random.split(rng)
    encoder_hidden_states = jax.random.normal(
        text_key, 
        shape=(batch_size, 77, 64),  # 77 is standard for CLIP text tokens
        dtype=jnp.float32
    )
    
    # Initialize the model
    rng, init_key = jax.random.split(rng)
    params = model.init(init_key, sample, timestep, encoder_hidden_states)
    
    # Print model summary
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model initialized with {param_count:,} parameters")
    
    # Run a forward pass
    output = model.apply(params, sample, timestep, encoder_hidden_states)
    
    # Print output shape
    print(f"Input shape: {sample.shape}")
    print(f"Output shape: {output['sample'].shape}")
    
    # Visualize a sample frame from both input and output
    frame_idx = 0
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show input (only first 3 channels for visualization)
    axes[0].imshow(sample[0, frame_idx, :, :, :3])
    axes[0].set_title("Input Sample (Frame 0)")
    axes[0].axis('off')
    
    # Show output (only first 3 channels for visualization)
    axes[1].imshow(output['sample'][0, frame_idx, :, :, :3])
    axes[1].set_title("Model Output (Frame 0)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('unet3d_test_output.png')
    plt.close()
    
    print(f"Visualization saved to 'unet3d_test_output.png'")
    
    return model, params

if __name__ == "__main__":
    test_unet3d_model()