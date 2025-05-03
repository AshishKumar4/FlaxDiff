#!/usr/bin/env python3
"""
Demo script for visualizing Hilbert curve patching in Vision Transformers.

This script demonstrates:
1. How a Hilbert curve maps through an image grid
2. How patching/unpatching with Hilbert ordering works
3. Visual comparison between row-major and Hilbert curve patch ordering

Usage:
    python demo_hilbert_curve.py [--image IMAGE_PATH] [--patch_size PATCH_SIZE]

Options:
    --image: Path to an image file (default: will use a sample image)
    --patch_size: Size of patches (default: 16)
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import os
import cv2
from flaxdiff.models.hilbert import (
    visualize_hilbert_curve, 
    demo_hilbert_patching,
    hilbert_patchify,
    hilbert_unpatchify,
    hilbert_indices,
    inverse_permutation,
    patchify
)

def load_sample_image():
    """Load a sample image if no image path is provided."""
    print("Downloading a sample image...")
    # Use a relatively small but detailed image
    url = 'https://www.caledoniaplay.com/wp-content/uploads/2016/01/EDU-PRODUCT-DESCRIPTION-gallery-image-OUTDOOR-SEATING-RUSTIC-LOG-BENCH-1-555x462.jpg'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img) / 255.0  # Normalize to [0, 1]

def load_image(path):
    """Load an image from the given path."""
    img = Image.open(path)
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize to ensure dimensions are divisible by patch_size
    w, h = img.size
    print(f"Loaded image of size: {img.size}")
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    return img

def main():
    parser = argparse.ArgumentParser(description='Demonstrate Hilbert curve patching for ViTs')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    args = parser.parse_args()
    
    # Load image
    if args.image and os.path.exists(args.image):
        print(f"Loading image from {args.image}...")
        image = load_image(args.image)
    else:
        image = load_sample_image()
    
    print(f"Original image shape: {image.shape}")
    image = cv2.resize(image, (512, 512))  # Resize to a fixed size for demo
    print(f"Image shape: {image.shape}")
    # Ensure image dimensions are divisible by patch_size
    h, w = image.shape[:2]
    patch_size = args.patch_size
    
    # Crop to make dimensions divisible by patch_size
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    if new_h != h or new_w != w:
        print(f"Cropping image from {h}x{w} to {new_h}x{new_w} to make divisible by patch size {patch_size}")
        image = image[:new_h, :new_w]
    
    # 1. Visualize the Hilbert curve mapping
    print("\n1. Visualizing Hilbert curve mapping...")
    fig_map = visualize_hilbert_curve(new_h, new_w, patch_size)
    
    # 2. Demonstrate the patching process
    print("\n2. Demonstrating Hilbert curve patching...")
    fig_demo, fig_recon = demo_hilbert_patching(image, patch_size)
    
    # 3. Additional example: Process through a simulated transformer block
    print("\n3. Simulating how patches would flow through a transformer...")
    
    # Convert to JAX array and add batch dimension
    jax_img = jnp.array(image)[None, ...]  # [1, H, W, C]
    
    # Get Hilbert curve patches and inverse indices
    patches, inv_idx = hilbert_patchify(jax_img, patch_size)
    
    print(f"Original image shape: {jax_img.shape}")
    print(f"Patches shape: {patches.shape}")
    
    # Simulate a transformer block that operates on the patch sequence
    def simulate_transformer_block(patches):
        """
        Simulate a transformer block by applying a simple operation to patches.
        For demonstration purposes, we'll just multiply by a learned weight matrix.
        """
        batch, n_patches, patch_dim = patches.shape
        
        # Simulate learned weights (identity + small random values)
        key = jax.random.PRNGKey(42)
        weights = jnp.eye(patch_dim) + jax.random.normal(key, (patch_dim, patch_dim)) * 0.05
        
        # Apply "attention" (just a matrix multiply for demo)
        return jnp.matmul(patches, weights)
    
    # Process patches as if through a transformer
    processed_patches = simulate_transformer_block(patches)
    
    # Unpatchify back to image space
    h, w, c = jax_img.shape[1:]
    reconstructed = hilbert_unpatchify(processed_patches, inv_idx, patch_size, h, w, c)
    
    # Visualize the processed result
    fig_processed, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(np.array(jax_img[0]))
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(np.clip(np.array(reconstructed[0]), 0, 1))
    ax[1].set_title("After Simulated Transformer Processing")
    ax[1].axis('off')
    plt.tight_layout()
    
    # Save all figures
    print("\nSaving visualization figures...")
    fig_map.savefig("hilbert_curve_mapping.png")
    fig_demo.savefig("hilbert_patch_demo.png")
    fig_recon.savefig("hilbert_patch_reconstruction.png")
    fig_processed.savefig("hilbert_transformer_simulation.png")
    
    print("\nDone! Check the following output files:")
    print("- hilbert_curve_mapping.png - Visualizes how Hilbert curve maps through a grid")
    print("- hilbert_patch_demo.png - Shows patch ordering comparison")
    print("- hilbert_patch_reconstruction.png - Shows original vs reconstructed image")
    print("- hilbert_transformer_simulation.png - Shows a simple simulated transformer effect")
    
    # Display plots if running in interactive environment
    plt.show()

if __name__ == "__main__":
    main()