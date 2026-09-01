#!/usr/bin/env python3
"""
Demo script for visualizing Hilbert curve patching in Vision Transformers.

This script demonstrates:
1. How a Hilbert curve maps through an image grid
2. How patching/unpatching with Hilbert ordering works
3. Visual comparison between row-major and Hilbert curve patch ordering

The plotting helpers live here rather than in flaxdiff.models.hilbert so that
importing the package never pulls in matplotlib; the package keeps only the
pure jax/numpy patching machinery, which this script imports.

Usage:
    python demo_hilbert_curve.py [--image IMAGE_PATH] [--patch_size PATCH_SIZE]

Options:
    --image: Path to an image file (default: a synthetic gradient image)
    --patch_size: Size of patches (default: 16)
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from flaxdiff.models.hilbert import (
    hilbert_indices,
    hilbert_patchify,
    hilbert_unpatchify,
    patchify,
)

# --- Visualization ---


def visualize_hilbert_curve(H: int, W: int, patch_size: int, figsize=(12, 5)):
    """
    Visualize the Hilbert curve mapping for a given image patch grid size.

    Args:
        H: Image height.
        W: Image width.
        patch_size: Size of each patch.
        figsize: Figure size for the plot.

    Returns:
        The matplotlib Figure object.
    """
    H_P = H // patch_size
    W_P = W // patch_size
    if H_P * W_P == 0:
        print("Warning: Grid dimensions are zero, cannot visualize.")
        return None

    # Get Hilbert curve indices (idx[i] = row-major index of i-th Hilbert point)
    idx = np.array(hilbert_indices(H_P, W_P)) # Convert to numpy for plotting logic

    # Create a grid representation for visualization: grid[row, col] = Hilbert sequence index
    grid = np.full((H_P, W_P), -1.0) # Use float and -1 for unmapped cells
    for i, idx_val in enumerate(idx):
        # Convert linear row-major index to row, col
        row = idx_val // W_P
        col = idx_val % W_P
        if 0 <= row < H_P and 0 <= col < W_P:
             grid[row, col] = i # Assign Hilbert sequence index 'i'

    # Create a colormap that transitions smoothly along the Hilbert path
    cmap = LinearSegmentedColormap.from_list('hilbert', ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']) # Blue -> Green -> Yellow -> Red

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: Original Grid (Row-Major Order) ---
    orig_grid = np.arange(H_P * W_P).reshape((H_P, W_P))
    im0 = axes[0].imshow(orig_grid, cmap='viridis', aspect='auto')
    axes[0].set_title(f"Original Grid ({H_P}x{W_P})\n(Row-Major Order)")
    # Remove text labels for indices
    axes[0].set_xticks(np.arange(W_P))
    axes[0].set_yticks(np.arange(H_P))
    axes[0].set_xticklabels(np.arange(W_P))
    axes[0].set_yticklabels(np.arange(H_P))
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Row-Major Index")

    # Draw the row-major scanning line
    if H_P * W_P > 1:
        row_major_coords_y = []
        row_major_coords_x = []
        for r_idx in range(H_P * W_P):
            r = r_idx // W_P
            c = r_idx % W_P
            row_major_coords_y.append(r + 0.5) # Cell center
            row_major_coords_x.append(c + 0.5) # Cell center
        axes[0].plot(row_major_coords_x, row_major_coords_y, color='red', linestyle='-', linewidth=1.5, alpha=0.8)
        axes[0].plot(row_major_coords_x[0], row_major_coords_y[0], 'go', markersize=5, label='Start') # Smaller marker
        axes[0].plot(row_major_coords_x[-1], row_major_coords_y[-1], 'mo', markersize=5, label='End') # Smaller marker
        axes[0].legend(fontsize='x-small')

    # --- Plot 2: Hilbert Curve Ordering ---
    # Mask unmapped cells for visualization
    masked_grid = np.ma.masked_where(grid == -1, grid)
    im1 = axes[1].imshow(masked_grid, cmap=cmap, aspect='auto', vmin=0, vmax=max(0, len(idx)-1))
    axes[1].set_title(f"Hilbert Curve Ordering ({len(idx)} points)")
    # Add text labels for Hilbert indices
    # for r in range(H_P):
    #     for c in range(W_P):
    #         if grid[r,c] != -1:
    #             axes[1].text(c, r, f'{int(grid[r, c])}', ha='center', va='center', color='black', fontsize=8)
    axes[1].set_xticks(np.arange(W_P))
    axes[1].set_yticks(np.arange(H_P))
    axes[1].set_xticklabels(np.arange(W_P))
    axes[1].set_yticklabels(np.arange(H_P))
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Hilbert Sequence Index")

    # Draw the actual curve connecting centers of patches in Hilbert order
    if len(idx) > 1:
        coords = []
        # Find the (row, col) for each Hilbert index i
        # This is faster than np.where in a loop for dense grids
        row_col_map = {int(grid[r, c]): (r, c) for r in range(H_P) for c in range(W_P) if grid[r,c] != -1}
        for i in range(len(idx)):
             if i in row_col_map:
                 coords.append(row_col_map[i])

        if coords:
             # Get coordinates for plotting (centers of cells)
             y_coords = [r + 0.5 for r, c in coords] # Cell center
             x_coords = [c + 0.5 for r, c in coords] # Cell center
             axes[1].plot(x_coords, y_coords, color='red', linestyle='-', linewidth=1.5, alpha=0.8) # Ensure Hilbert curve is red
             # Mark start point
             axes[1].plot(x_coords[0], y_coords[0], 'go', markersize=5, label='Start') # Smaller marker
             # Mark end point
             axes[1].plot(x_coords[-1], y_coords[-1], 'mo', markersize=5, label='End') # Smaller marker
             axes[1].legend(fontsize='x-small')

    plt.tight_layout()
    return fig

def create_patch_grid(patches_np: np.ndarray, patch_size: int, channels: int, grid_cols: int = 10, border: int = 1):
    """
    Create a visualization grid from a sequence of patches.

    Args:
        patches_np: Patch tensor [N, P*P*C] as NumPy array.
        patch_size: Size of square patches (P).
        channels: Number of channels (C).
        grid_cols: How many patches wide the grid should be.
        border: Width of the border between patches.

    Returns:
        Grid image as NumPy array.
    """
    n_patches = patches_np.shape[0]
    if n_patches == 0:
        return np.zeros((patch_size, patch_size, channels), dtype=patches_np.dtype)

    # Reshape patches to actual images [N, P, P, C]
    try:
        patch_imgs = patches_np.reshape(n_patches, patch_size, patch_size, channels)
    except ValueError as e:
        print(f"Error reshaping patches: {e}")
        print(f"Input shape: {patches_np.shape}, Expected P*P*C: {patch_size*patch_size*channels}")
        # Return a placeholder or re-raise
        return np.zeros((patch_size, patch_size, channels), dtype=patches_np.dtype)


    # Determine grid size
    grid_cols = min(grid_cols, n_patches)
    grid_rows = int(np.ceil(n_patches / grid_cols))

    # Create the grid canvas (add border space)
    grid_h = grid_rows * (patch_size + border) - border
    grid_w = grid_cols * (patch_size + border) - border

    # Initialize grid (e.g., with white background)
    if channels == 1:
         grid = np.ones((grid_h, grid_w), dtype=patch_imgs.dtype) * 255
    else:
         grid = np.ones((grid_h, grid_w, channels), dtype=patch_imgs.dtype) * 255


    # Fill the grid with patches
    for i in range(n_patches):
        row = i // grid_cols
        col = i % grid_cols

        # Calculate top-left corner for the patch
        y_start = row * (patch_size + border)
        x_start = col * (patch_size + border)

        # Place the patch
        if channels == 1:
             grid[y_start:y_start+patch_size, x_start:x_start+patch_size] = patch_imgs[i, :, :, 0]
        else:
             grid[y_start:y_start+patch_size, x_start:x_start+patch_size] = patch_imgs[i]

    # Clip to valid range ([0, 1] for float, [0, 255] for int)
    if np.issubdtype(grid.dtype, np.floating):
        grid = np.clip(grid, 0, 1)
    elif np.issubdtype(grid.dtype, np.integer):
        grid = np.clip(grid, 0, 255).astype(np.uint8) # Ensure uint8 for imshow

    # Squeeze if grayscale
    if channels == 1:
       grid = grid.squeeze()

    return grid


def demo_hilbert_patching(image_np: np.ndarray, patch_size: int = 8, figsize=(15, 12)):
    """
    Demonstrate the Hilbert curve patching process on an image.

    Args:
        image_np: NumPy array of shape [H, W, C] or [H, W].
        patch_size: Size of square patches.
        figsize: Figure size for the plot.

    Returns:
        Tuple of (fig_main, fig_reconstruction) matplotlib Figure objects.
    """
    # Handle grayscale images
    if image_np.ndim == 2:
        image_np = np.expand_dims(image_np, axis=-1) # Add channel dim

    # Ensure image dimensions are divisible by patch_size by cropping
    H_orig, W_orig, C = image_np.shape
    H = (H_orig // patch_size) * patch_size
    W = (W_orig // patch_size) * patch_size
    if H != H_orig or W != W_orig:
        print(f"Warning: Cropping image from ({H_orig}, {W_orig}) to ({H}, {W}) to be divisible by patch_size={patch_size}")
        image_np = image_np[:H, :W, :]

    # Convert to JAX array and add batch dimension
    image = jnp.expand_dims(jnp.array(image_np), axis=0) # [1, H, W, C]
    B, H, W, C = image.shape
    H_P = H // patch_size
    W_P = W // patch_size

    print(f"Image shape: {image.shape}, Patch size: {patch_size}, Grid: {H_P}x{W_P}")

    # --- Create Main Visualization Figure ---
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Original image (cropped)
    display_img = np.array(image[0]) # Back to numpy for display
    axes[0, 0].imshow(display_img.squeeze(), cmap='gray' if C==1 else None)
    axes[0, 0].set_title(f"Original Image ({H}x{W})")
    axes[0, 0].axis('off')

    # 2. Original image with Hilbert curve overlay
    axes[0, 1].imshow(display_img.squeeze(), cmap='gray' if C==1 else None)
    axes[0, 1].set_title("Image with Hilbert Curve Overlay")

    # Calculate Hilbert path coordinates on the image scale
    idx = np.array(hilbert_indices(H_P, W_P))
    if len(idx) > 0:
        # Create grid to find coordinates easily
        grid = np.full((H_P, W_P), -1)
        for i, idx_val in enumerate(idx):
            row, col = idx_val // W_P, idx_val % W_P
            grid[row, col] = i

        # Get patch center coordinates in Hilbert order
        coords = []
        row_col_map = {int(grid[r, c]): (r, c) for r in range(H_P) for c in range(W_P) if grid[r,c] != -1}
        for i in range(len(idx)):
             if i in row_col_map:
                 coords.append(row_col_map[i])

        if len(coords) > 1:
            # Scale coordinates to image pixel space
            y_coords = [(r * patch_size + patch_size / 2) for r, c in coords]
            x_coords = [(c * patch_size + patch_size / 2) for r, c in coords]
            axes[0, 1].plot(x_coords, y_coords, 'r-', linewidth=1.5, alpha=0.7)
            axes[0, 1].plot(x_coords[0], y_coords[0], 'go', markersize=5) # Start
            axes[0, 1].plot(x_coords[-1], y_coords[-1], 'mo', markersize=5) # End
    axes[0, 1].axis('off')


    # 3. Apply Hilbert Patchify
    patches_hilbert, inv_idx = hilbert_patchify(image, patch_size)
    print(f"Hilbert patches shape: {patches_hilbert.shape}") # [B, N, P*P*C]
    print(f"Inverse index shape: {inv_idx.shape}") # [total_patches_expected]

    # For comparison, get row-major patches
    patches_row_major = patchify(image, patch_size)
    print(f"Row-major patches shape: {patches_row_major.shape}") # [B, N, P*P*C]

    # Display a subset of patches in both orderings
    n_display = min(60, patches_hilbert.shape[1]) # Show first N patches

    # Convert JAX arrays to NumPy for visualization function
    patches_hilbert_np = np.array(patches_hilbert[0, :n_display])
    patches_row_major_np = np.array(patches_row_major[0, :n_display])

    # Create visualization grids
    patch_grid_row = create_patch_grid(patches_row_major_np, patch_size, C, grid_cols=10)
    patch_grid_hil = create_patch_grid(patches_hilbert_np, patch_size, C, grid_cols=10)

    axes[1, 0].imshow(patch_grid_row, cmap='gray' if C==1 else None, aspect='auto')
    axes[1, 0].set_title(f"First {n_display} Patches (Row-Major Order)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(patch_grid_hil, cmap='gray' if C==1 else None, aspect='auto')
    axes[1, 1].set_title(f"First {n_display} Patches (Hilbert Order)")
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    fig.suptitle(f"Hilbert Patching Demo (Patch Size: {patch_size}x{patch_size})", fontsize=16)


    # --- Create Reconstruction Figure ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))

    # 4. Unpatchify and verify
    reconstructed = hilbert_unpatchify(patches_hilbert, inv_idx, patch_size, H, W, C)
    print(f"Reconstructed image shape: {reconstructed.shape}")

    # Compute and print reconstruction error
    error = jnp.mean(jnp.abs(image - reconstructed))
    print(f"Reconstruction Mean Absolute Error: {error:.6f}")

    # Display original and reconstructed
    reconstructed_np = np.array(reconstructed[0]) # Back to numpy
    axes2[0].imshow(display_img.squeeze(), cmap='gray' if C==1 else None)
    axes2[0].set_title("Original Image (Cropped)")
    axes2[0].axis('off')

    axes2[1].imshow(reconstructed_np.squeeze(), cmap='gray' if C==1 else None)
    axes2[1].set_title(f"Reconstructed from Hilbert Patches\nMAE: {error:.4f}")
    axes2[1].axis('off')

    plt.tight_layout()
    fig2.suptitle("Image Reconstruction Verification", fontsize=16)

    return fig, fig2


# --- Image sources ---


def gradient_image(H: int = 512, W: int = 512, C: int = 3) -> np.ndarray:
    """Synthetic gradient image in [0, 1], so the demo runs offline."""
    img = np.zeros((H, W, C), dtype=np.float32)
    xv, yv = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    if C == 3:
        img[..., 0] = xv           # Red varies with width
        img[..., 1] = yv           # Green varies with height
        img[..., 2] = (xv + yv) / 2
    else:
        img[..., 0] = (xv + yv) / 2
    return img


def load_image(path: str, size: int = 512) -> np.ndarray:
    """Load an image, convert to RGB and resize to size x size, in [0, 1]."""
    img = Image.open(path)
    print(f"Loaded image of size: {img.size}")
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def simulate_transformer_block(patches: jnp.ndarray) -> jnp.ndarray:
    """Stand-in for a transformer block: mix patch features with fixed weights.

    Enough to show that a sequence model consumes the Hilbert-ordered patches
    and the result unpatchifies back into image space.
    """
    _, _, patch_dim = patches.shape
    key = jax.random.PRNGKey(42)
    weights = jnp.eye(patch_dim) + jax.random.normal(key, (patch_dim, patch_dim)) * 0.05
    return jnp.matmul(patches, weights)


def main():
    parser = argparse.ArgumentParser(description='Demonstrate Hilbert curve patching for ViTs')
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--size', type=int, default=512, help='Image is resized to size x size')
    args = parser.parse_args()

    if args.image and os.path.exists(args.image):
        print(f"Loading image from {args.image}...")
        image = load_image(args.image, size=args.size)
    else:
        print("No --image given, using a synthetic gradient image...")
        image = gradient_image(args.size, args.size)

    patch_size = args.patch_size
    h, w = image.shape[:2]

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

    # 3. Push the patch sequence through a stand-in transformer block
    print("\n3. Simulating how patches would flow through a transformer...")
    jax_img = jnp.array(image)[None, ...]  # [1, H, W, C]
    patches, inv_idx = hilbert_patchify(jax_img, patch_size)
    print(f"Original image shape: {jax_img.shape}")
    print(f"Patches shape: {patches.shape}")

    processed_patches = simulate_transformer_block(patches)
    h, w, c = jax_img.shape[1:]
    reconstructed = hilbert_unpatchify(processed_patches, inv_idx, patch_size, h, w, c)

    fig_processed, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(np.array(jax_img[0]))
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[1].imshow(np.clip(np.array(reconstructed[0]), 0, 1))
    ax[1].set_title("After Simulated Transformer Processing")
    ax[1].axis('off')
    plt.tight_layout()

    # 4. Edge case: a small, non-square patch grid
    print("\n4. Small image (3x5 patches) edge case...")
    small = gradient_image(3 * 4, 5 * 4)
    fig_small, fig_small_recon = demo_hilbert_patching(small, patch_size=4)

    print("\nSaving visualization figures...")
    figures = {
        "hilbert_curve_mapping.png": fig_map,
        "hilbert_patch_demo.png": fig_demo,
        "hilbert_patch_reconstruction.png": fig_recon,
        "hilbert_transformer_simulation.png": fig_processed,
        "hilbert_small_image_demo.png": fig_small,
        "hilbert_small_image_reconstruction.png": fig_small_recon,
    }
    for filename, figure in figures.items():
        if figure is not None:
            figure.savefig(filename)
            print(f"- {filename}")

    plt.show()


if __name__ == "__main__":
    main()
