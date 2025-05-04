import jax
import jax.numpy as jnp
import numpy as np
import math
import einops
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple

# --- Core Hilbert Curve Logic ---

def _d2xy(n: int, d: int) -> Tuple[int, int]:
    """
    Convert a 1D Hilbert curve index to 2D (x, y) coordinates.
    Based on the algorithm from Wikipedia / common implementations.

    Args:
        n: Size of the grid (must be a power of 2).
        d: 1D Hilbert curve index (0 to n*n-1).

    Returns:
        Tuple of (x, y) coordinates (column, row).
    """
    x = y = 0
    t = d
    s = 1
    while (s < n):
        # Extract the two bits for the current level
        rx = (t >> 1) & 1
        ry = (t ^ rx) & 1 # Use XOR to determine the y bit based on d's pattern

        # Rotate and flip the quadrant appropriately
        if ry == 0:
            if rx == 1:
                x = (s - 1) - x
                y = (s - 1) - y
            # Swap x and y
            x, y = y, x

        # Add the offsets for the current quadrant
        x += s * rx
        y += s * ry

        # Move to the next level
        t >>= 2 # Equivalent to t //= 4
        s <<= 1 # Equivalent to s *= 2
    return x, y # Returns (column, row)

def hilbert_indices(H_P: int, W_P: int) -> jnp.ndarray:
    """
    Generate Hilbert curve indices for a rectangular grid of H_P x W_P patches.
    The indices map Hilbert sequence order to row-major order.

    Args:
        H_P: Height in patches.
        W_P: Width in patches.

    Returns:
        1D JAX array where result[i] is the row-major index of the i-th patch
        in the Hilbert curve sequence. The length of the array is the number
        of valid patches (H_P * W_P).
    """
    # Find the smallest power of 2 that fits both dimensions
    size = max(H_P, W_P)
    # Calculate the order (e.g., order=3 means n=8)
    order = math.ceil(math.log2(size)) if size > 0 else 0
    n = 1 << order # n = 2**order

    # Generate (row, col) coordinates for each index in the Hilbert curve order
    # within the square n x n grid
    coords_in_hilbert_order = []
    total_patches_needed = H_P * W_P
    if total_patches_needed == 0:
        return jnp.array([], dtype=jnp.int32)

    for d in range(n * n):
        # Get (col, row) for Hilbert index d in the n x n grid
        x, y = _d2xy(n, d)

        # Keep only coordinates within the actual H_P x W_P grid
        if x < W_P and y < H_P:
            coords_in_hilbert_order.append((y, x)) # Store as (row, col)

            # Early exit once we have all needed coordinates
            if len(coords_in_hilbert_order) == total_patches_needed:
                break

    # Convert (row, col) pairs (which are in Hilbert order)
    # to linear indices in row-major order
    # indices[i] = row-major index of the i-th point in the Hilbert sequence
    indices = jnp.array([r * W_P + c for r, c in coords_in_hilbert_order], dtype=jnp.int32)
    return indices

def inverse_permutation(idx: jnp.ndarray, total_size: int) -> jnp.ndarray:
    """
    Compute the inverse permutation of the given indices.
    Maps target index (e.g., row-major) back to source index (e.g., Hilbert sequence).

    Args:
        idx: Array where idx[i] is the target index for source index i.
             (e.g., idx[h] = k, where h is Hilbert sequence index, k is row-major index)
             Assumes idx contains unique values representing the target indices.
             Length of idx is N (number of valid patches).
        total_size: The total number of possible target indices (e.g., H_P * W_P).

    Returns:
        Array `inv` of size `total_size` such that inv[k] = h if idx[h] = k,
        and inv[k] = -1 if target index k is not present in `idx`.
    """
    # Initialize inverse mapping with -1 (or another indicator for "not mapped")
    inv = jnp.full((total_size,), -1, dtype=jnp.int32)

    # Source indices are 0, 1, ..., N-1 (representing Hilbert sequence order)
    source_indices = jnp.arange(idx.shape[0], dtype=jnp.int32)

    # Set inv[target_index] = source_index
    # inv.at[idx] accesses the elements of inv at the indices specified by idx
    # .set(source_indices) sets these elements to the corresponding source index
    inv = inv.at[idx].set(source_indices)
    return inv

# --- Patching Logic ---

def patchify(x: jnp.ndarray, patch_size: int) -> jnp.ndarray:
    """
    Convert an image tensor to a sequence of patches in row-major order.

    Args:
        x: Image tensor of shape [B, H, W, C].
        patch_size: Size of square patches.

    Returns:
        Tensor of patches [B, N, P*P*C], where N = (H/ps)*(W/ps).
    """
    # Check if dimensions are divisible by patch_size
    B, H, W, C = x.shape
    if H % patch_size != 0 or W % patch_size != 0:
         raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by patch_size ({patch_size})")

    return einops.rearrange(
        x,
        'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', # (h w) becomes the sequence dim
        p1=patch_size, p2=patch_size
    )

def unpatchify(x: jnp.ndarray, patch_size: int, H: int, W: int, C: int) -> jnp.ndarray:
    """
    Convert a sequence of patches (assumed row-major) back to an image tensor.

    Args:
        x: Patch tensor of shape [B, N, P*P*C] where N = (H/ps) * (W/ps).
        patch_size: Size of square patches.
        H: Original image height.
        W: Original image width.
        C: Number of channels.

    Returns:
        Image tensor of shape [B, H, W, C].
    """
    H_P = H // patch_size
    W_P = W // patch_size
    expected_patches = H_P * W_P
    actual_patches = x.shape[1]

    # Ensure the input has the correct number of patches for the target dimensions
    assert actual_patches == expected_patches, \
        f"Number of patches ({actual_patches}) does not match expected ({expected_patches}) for H={H}, W={W}, patch_size={patch_size}"

    return einops.rearrange(
        x,
        'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
        h=H_P, w=W_P, p1=patch_size, p2=patch_size, c=C
    )

def hilbert_patchify(x: jnp.ndarray, patch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract patches from an image and reorder them according to the Hilbert curve.

    Args:
        x: Image tensor of shape [B, H, W, C].
        patch_size: Size of square patches.

    Returns:
        Tuple of:
        - patches_hilbert: Reordered patches tensor [B, N, P*P*C] (N = H_P * W_P).
        - inv_idx: Inverse permutation indices [N] (maps row-major index to Hilbert sequence index, or -1).
    """
    B, H, W, C = x.shape
    H_P = H // patch_size
    W_P = W // patch_size
    total_patches_expected = H_P * W_P

    # Extract patches in row-major order
    patches_row_major = patchify(x, patch_size) # Shape [B, N, P*P*C]

    # Get Hilbert curve indices (maps Hilbert sequence index -> row-major index)
    # idx[h] = k, where h is Hilbert index, k is row-major index
    idx = hilbert_indices(H_P, W_P) # Shape [N]

    # Store inverse mapping for unpatchify
    # inv_idx[k] = h, where k is row-major index, h is Hilbert sequence index
    inv_idx = inverse_permutation(idx, total_patches_expected) # Shape [N]

    # Reorder patches according to Hilbert curve using advanced indexing
    # Select the patches from patches_row_major at the row-major indices specified by idx
    patches_hilbert = patches_row_major[:, idx, :] # Shape [B, N, P*P*C]

    return patches_hilbert, inv_idx

def hilbert_unpatchify(x: jnp.ndarray, inv_idx: jnp.ndarray, patch_size: int, H: int, W: int, C: int) -> jnp.ndarray:
    """
    Restore the original row-major order of patches and convert back to image.

    Args:
        x: Hilbert-ordered patches tensor [B, N, P*P*C] (N = number of patches in Hilbert order).
        inv_idx: Inverse permutation indices [total_patches_expected]
                 (maps row-major index k to Hilbert sequence index h, or -1).
        patch_size: Size of square patches.
        H: Original image height.
        W: Original image width.
        C: Number of channels.

    Returns:
        Image tensor of shape [B, H, W, C].
    """
    B = x.shape[0]
    N = x.shape[1] # Number of patches provided in Hilbert order
    patch_dim = x.shape[2]
    H_P = H // patch_size
    W_P = W // patch_size
    total_patches_expected = H_P * W_P

    # Ensure inv_idx has the expected total size
    # assert inv_idx.shape[0] == total_patches_expected, \
    #     f"Inverse index size {inv_idx.shape[0]} does not match expected total patches {total_patches_expected}"

    # Create output array for row-major patches, initialized with zeros
    # This ensures that any position not covered by the Hilbert curve remains zero
    row_major_patches = jnp.zeros((B, total_patches_expected, patch_dim), dtype=x.dtype)

    # --- Vectorized Scatter Operation ---
    # Find the row-major indices (k) and Hilbert indices (h) that are valid
    all_k_indices = jnp.arange(total_patches_expected, dtype=inv_idx.dtype) # All possible row-major indices k
    all_h_indices = inv_idx # Corresponding Hilbert indices h (or -1)

    # Create a mask for valid indices: h must be non-negative and less than N (the number of input patches)
    valid_mask = (all_h_indices >= 0) & (all_h_indices < N)

    # Filter k and h based on the mask
    target_k = all_k_indices[valid_mask] # Row-major indices to write to
    source_h = all_h_indices[valid_mask] # Hilbert indices to read from

    # Define a function to perform the scatter for a single batch item
    def scatter_one_batch(single_x, single_row_major_patches, source_h_indices, target_k_indices):
        # Gather patches from the Hilbert-ordered input using source_h indices
        source_patches = single_x[source_h_indices, :] # Shape: [num_valid, patch_dim]
        # Scatter these patches into the row-major output array at target_k indices
        return single_row_major_patches.at[target_k_indices, :].set(source_patches)

    # Use vmap to apply the scatter operation across the batch dimension
    scatter_vmapped = jax.vmap(scatter_one_batch, in_axes=(0, 0, None, None), out_axes=0)
    row_major_patches = scatter_vmapped(x, row_major_patches, source_h, target_k)
    # --- End Vectorized Scatter ---

    # Convert the fully populated (or zero-padded) row-major patches back to image
    return unpatchify(row_major_patches, patch_size, H, W, C)


# --- Visualization and Demo ---

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
    # Add text labels for indices
    for r in range(H_P):
        for c in range(W_P):
            axes[0].text(c, r, f'{orig_grid[r, c]}', ha='center', va='center', color='white' if orig_grid[r,c] < (H_P*W_P)/2 else 'black', fontsize=8)
    axes[0].set_xticks(np.arange(W_P))
    axes[0].set_yticks(np.arange(H_P))
    axes[0].set_xticklabels(np.arange(W_P))
    axes[0].set_yticklabels(np.arange(H_P))
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Row-Major Index")

    # --- Plot 2: Hilbert Curve Ordering ---
    # Mask unmapped cells for visualization
    masked_grid = np.ma.masked_where(grid == -1, grid)
    im1 = axes[1].imshow(masked_grid, cmap=cmap, aspect='auto', vmin=0, vmax=max(0, len(idx)-1))
    axes[1].set_title(f"Hilbert Curve Ordering ({len(idx)} points)")
    # Add text labels for Hilbert indices
    for r in range(H_P):
        for c in range(W_P):
            if grid[r,c] != -1:
                axes[1].text(c, r, f'{int(grid[r, c])}', ha='center', va='center', color='black', fontsize=8)
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
             # Fallback (slower):
             # row_indices, col_indices = np.where(grid == i)
             # if len(row_indices) > 0:
             #     coords.append((row_indices[0], col_indices[0]))

        if coords:
             # Get coordinates for plotting (centers of cells)
             y_coords = [r + 0.5 for r, c in coords]
             x_coords = [c + 0.5 for r, c in coords]
             axes[1].plot(x_coords, y_coords, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
             # Mark start point
             axes[1].plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Start (Idx 0)') # Green circle
             # Mark end point
             axes[1].plot(x_coords[-1], y_coords[-1], 'mo', markersize=8, label=f'End (Idx {len(idx)-1})') # Magenta circle
             axes[1].legend(fontsize='small')


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


# --- Example Usage ---
if __name__ == '__main__':
    # Create a sample image (e.g., gradient)
    H, W, C = 64, 80, 3 # Rectangular image
    # H, W, C = 64, 64, 1 # Square grayscale image
    img_np = np.zeros((H, W, C), dtype=np.float32)
    x_coords = np.linspace(0, 1, W)
    y_coords = np.linspace(0, 1, H)
    xv, yv = np.meshgrid(x_coords, y_coords)

    if C == 3:
        img_np[..., 0] = xv  # Red channel varies with width
        img_np[..., 1] = yv  # Green channel varies with height
        img_np[..., 2] = (xv + yv) / 2 # Blue channel is average
    else: # Grayscale
        img_np[..., 0] = (xv + yv) / 2

    # --- Test Visualization ---
    patch_size_vis = 16
    H_vis, W_vis = 4*patch_size_vis, 5*patch_size_vis # e.g., 64x80
    print(f"\nVisualizing Hilbert curve for {H_vis//patch_size_vis}x{W_vis//patch_size_vis} patch grid...")
    fig_vis = visualize_hilbert_curve(H_vis, W_vis, patch_size_vis)
    if fig_vis:
        # fig_vis.savefig("hilbert_curve_visualization.png")
        plt.show() # Display the plot

    # --- Test Patching Demo ---
    patch_size_demo = 8
    print(f"\nRunning Hilbert patching demo with patch size {patch_size_demo}...")
    fig_main, fig_recon = demo_hilbert_patching(img_np, patch_size=patch_size_demo)
    # fig_main.savefig("hilbert_patching_demo.png")
    # fig_recon.savefig("hilbert_reconstruction.png")
    plt.show() # Display the plots

    # --- Test edge case: small image ---
    print("\nTesting small image (3x5 patches)...")
    img_small_np = img_np[:3*4, :5*4, :] # 12x20 image
    fig_main_small, fig_recon_small = demo_hilbert_patching(img_small_np, patch_size=4)
    plt.show()
