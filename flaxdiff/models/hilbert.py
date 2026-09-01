import jax
import jax.numpy as jnp
import numpy as np
import math
import einops
from typing import Tuple

# --- 2D Positional Encoding (shared by simple_dit and ssm_dit) ---

def build_2d_sincos_pos_embed(emb_dim: int, H_P: int, W_P: int) -> np.ndarray:
    """Fixed MAE-style 2D sin-cos positional embedding, row-major [H_P * W_P, emb_dim].
    Half the channels encode the row, half the column. For Hilbert order, index
    the result by hilbert_indices(H_P, W_P).
    """
    assert emb_dim % 4 == 0, f"emb_dim must be divisible by 4 for 2D sincos, got {emb_dim}"
    half = emb_dim // 2
    quarter = half // 2

    omega = np.arange(quarter, dtype=np.float32) / quarter
    omega = 1.0 / (10000.0 ** omega)  # [quarter]

    rows = np.arange(H_P, dtype=np.float32)
    cols = np.arange(W_P, dtype=np.float32)

    row_emb = np.einsum('h,d->hd', rows, omega)
    col_emb = np.einsum('w,d->wd', cols, omega)

    row_sin = np.sin(row_emb)
    row_cos = np.cos(row_emb)
    col_sin = np.sin(col_emb)
    col_cos = np.cos(col_emb)

    pos = np.zeros((H_P, W_P, emb_dim), dtype=np.float32)
    pos[..., 0:quarter] = row_sin[:, None, :]
    pos[..., quarter:half] = row_cos[:, None, :]
    pos[..., half:half + quarter] = col_sin[None, :, :]
    pos[..., half + quarter:] = col_cos[None, :, :]
    return pos.reshape(H_P * W_P, emb_dim)


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

def zigzag_indices(H_P: int, W_P: int) -> jnp.ndarray:
    """
    Zigzag (serpentine) scan indices for an H_P x W_P patch grid, as in ZigMa.
    Even rows go left-to-right, odd rows right-to-left. result[i] is the
    row-major index of the i-th patch in the zigzag sequence.
    """
    indices = []
    for r in range(H_P):
        if r % 2 == 0:
            for c in range(W_P):
                indices.append(r * W_P + c)
        else:
            for c in range(W_P - 1, -1, -1):
                indices.append(r * W_P + c)
    return jnp.array(indices, dtype=jnp.int32)


def zigzag_patchify(x: jnp.ndarray, patch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract patches and reorder them in zigzag scan order.
    Same contract as hilbert_patchify: returns (patches_zigzag, inv_idx).
    """
    B, H, W, C = x.shape
    H_P = H // patch_size
    W_P = W // patch_size
    total_patches_expected = H_P * W_P

    patches_row_major = patchify(x, patch_size)
    idx = zigzag_indices(H_P, W_P)
    inv_idx = inverse_permutation(idx, total_patches_expected)
    patches_zigzag = patches_row_major[:, idx, :]
    return patches_zigzag, inv_idx


def zigzag_unpatchify(x: jnp.ndarray, inv_idx: jnp.ndarray, patch_size: int, H: int, W: int, C: int) -> jnp.ndarray:
    """
    Restore row-major order from zigzag-ordered patches and convert to image.
    The scatter in hilbert_unpatchify only depends on inv_idx, so just use it.
    """
    return hilbert_unpatchify(x, inv_idx, patch_size, H, W, C)


def hilbert_unpatchify(x: jnp.ndarray, inv_idx: jnp.ndarray, patch_size: int, H: int, W: int, C: int) -> jnp.ndarray:
    """
    Restore the original row-major order of patches and convert back to image.
    (Revised version to be JIT-compatible)

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
    N = x.shape[1] # Number of patches provided in Hilbert order (h dimension)
    patch_dim = x.shape[2]
    H_P = H // patch_size
    W_P = W // patch_size
    total_patches_expected = H_P * W_P # Number of patches expected in output (k dimension)

    # Ensure inv_idx has the expected total size
    assert inv_idx.shape[0] == total_patches_expected, \
         f"Inverse index size {inv_idx.shape[0]} does not match expected total patches {total_patches_expected}"

    # --- JIT-compatible Scatter using Gather and Where ---

    # Target shape for row-major patches
    target_shape = (B, total_patches_expected, patch_dim)

    # Create indices for gathering from x (Hilbert order h) based on inv_idx (map k -> h)
    # Clamp invalid indices (-1) to 0; we'll mask these results later.
    # Values must be < N (the actual number of patches in x).
    h_indices_for_gather = jnp.maximum(inv_idx, 0) # Shape [total_patches_expected]

    # Define gather for one batch item: output[k] = input[h_indices[k]]
    def gather_one_batch(single_x, h_indices):
        # single_x: [N, D], h_indices: [K] where K = total_patches_expected
        # Check bounds: Ensure indices used are within the valid range [0, N-1] of single_x
        # This check might be redundant if inv_idx < N mask is applied correctly later,
        # but can prevent out-of-bounds access if N is smaller than expected.
        safe_h_indices = jnp.minimum(h_indices, N - 1)
        return single_x[safe_h_indices, :] # Result: [K, D]

    # Use vmap to gather across the batch dimension
    # Result `gathered_patches` has shape [B, total_patches_expected, patch_dim]
    gathered_patches = jax.vmap(gather_one_batch, in_axes=(0, None))(x, h_indices_for_gather)

    # Create a mask for valid k indices (where corresponding h was valid)
    # A valid h must be >= 0 and < N (number of patches provided in x).
    valid_k_mask = (inv_idx >= 0) & (inv_idx < N) # Shape [total_patches_expected]

    # Expand mask for broadcasting with patch dimensions: [1, K, 1]
    valid_k_mask_broadcast = valid_k_mask[None, :, None]

    # Use `where` to select gathered patches for valid k, and zeros otherwise.
    # This is JIT-friendly as shapes are consistent.
    row_major_patches = jnp.where(
        valid_k_mask_broadcast,
        gathered_patches,
        jnp.zeros(target_shape, dtype=x.dtype) # Use explicit shape for zeros
    )
    # --- End JIT-compatible Scatter ---

    # Convert the fully populated (or zero-padded) row-major patches back to image
    return unpatchify(row_major_patches, patch_size, H, W, C)
