"""I-JEPA multi-block masking over a patch grid.

The paper samples several target blocks with a random scale and aspect ratio
and takes the context to be what is left. Under jit every mask has to come out
the same shape on every step, so the geometry is resolved once, at
construction, against the static patch grid:

  - the block area is fixed to the value inside the configured scale range that
    admits the most (height, width) factorizations within the aspect ratio
    range, so each sample still draws a genuinely different block shape and
    position, and every target block has exactly the same token count;
  - the context is a uniformly random subset of the complement of the target
    union, sized S - num_targets * area. That size is always available (the
    union is at most num_targets disjoint blocks), and when the blocks happen
    to overlap the context simply drops the surplus rather than changing shape.

Indices refer to positions in the scan-ordered token sequence that
PatchSequenceEmbed produces, and come out sorted so that an SSM mixer scans
them in a meaningful order.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


def _factorizations(area: int, grid: Tuple[int, int], aspect: Tuple[float, float]):
    """(h, w) pairs of the given area that fit the grid and the aspect range."""
    H_P, W_P = grid
    pairs = []
    for h in range(1, min(H_P, area) + 1):
        if area % h:
            continue
        w = area // h
        if w <= W_P and aspect[0] <= h / w <= aspect[1]:
            pairs.append((h, w))
    return pairs


@dataclass(frozen=True)
class MultiBlockMask:
    """Static mask geometry for one patch grid, plus the sampler over it."""
    grid: Tuple[int, int]
    num_targets: int
    block_shapes: Tuple[Tuple[int, int], ...]
    num_context: int

    @property
    def num_patches(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def block_area(self) -> int:
        h, w = self.block_shapes[0]
        return h * w

    def sample(self, rng: jax.Array, batch_size: int):
        """Draw (context_idx [B, num_context], target_idx [B, M, block_area])."""
        H_P, W_P = self.grid
        S = self.num_patches
        M = self.num_targets

        heights = jnp.asarray([h for h, _ in self.block_shapes])
        widths = jnp.asarray([w for _, w in self.block_shapes])
        # flat offsets of a block's tokens from its top-left corner, per shape
        offsets = jnp.asarray([[(i // w) * W_P + (i % w) for i in range(h * w)]
                               for h, w in self.block_shapes])

        shape_key, pos_key, context_key = jax.random.split(rng, 3)
        choice = jax.random.randint(shape_key, (batch_size, M), 0, len(self.block_shapes))
        corner = jax.random.uniform(pos_key, (batch_size, M, 2))
        top = jnp.floor(corner[..., 0] * (H_P - heights[choice] + 1)).astype(jnp.int32)
        left = jnp.floor(corner[..., 1] * (W_P - widths[choice] + 1)).astype(jnp.int32)
        target_idx = (top * W_P + left)[..., None] + offsets[choice]

        is_target = jnp.zeros((batch_size, S), dtype=bool).at[
            jnp.arange(batch_size)[:, None], target_idx.reshape(batch_size, -1)].set(True)
        # target tokens sort last, so the first num_context entries of a random
        # ordering are a uniform sample of the complement
        order = jax.random.uniform(context_key, (batch_size, S)) + is_target
        context_idx = jnp.sort(jnp.argsort(order, axis=1)[:, :self.num_context], axis=1)
        return context_idx, target_idx


def multi_block_mask(
    grid: Tuple[int, int],
    num_targets: int = 4,
    scale: Tuple[float, float] = (0.15, 0.2),
    aspect: Tuple[float, float] = (0.75, 1.5),
) -> MultiBlockMask:
    """Resolve the I-JEPA mask geometry for a patch grid."""
    S = grid[0] * grid[1]
    candidates = [
        (area, _factorizations(area, grid, aspect))
        for area in range(max(1, math.ceil(scale[0] * S)), math.floor(scale[1] * S) + 1)
    ]
    candidates = [(area, shapes) for area, shapes in candidates if shapes]
    if not candidates:
        raise ValueError(
            f"No block of scale {scale} on a {grid} grid has an aspect ratio in "
            f"{aspect}; widen one of the ranges or use a finer patch grid.")

    midpoint = 0.5 * (scale[0] + scale[1]) * S
    area, shapes = min(candidates, key=lambda c: (-len(c[1]), abs(c[0] - midpoint)))

    num_context = S - num_targets * area
    if num_context <= 0:
        raise ValueError(
            f"{num_targets} target blocks of {area} tokens leave no context on a "
            f"{grid} grid ({S} tokens).")

    return MultiBlockMask(
        grid=grid,
        num_targets=num_targets,
        block_shapes=tuple(shapes),
        num_context=num_context,
    )
