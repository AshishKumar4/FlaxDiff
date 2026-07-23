"""One-time conversion of pre-consolidation checkpoints to the new param layout.

The DiT backbone consolidation moved the patchify/conditioning/output modules
into submodules (embed/, conditioning/, output/), which renames their param
paths. This remaps an old orbax checkpoint in place and re-saves it so it
loads against the new models. UNet and UViT checkpoints need no conversion.
MMDiT checkpoints are NOT convertible - the architecture itself changed
(dual-stream rewrite), not just the naming.

Usage:
    python scripts/convert_legacy_checkpoint.py <checkpoint_dir> <output_dir>

Note on time conditioning: FourierEmbedding frequencies used to come from
jax's PRNG, whose default implementation changed in jax 0.5.0. Checkpoints
trained before that already sample with subtly different time conditioning on
modern jax; the frequencies are now fixed numpy values, so models trained
from this version onward are stable. This script cannot repair the old
provenance - expect slightly off conditioning on pre-0.5.0 checkpoints.
"""

import sys

import numpy as np
import orbax.checkpoint

# Top-level module renames from the DiT backbone consolidation
RENAMES = {
    'patch_embed': ('embed', 'patch_embed'),
    'hilbert_projection': ('embed', 'hilbert_projection'),
    'time_embed': ('conditioning', 'time_embed'),
    'text_context_proj': ('conditioning', 'text_context_proj'),
    'final_norm': ('output', 'final_norm'),
    'final_proj': ('output', 'final_proj'),
}


def convert_params(params: dict) -> dict:
    """Remap the top-level module names of a DiT-family param tree."""
    converted = {}
    for key, value in params.items():
        if key in RENAMES:
            parent, child = RENAMES[key]
            converted.setdefault(parent, {})[child] = value
        else:
            converted[key] = value
    return converted


def convert_state(state: dict) -> dict:
    state = dict(state)
    for key in ('params', 'ema_params'):
        if key in state and isinstance(state[key], dict) and 'params' in state[key]:
            state[key] = {**state[key], 'params': convert_params(state[key]['params'])}
    return state


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    checkpoint_dir, output_dir = sys.argv[1], sys.argv[2]

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, checkpointer,
        orbax.checkpoint.CheckpointManagerOptions(create=False))
    step = manager.latest_step()
    print(f"Converting checkpoint at step {step} from {checkpoint_dir}")
    ckpt = manager.restore(step)

    for key in ('state', 'best_state'):
        if key in ckpt and ckpt[key] is not None:
            ckpt[key] = convert_state(ckpt[key])
            print(f"Converted {key}")

    out_manager = orbax.checkpoint.CheckpointManager(
        output_dir, orbax.checkpoint.PyTreeCheckpointer(),
        orbax.checkpoint.CheckpointManagerOptions(create=True))
    out_manager.save(step, ckpt)
    out_manager.wait_until_finished()
    print(f"Saved converted checkpoint to {output_dir} at step {step}")


if __name__ == '__main__':
    main()
