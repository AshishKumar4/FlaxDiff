"""Forward-pass tests for every model architecture.

Every architecture must build, run a forward pass at a small resolution,
and return the input shape back in float32. This is the first line of
defense against dead configs and shape bugs.
"""

import jax
import jax.numpy as jnp
import pytest

from flaxdiff.models.simple_unet import Unet
from flaxdiff.models.simple_dit import SimpleDiT
from flaxdiff.models.simple_mmdit import SimpleMMDiT
from flaxdiff.models.simple_vit import UViT
from flaxdiff.models.ssm_dit import HybridSSMAttentionDiT

RES = 32


def run_forward(model, rng, x, temb, textcontext):
    params = model.init(rng, x, temb, textcontext)
    return model.apply(params, x, temb, textcontext)


def small_inputs(rng, res=RES, channels=3):
    x = jax.random.normal(rng, (2, res, res, channels))
    temb = jnp.ones((2,))
    textcontext = jnp.ones((2, 77, 768), dtype=jnp.float32)
    return x, temb, textcontext


def test_unet_forward(rng):
    model = Unet(
        emb_features=64,
        feature_depths=[16, 32],
        attention_configs=[None, {"heads": 2, "dtype": jnp.float32, "flash_attention": False,
                                  "use_projection": False, "use_self_and_cross": False}],
        num_res_blocks=1,
        num_middle_res_blocks=1,
    )
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32


def test_simple_dit_forward(rng):
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape
    assert out.dtype == jnp.float32


@pytest.mark.xfail(strict=True, reason="bug: DiT final_proj keeps compute dtype, bf16 output vs fp32 loss")
def test_simple_dit_bf16_outputs_fp32(rng):
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2,
                      mlp_ratio=2, dtype=jnp.bfloat16)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.dtype == jnp.float32


@pytest.mark.xfail(strict=True, reason="bug: unpatchify assumes a square patch grid")
def test_simple_dit_non_square(rng):
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    x = jax.random.normal(rng, (2, 16, 64, 3))
    temb = jnp.ones((2,))
    textcontext = jnp.ones((2, 77, 768), dtype=jnp.float32)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


def test_simple_mmdit_forward(rng):
    model = SimpleMMDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


def test_uvit_forward(rng):
    model = UViT(patch_size=4, emb_features=64, num_layers=4, num_heads=2)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


@pytest.mark.parametrize("kwargs", [
    {},
    {"use_hilbert": True},
    {"use_zigzag": True},
    {"use_zigzag": True, "use_2d_fusion": True},
    {"use_hilbert": True, "use_2d_fusion": True},
])
def test_hybrid_dit_forward(rng, kwargs):
    model = HybridSSMAttentionDiT(patch_size=4, emb_features=64, num_layers=4,
                                  num_heads=2, mlp_ratio=2, ssm_state_dim=8, **kwargs)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


@pytest.mark.xfail(strict=True, reason="bug: UViT applies the hilbert permutation twice")
def test_uvit_hilbert_matches_raster_information(rng):
    """A zero-layer sanity check: with the permutation applied and inverted once,
    patch content must land back at its own spatial position. UViT permutes twice
    on the way in but inverts once on the way out, scrambling the output."""
    from flaxdiff.models.hilbert import hilbert_patchify, hilbert_unpatchify, hilbert_indices, inverse_permutation

    x = jax.random.normal(rng, (1, 16, 16, 3))
    patches, inv_idx = hilbert_patchify(x, 4)
    # UViT's forward applies idx again on the already-permuted patches
    idx = hilbert_indices(4, 4)
    double_permuted = patches[:, idx, :]
    rec = hilbert_unpatchify(double_permuted, inv_idx, 4, 16, 16, 3)
    assert jnp.allclose(rec, x)
