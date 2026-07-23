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
        attention_configs=[None, {"heads": 2, "dtype": jnp.float32,
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


def test_simple_dit_bf16_outputs_fp32(rng):
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2,
                      mlp_ratio=2, dtype=jnp.bfloat16)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.dtype == jnp.float32


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


def test_uvit_hilbert_forward(rng):
    """UViT used to apply the hilbert permutation twice on the way in but
    invert it only once on the way out, scrambling every output spatially."""
    model = UViT(patch_size=4, emb_features=64, num_layers=4, num_heads=2, use_hilbert=True)
    x, temb, textcontext = small_inputs(rng)
    out = run_forward(model, rng, x, temb, textcontext)
    assert out.shape == x.shape


def test_hilbert_patchify_roundtrip(rng):
    """patchify returns patches in hilbert order plus the inverse permutation;
    unpatchify with that permutation must be an exact identity."""
    from flaxdiff.models.hilbert import hilbert_patchify, hilbert_unpatchify

    x = jax.random.normal(rng, (2, 16, 16, 3))
    patches, inv_idx = hilbert_patchify(x, 4)
    rec = hilbert_unpatchify(patches, inv_idx, 4, 16, 16, 3)
    assert jnp.allclose(rec, x)


def test_dropout_is_active_in_train_mode(rng):
    """--dropout_rate used to be threaded into every block and applied nowhere."""
    model = SimpleDiT(patch_size=4, emb_features=64, num_layers=2, num_heads=2,
                      mlp_ratio=2, dropout_rate=0.5)
    x, temb, textcontext = small_inputs(rng)
    params = model.init(rng, x, temb, textcontext)
    # nudge every param off init: the zero-initialized adaLN gates and final
    # projection make a fresh DiT output exactly zero, hiding dropout
    params = jax.tree.map(lambda p: p + 0.02, params)

    d0 = model.apply(params, x, temb, textcontext, train=True, rngs={"dropout": jax.random.PRNGKey(1)})
    d1 = model.apply(params, x, temb, textcontext, train=True, rngs={"dropout": jax.random.PRNGKey(2)})
    assert not jnp.allclose(d0, d1), "different dropout rngs must give different outputs"

    e0 = model.apply(params, x, temb, textcontext)
    e1 = model.apply(params, x, temb, textcontext)
    assert jnp.allclose(e0, e1), "eval mode must be deterministic"
