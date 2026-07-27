"""Rematerialization must be invisible to everything except memory use.

Same parameter tree, same forward values, same gradients - otherwise --remat
would silently invalidate checkpoints or change what a run converges to.
"""

import jax
import jax.numpy as jnp
import pytest

from flaxdiff.models.simple_dit import SimpleDiT
from flaxdiff.models.simple_mmdit import SimpleMMDiT, HierarchicalMMDiT
from flaxdiff.models.simple_vit import SimpleUDiT
from flaxdiff.models.ssm_dit import HybridSSMAttentionDiT
from flaxdiff.models.video_dit import VideoDiT

RES = 32

BUILDERS = {
    'simple_dit': lambda remat: SimpleDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
    'simple_udit': lambda remat: SimpleUDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
    'simple_mmdit': lambda remat: SimpleMMDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
    'hierarchical_mmdit': lambda remat: HierarchicalMMDiT(
        base_patch_size=2, emb_features=(32, 64, 96), num_layers=(1, 1, 1),
        num_heads=(2, 2, 2), mlp_ratio=2, remat=remat),
    'hybrid_dit': lambda remat: HybridSSMAttentionDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
}


def image_inputs(rng):
    return (jax.random.normal(rng, (2, RES, RES, 3)), jnp.ones((2,)),
            jnp.ones((2, 77, 768), jnp.float32))


@pytest.mark.parametrize('arch', sorted(BUILDERS))
def test_remat_keeps_parameter_tree_identical(rng, arch):
    x, temb, ctx = image_inputs(rng)
    plain = BUILDERS[arch](False).init(rng, x, temb, ctx)
    remat = BUILDERS[arch](True).init(rng, x, temb, ctx)

    def paths(tree):
        return [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(tree)]

    assert paths(plain) == paths(remat), "remat changed the checkpoint layout"


@pytest.mark.parametrize('arch', sorted(BUILDERS))
def test_remat_preserves_outputs_and_gradients(rng, arch):
    x, temb, ctx = image_inputs(rng)
    plain, remat = BUILDERS[arch](False), BUILDERS[arch](True)
    params = plain.init(rng, x, temb, ctx)

    def loss(model, p):
        return jnp.sum(model.apply(p, x, temb, ctx) ** 2)

    assert jnp.allclose(loss(plain, params), loss(remat, params), rtol=1e-5, atol=1e-4)

    g_plain = jax.grad(lambda p: loss(plain, p))(params)
    g_remat = jax.grad(lambda p: loss(remat, p))(params)
    for a, b in zip(jax.tree.leaves(g_plain), jax.tree.leaves(g_remat)):
        assert jnp.allclose(a, b, rtol=1e-3, atol=1e-4)


def test_video_dit_remat_matches():
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 3, 16, 16, 3))
    temb, ctx = jnp.ones((1,)), jnp.ones((1, 77, 768), jnp.float32)
    plain = VideoDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1)
    remat = VideoDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1,
                     remat=True)
    params = plain.init(rng, x, temb, ctx)
    assert jnp.allclose(plain.apply(params, x, temb, ctx),
                        remat.apply(params, x, temb, ctx), rtol=1e-5, atol=1e-4)
