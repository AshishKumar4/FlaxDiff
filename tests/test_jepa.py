"""I-JEPA and V-JEPA: masking, shapes, the objective, and collapse telemetry."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flaxdiff.inputs import DiffusionInputConfig
from flaxdiff.jepa import (
    JepaEncoder, JepaVideoEncoder, JepaObjective, multi_block_mask,
    representation_health, normalize_targets, linear_probe, knn_probe,
)
from flaxdiff.jepa.models import JepaPredictor
from flaxdiff.trainer import GeneralDiffusionTrainer

RES = 32
PATCH = 4
GRID = (RES // PATCH, RES // PATCH)
FRAMES = 3


@pytest.fixture
def mask():
    return multi_block_mask(GRID, num_targets=4, scale=(0.15, 0.2))


def make_encoder(**kwargs):
    return JepaEncoder(patch_size=PATCH, emb_features=32, num_layers=2, num_heads=2,
                       mlp_ratio=2, **kwargs)


def make_predictor(**kwargs):
    return JepaPredictor(grid=GRID, emb_features=32, predictor_features=16,
                         num_layers=1, num_heads=2, mlp_ratio=2, **kwargs)


def make_objective(mask, **kwargs):
    return JepaObjective(make_encoder(), make_predictor(), mask,
                         sample_data_key="image", sample_data_shape=(RES, RES, 3), **kwargs)


def images(seed=0, batch=4):
    return jnp.asarray(np.random.RandomState(seed).uniform(0, 255, (batch, RES, RES, 3)))


def videos(seed=0, batch=2):
    return jnp.asarray(
        np.random.RandomState(seed).uniform(0, 255, (batch, FRAMES, RES, RES, 3)))


# --- masking ---------------------------------------------------------------

def test_context_and_targets_are_disjoint(mask):
    context_idx, target_idx = mask.sample(jax.random.PRNGKey(3), 8)
    for b in range(8):
        context = set(np.array(context_idx[b]).tolist())
        targets = set(np.array(target_idx[b]).reshape(-1).tolist())
        assert not (context & targets), "context encoder can see a target patch"
        assert max(targets) < mask.num_patches and min(context) >= 0


def test_target_coverage_sits_inside_the_configured_scale():
    scale = (0.15, 0.2)
    mask = multi_block_mask(GRID, num_targets=4, scale=scale)
    coverage = mask.block_area / mask.num_patches
    assert scale[0] <= coverage <= scale[1]
    for h, w in mask.block_shapes:
        assert h * w == mask.block_area
        assert 0.75 <= h / w <= 1.5


def test_masks_are_reproducible_from_a_seed(mask):
    a = mask.sample(jax.random.PRNGKey(11), 4)
    b = mask.sample(jax.random.PRNGKey(11), 4)
    c = mask.sample(jax.random.PRNGKey(12), 4)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b))
    assert not all(jnp.array_equal(x, y) for x, y in zip(a, c))


def test_block_shapes_and_positions_actually_vary(mask):
    blocks = np.array(mask.sample(jax.random.PRNGKey(5), 64)[1]).reshape(-1, mask.block_area)
    corners = {int(block.min()) for block in blocks}
    widths = {len(np.unique(block % GRID[1])) for block in blocks}
    assert len(corners) > 1, "every block landed in the same place"
    assert len(widths) > 1, "every block came out the same shape"


def test_geometry_that_cannot_exist_is_rejected():
    with pytest.raises(ValueError, match="aspect ratio"):
        multi_block_mask((4, 4), num_targets=2, scale=(0.18, 0.19))
    with pytest.raises(ValueError, match="no context"):
        multi_block_mask(GRID, num_targets=6, scale=(0.15, 0.2))


# --- forward shapes --------------------------------------------------------

def test_image_encoder_shapes(mask, rng):
    encoder = make_encoder()
    context_idx, _ = mask.sample(rng, 4)
    x = images()
    params = encoder.init(rng, x, context_idx)

    assert encoder.apply(params, x, context_idx).shape == (4, mask.num_context, 32)
    assert encoder.apply(params, x).shape == (4, mask.num_patches, 32)


def test_image_predictor_shapes(mask, rng):
    encoder, predictor = make_encoder(), make_predictor()
    context_idx, target_idx = mask.sample(rng, 4)
    x = images()
    encoder_params = encoder.init(rng, x, context_idx)
    context = encoder.apply(encoder_params, x, context_idx)

    block = target_idx[:, 0]
    params = predictor.init(rng, context, context_idx, block)
    assert predictor.apply(params, context, context_idx, block).shape == (4, mask.block_area, 32)


def test_video_encoder_and_predictor_shapes(mask, rng):
    encoder = JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1,
                               num_heads=2, mlp_ratio=2)
    predictor = make_predictor(factorized=True)
    context_idx, target_idx = mask.sample(rng, 2)
    x = videos()

    encoder_params = encoder.init(rng, x, context_idx)
    context = encoder.apply(encoder_params, x, context_idx)
    assert context.shape == (2, FRAMES, mask.num_context, 32)
    assert encoder.apply(encoder_params, x).shape == (2, FRAMES, mask.num_patches, 32)

    block = target_idx[:, 0]
    params = predictor.init(rng, context, context_idx, block)
    out = predictor.apply(params, context, context_idx, block)
    assert out.shape == (2, FRAMES, mask.block_area, 32)


def test_video_encoder_carries_a_temporal_signal(mask, rng):
    """Tubelet masking keeps the spatial layout, so time is the only axis the
    encoder can mix along: perturbing frame 0 must move frame 2's embedding."""
    encoder = JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1,
                               num_heads=2, mlp_ratio=2)
    context_idx, _ = mask.sample(rng, 2)
    x = videos()
    params = jax.tree.map(lambda p: p + 0.02, encoder.init(rng, x, context_idx))

    before = encoder.apply(params, x, context_idx)
    after = encoder.apply(params, x.at[:, 0].add(1.0), context_idx)
    assert not jnp.allclose(before[:, 2], after[:, 2]), "no information flow across frames"


def test_encoder_runs_on_the_ssm_mixer(mask, rng):
    """The hybrid SSM encoder is the point of the shared block: same interface."""
    encoder = make_encoder(ssm_attention_ratio="3:1", ssm_state_dim=8)
    context_idx, _ = mask.sample(rng, 4)
    x = images()
    params = encoder.init(rng, x, context_idx)
    out = encoder.apply(params, x, context_idx)
    assert out.shape == (4, mask.num_context, 32)
    assert jnp.all(jnp.isfinite(out))


# --- the objective ---------------------------------------------------------

def test_fresh_loss_is_non_trivial_and_training_reduces_it(mask, rng):
    objective = make_objective(mask)
    params = objective.init_params(rng)
    batch = {"image": images()}

    def loss_of(p):
        return objective.loss(p, params, batch, jax.random.PRNGKey(7), 0)[0]

    initial = float(loss_of(params))
    assert initial > 0.1, "a fresh model already predicts the targets"

    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)
    for _ in range(20):
        grads = jax.grad(loss_of)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    assert float(loss_of(params)) < initial * 0.9, "the objective does not train"


def test_video_objective_trains(mask, rng):
    objective = JepaObjective(
        JepaVideoEncoder(patch_size=PATCH, emb_features=32, num_layers=1,
                         num_heads=2, mlp_ratio=2),
        make_predictor(factorized=True), mask,
        sample_data_key="video", sample_data_shape=(FRAMES, RES, RES, 3))
    params = objective.init_params(rng)
    batch = {"video": videos()}

    def loss_of(p):
        return objective.loss(p, params, batch, jax.random.PRNGKey(7), 0)[0]

    initial = float(loss_of(params))
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)
    for _ in range(15):
        grads = jax.grad(loss_of)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    assert float(loss_of(params)) < initial * 0.9


def solid_colour_images(rs, n):
    """One random colour per image, so a target block is fully determined by
    anything else in the same image and by nothing in any other image."""
    colours = rs.uniform(20, 235, (n, 1, 1, 3))
    return jnp.asarray(np.clip(colours + rs.uniform(-15, 15, (n, RES, RES, 3)), 0, 255))


def context_ablation(objective, params, ema, data, mask, rng):
    """Prediction error from the true context vs. from another image's context."""
    n, blocks = data.shape[0], mask.num_targets
    context_idx, target_idx = mask.sample(rng, n)
    full = normalize_targets(objective.encode(ema["params"]["context_encoder"], data))
    targets = jnp.take_along_axis(
        full[:, None], target_idx.reshape(n, blocks, -1, 1), axis=-2)
    context = objective.encode(params["params"]["context_encoder"], data, context_idx)

    def error(ctx):
        predictions = objective.predictor.apply(
            {"params": params["params"]["predictor"]},
            jnp.repeat(ctx, blocks, axis=0),
            jnp.repeat(context_idx, blocks, axis=0),
            target_idx.reshape(n * blocks, -1),
        ).reshape(targets.shape)
        return float(jnp.mean((predictions - targets) ** 2))

    return error(context), error(jnp.roll(context, 1, axis=0))


def test_training_makes_the_prediction_depend_on_the_context(tmp_path, mask):
    """The loss can be driven down by predicting the average target while
    ignoring the context entirely - which would teach the encoder nothing.
    Swapping in another image's context must cost real accuracy."""
    rs = np.random.RandomState(0)
    train_x, test_x = solid_colour_images(rs, 128), solid_colour_images(rs, 32)
    normalized_test = (test_x - 127.5) / 127.5

    trainer = make_jepa_trainer(tmp_path, mask, learning_rate=3e-3,
                                momentum=(0.9, 0.99), momentum_steps=150)
    objective = trainer.objective
    before = context_ablation(objective, trainer.state.params, trainer.state.ema_params,
                              normalized_test, mask, jax.random.PRNGKey(9))
    assert before[1] / before[0] < 1.5, "a fresh predictor should not favour any context"

    def batches():
        while True:
            yield {"image": train_x[rs.randint(0, len(train_x), 16)]}

    state = trainer.fit({"train": batches, "train_len": 128, "local_batch_size": 16},
                        training_steps_per_epoch=150, epochs=1, val_steps_per_epoch=0)

    after = context_ablation(objective, state.params, state.ema_params,
                             normalized_test, mask, jax.random.PRNGKey(9))
    assert after[0] < before[0] / 2, "held-out prediction error did not improve"
    assert after[1] / after[0] > 5.0, "the predictor still ignores its context"


def test_no_gradient_reaches_the_target_branch(mask, rng):
    """stop_gradient on the target encoder is what stops the trivial solution."""
    objective = make_objective(mask)
    params = objective.init_params(rng)
    batch = {"image": images()}

    grads = jax.grad(
        lambda ema: objective.loss(params, ema, batch, jax.random.PRNGKey(7), 0)[0]
    )(params)
    assert all(float(jnp.max(jnp.abs(g))) == 0.0 for g in jax.tree.leaves(grads))


# --- collapse telemetry ----------------------------------------------------

def test_representation_std_detects_collapse():
    healthy = jax.random.normal(jax.random.PRNGKey(0), (32, 16))
    collapsed = jnp.tile(healthy[:1], (32, 1))

    assert float(representation_health(healthy)["repr_std"]) > 0.5
    assert float(representation_health(collapsed)["repr_std"]) < 1e-6


def test_collapse_telemetry_flows_through_a_degenerate_encoder(mask, rng):
    """A constant encoder is the failure mode; the aux dict must show it."""
    class ConstantEncoder(JepaEncoder):
        def __call__(self, x, token_idx=None, train: bool = False):
            out = super().__call__(x, token_idx, train)
            return jnp.zeros_like(out) + jnp.arange(out.shape[-1], dtype=out.dtype)

    batch = {"image": images()}
    collapsed = JepaObjective(
        ConstantEncoder(patch_size=PATCH, emb_features=32, num_layers=2, num_heads=2,
                        mlp_ratio=2),
        make_predictor(), mask, "image", (RES, RES, 3))
    collapsed_params = collapsed.init_params(rng)
    _, collapsed_aux = collapsed.loss(
        collapsed_params, collapsed_params, batch, jax.random.PRNGKey(7), 0)

    healthy = make_objective(mask)
    healthy_params = healthy.init_params(rng)
    _, healthy_aux = healthy.loss(
        healthy_params, healthy_params, batch, jax.random.PRNGKey(7), 0)

    assert float(collapsed_aux["repr_std"]) < 1e-5
    assert float(healthy_aux["repr_std"]) > 1e-3


def test_offdiagonal_covariance_rises_with_redundant_dimensions():
    base = jax.random.normal(jax.random.PRNGKey(0), (64, 1))
    redundant = jnp.tile(base, (1, 8))                       # every dim identical
    independent = jax.random.normal(jax.random.PRNGKey(1), (64, 8))
    assert (float(representation_health(redundant)["repr_cov_offdiag"])
            > 5 * float(representation_health(independent)["repr_cov_offdiag"]))


# --- EMA target encoder ----------------------------------------------------

def test_momentum_schedule_endpoints(mask):
    objective = make_objective(mask, momentum=(0.996, 1.0), momentum_steps=1000)
    assert float(objective.ema.decay(0)) == pytest.approx(0.996)
    assert float(objective.ema.decay(1000)) == pytest.approx(1.0)
    assert objective.ema.path == ("params", "context_encoder")


def make_jepa_trainer(tmp_path, mask, learning_rate=1e-3, **kwargs):
    encoder = make_encoder()
    return GeneralDiffusionTrainer(
        model=encoder,
        optimizer=optax.adam(learning_rate),
        input_config=DiffusionInputConfig(
            sample_data_key="image", sample_data_shape=(RES, RES, 3), conditions=[]),
        rngs=jax.random.PRNGKey(0),
        objective=JepaObjective(encoder, make_predictor(), mask, "image",
                                (RES, RES, 3), **kwargs),
        name="jepa-smoke", wandb_config=None, distributed_training=False,
        checkpoint_base_path=str(tmp_path),
    )


def test_target_encoder_tracks_the_context_encoder(tmp_path, mask):
    trainer = make_jepa_trainer(tmp_path, mask, momentum=(0.5, 0.5), momentum_steps=1)
    # Copied to the host: the train step donates the state, so the device
    # buffers behind this reference are gone once fit has run.
    initial = jax.tree.map(np.asarray, trainer.state.ema_params)

    def batches():
        while True:
            yield {"image": images()}

    state = trainer.fit({"train": batches, "train_len": 16, "local_batch_size": 4},
                        training_steps_per_epoch=3, epochs=1, val_steps_per_epoch=0)

    context_moved = any(
        not np.allclose(a, b) for a, b in zip(
            jax.tree.leaves(state.ema_params["params"]["context_encoder"]),
            jax.tree.leaves(initial["params"]["context_encoder"])))
    predictor_frozen = all(
        np.allclose(a, b) for a, b in zip(
            jax.tree.leaves(state.ema_params["params"]["predictor"]),
            jax.tree.leaves(initial["params"]["predictor"])))
    assert context_moved, "the target encoder never followed the context encoder"
    assert predictor_frozen, "EMA leaked outside the context encoder subtree"

    # and it followed rather than jumped: still between where it started and now
    ema = jax.tree.leaves(state.ema_params["params"]["context_encoder"])
    live = jax.tree.leaves(state.params["params"]["context_encoder"])
    assert any(not np.allclose(a, b) for a, b in zip(ema, live)), "EMA is not lagging"


def test_validation_step_returns_pooled_embeddings(tmp_path, mask):
    trainer = make_jepa_trainer(tmp_path, mask)
    embed = trainer._define_validation_step()
    out = embed(trainer.state, {"image": images()})
    assert out.shape == (4, 32)


# --- probes ----------------------------------------------------------------

def separable_embeddings(num_classes=4, per_class=8, dim=6, noise=0.05):
    rng = np.random.RandomState(0)
    centers = rng.normal(size=(num_classes, dim)) * 3
    labels = np.repeat(np.arange(num_classes), per_class)
    x = centers[labels] + rng.normal(size=(len(labels), dim)) * noise
    order = rng.permutation(len(labels))
    return jnp.asarray(x[order], dtype=jnp.float32), jnp.asarray(labels[order])


def test_probes_separate_clustered_embeddings():
    x, y = separable_embeddings()
    assert float(linear_probe(x, y, num_classes=4, steps=200)) > 0.9
    assert float(knn_probe(x, y, num_classes=4, k=3)) > 0.9


def test_probes_are_at_chance_on_noise():
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 6))
    y = jnp.asarray(np.random.RandomState(1).randint(0, 4, 64))
    assert float(knn_probe(x, y, num_classes=4, k=5)) < 0.6


# --- registry and entrypoint ----------------------------------------------

@pytest.mark.parametrize("architecture", ["jepa_encoder", "jepa_video_encoder"])
def test_registry_builds_the_jepa_models(architecture):
    from flaxdiff.models.registry import build_model
    model = build_model(f"{architecture}+hilbert",
                        {"patch_size": PATCH, "emb_features": 32, "num_layers": 1})
    assert model.emb_features == 32 and model.scan_order == 'hilbert'


def test_training_entrypoint_runs_end_to_end(tmp_path, monkeypatch):
    """Registry -> mask -> objective -> trainer -> probes, as the CLI wires them."""
    import training_jepa

    monkeypatch.setenv("WANDB_MODE", "disabled")
    classes = 4

    def fake_dataset(*args, **kwargs):
        def batches():
            rs = np.random.RandomState(0)
            while True:
                yield {"image": jnp.asarray(rs.uniform(0, 255, (4, RES, RES, 3))),
                       "label": jnp.asarray(rs.randint(0, classes, 4))}
        return {"train": batches, "val": batches, "train_len": 16, "local_batch_size": 4}

    monkeypatch.setattr(training_jepa, "get_dataset_grain", fake_dataset)
    args = training_jepa.parser.parse_args([
        '--image_size', str(RES), '--patch_size', str(PATCH), '--batch_size', '4',
        '--emb_features', '32', '--num_layers', '1', '--num_heads', '2', '--mlp_ratio', '2',
        '--predictor_features', '16', '--predictor_layers', '1', '--predictor_heads', '2',
        '--epochs', '1', '--steps_per_epoch', '2', '--val_steps_per_epoch', '1',
        '--probe_classes', str(classes), '--distributed_training', 'False',
        '--ssm_attention_ratio', '3:1', '--ssm_state_dim', '8',
        '--checkpoint_dir', str(tmp_path),
    ])
    training_jepa.main(args)
