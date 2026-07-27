"""Sharding, FSDP and data-pipeline tests on a simulated 8-device CPU mesh.

The parity tests are the safety net for the shard_map -> jit + NamedSharding
migration: a partitioned run has to produce the same numbers as a single-device
one, otherwise the collectives GSPMD derived are not the ones we meant.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P

from flaxdiff.inputs import DiffusionInputConfig
from flaxdiff.models.simple_dit import SimpleDiT
from flaxdiff.predictors import get_diffusion_preset
from flaxdiff.trainer import GeneralDiffusionTrainer
from flaxdiff.utils import (
    DevicePrefetchIterator, batch_sharding, build_mesh, parameter_spec, shard_batch,
)

RES = 8
BATCH = 8
# The test model's parameters are far below the production shard threshold, so
# lower it or "FSDP on" would silently mean "everything replicated".
TINY = 256


def make_trainer(tmp_path, name, distributed_training, fsdp_size=1,
                 optimizer=None, **kwargs):
    train_schedule, _, transform = get_diffusion_preset("edm")
    return GeneralDiffusionTrainer(
        model=SimpleDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1),
        optimizer=optax.adam(1e-3) if optimizer is None else optimizer,
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image", sample_data_shape=(RES, RES, 3), conditions=[]),
        rngs=jax.random.PRNGKey(0),
        name=name,
        wandb_config=None,
        distributed_training=distributed_training,
        fsdp_size=fsdp_size,
        checkpoint_base_path=str(tmp_path),
        **kwargs,
    )


def batches():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(BATCH, RES, RES, 3)).astype(np.float32)
    while True:
        yield {"image": images}


def run_losses(trainer, steps):
    """Per-step losses from the real compiled training step."""
    train_step = trainer._define_train_step(batch_size=BATCH)
    source = DevicePrefetchIterator(batches(), trainer.batch_sharding)
    state, rng = trainer.state, trainer.rngstate
    losses = []
    for _ in range(steps):
        state, loss, _, rng, is_finite = train_step(state, rng, next(source))
        assert bool(is_finite)
        losses.append(float(loss))
    return losses


# --------------------------------------------------------------------------
# Sharding heuristic
# --------------------------------------------------------------------------

def test_parameter_spec_replicates_without_fsdp():
    assert parameter_spec((1024, 1024), fsdp_size=1, min_shard_size=16) == P()


def test_parameter_spec_replicates_small_params():
    assert parameter_spec((8, 8), fsdp_size=2, min_shard_size=2 ** 16) == P()


def test_parameter_spec_shards_largest_divisible_axis():
    assert parameter_spec((64, 1024), fsdp_size=2, min_shard_size=16) == P(None, 'fsdp')
    assert parameter_spec((1024, 64), fsdp_size=2, min_shard_size=16) == P('fsdp')


def test_parameter_spec_falls_back_to_replication_when_indivisible():
    assert parameter_spec((15, 15), fsdp_size=2, min_shard_size=16) == P()


def test_build_mesh_rejects_bad_fsdp_size():
    with pytest.raises(ValueError):
        build_mesh(fsdp_size=3)


def test_build_mesh_axes():
    mesh = build_mesh(fsdp_size=2)
    assert mesh.shape['data'] == jax.device_count() // 2
    assert mesh.shape['fsdp'] == 2


# --------------------------------------------------------------------------
# Batch placement and prefetch
# --------------------------------------------------------------------------

def test_shard_batch_splits_across_all_devices():
    mesh = build_mesh(fsdp_size=2)
    batch = {"image": np.zeros((jax.device_count(), 4), np.float32)}
    sharded = shard_batch(batch_sharding(mesh), batch)["image"]
    assert len(sharded.addressable_shards) == jax.device_count()
    assert sharded.addressable_shards[0].data.shape == (1, 4)


def test_prefetch_iterator_preserves_order_and_terminates():
    mesh = build_mesh()
    source = ({"x": np.full((jax.device_count(), 2), i, np.float32)} for i in range(5))
    it = DevicePrefetchIterator(source, batch_sharding(mesh), depth=2)
    seen = [float(np.asarray(b["x"])[0, 0]) for b in it]
    assert seen == [0.0, 1.0, 2.0, 3.0, 4.0]
    with pytest.raises(StopIteration):
        next(it)


def test_prefetch_iterator_surfaces_source_errors():
    mesh = build_mesh()

    def broken():
        yield {"x": np.zeros((jax.device_count(), 2), np.float32)}
        raise ValueError("source exploded")

    it = DevicePrefetchIterator(broken(), batch_sharding(mesh), depth=2)
    next(it)
    with pytest.raises(ValueError, match="source exploded"):
        next(it)


def test_prefetch_iterator_tracks_checkpointable_source_state():
    """The position handed out must be the consumed batch's, not the thread's."""
    import grain.python as pygrain

    loader = pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, 64, 1),
        sampler=pygrain.IndexSampler(num_records=64, shuffle=False, seed=0, num_epochs=1,
                                     shard_options=pygrain.NoSharding()),
        operations=[pygrain.Batch(jax.device_count(), drop_remainder=True)],
        worker_count=0,
    )
    mesh = build_mesh()
    it = DevicePrefetchIterator(iter(loader), batch_sharding(mesh), depth=2)
    next(it)
    next(it)
    state = it.source_state
    expected = np.asarray(next(it))

    resumed = DevicePrefetchIterator(iter(loader), batch_sharding(mesh), depth=2,
                                     source_state=state)
    assert np.array_equal(np.asarray(next(resumed)), expected)


# --------------------------------------------------------------------------
# Numerical parity
# --------------------------------------------------------------------------

def test_single_and_multi_device_losses_agree(tmp_path):
    """The whole point of the migration: partitioning must not change the maths."""
    steps = 20
    single = run_losses(make_trainer(tmp_path / "one", "one", distributed_training=False), steps)
    multi = run_losses(make_trainer(tmp_path / "many", "many", distributed_training=True), steps)
    assert jax.device_count() > 1
    np.testing.assert_allclose(single, multi, rtol=2e-4, atol=2e-5)


def test_fsdp_losses_match_replicated(tmp_path):
    """Sharding the parameters must not change the loss trajectory."""
    steps = 20
    replicated = run_losses(
        make_trainer(tmp_path / "dp", "dp", distributed_training=True, fsdp_size=1), steps)
    fsdp = make_trainer(tmp_path / "fsdp", "fsdp", distributed_training=True,
                        fsdp_size=2, fsdp_min_param_size=TINY)
    assert any('fsdp' in str(x.sharding.spec) for x in jax.tree.leaves(fsdp.state.params))
    np.testing.assert_allclose(replicated, run_losses(fsdp, steps), rtol=2e-4, atol=2e-5)


# --------------------------------------------------------------------------
# FSDP actually shards
# --------------------------------------------------------------------------

def test_fsdp_shards_parameters_and_optimizer_state(tmp_path):
    trainer = make_trainer(tmp_path, "fsdp-shapes", distributed_training=True,
                           fsdp_size=2, fsdp_min_param_size=TINY)
    leaves = jax.tree.leaves(trainer.state.params)
    sharded = [x for x in leaves if 'fsdp' in str(x.sharding.spec)]
    assert sharded, "no parameter was sharded over the fsdp axis"

    for param in sharded:
        biggest = max(param.shape)
        local = param.addressable_shards[0].data
        assert local.size == param.size // 2, "shard is not half the global param"
        assert biggest // 2 in local.shape

    # Adam moments and the EMA copy must follow the params they track, without
    # the optimizer or the model ever describing a layout.
    mu = trainer.state.opt_state[0].mu
    param_specs = [x.sharding.spec for x in jax.tree.leaves(trainer.state.params)]
    mu_specs = [x.sharding.spec for x in jax.tree.leaves(mu)]
    assert param_specs == mu_specs

    ema_specs = [x.sharding.spec for x in jax.tree.leaves(trainer.state.ema_params)]
    assert param_specs == ema_specs


def test_replicated_run_shards_nothing(tmp_path):
    trainer = make_trainer(tmp_path, "dp-shapes", distributed_training=True, fsdp_size=1)
    for leaf in jax.tree.leaves(trainer.state.params):
        assert leaf.sharding.spec == P()


# --------------------------------------------------------------------------
# Checkpointing under sharding
# --------------------------------------------------------------------------

def test_sharded_checkpoint_roundtrips(tmp_path):
    trainer = make_trainer(tmp_path, "ckpt", distributed_training=True,
                           fsdp_size=2, fsdp_min_param_size=TINY)
    grads = jax.tree.map(jnp.ones_like, trainer.state.params)
    trainer.state = trainer.state.apply_gradients(grads=grads).apply_ema(0.99)
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    restored = make_trainer(tmp_path, "ckpt", distributed_training=True, fsdp_size=2,
                            fsdp_min_param_size=TINY,
                            load_from_checkpoint=trainer.checkpoint_path())
    assert int(restored.state.step) == 1
    for before, after in zip(jax.tree.leaves(trainer.state.params),
                             jax.tree.leaves(restored.state.params)):
        assert before.sharding.spec == after.sharding.spec
        np.testing.assert_allclose(np.asarray(before), np.asarray(after))


def test_gradient_accumulation_updates_only_on_the_boundary(tmp_path):
    """MultiSteps must hold the params still until k micro-batches have run.

    Also covers the accumulator surviving the sharding heuristic: its buffers
    are param-shaped, so they pick up the param specs.
    """
    accum = 3
    trainer = make_trainer(tmp_path, "accum", distributed_training=True, fsdp_size=2,
                           fsdp_min_param_size=TINY,
                           optimizer=optax.MultiSteps(optax.sgd(0.5), every_k_schedule=accum))

    train_step = trainer._define_train_step(batch_size=BATCH)
    source = DevicePrefetchIterator(batches(), trainer.batch_sharding)
    state, rng = trainer.state, trainer.rngstate

    def snapshot(s):
        return [np.asarray(x).copy() for x in jax.tree.leaves(s.params)]

    reference = snapshot(state)
    for micro in range(1, accum * 2 + 1):
        state, _, _, rng, _ = train_step(state, rng, next(source))
        moved = any(not np.array_equal(a, b) for a, b in zip(reference, snapshot(state)))
        at_boundary = micro % accum == 0
        assert moved == at_boundary, f"micro-step {micro}: moved={moved}"
        if at_boundary:
            reference = snapshot(state)


def grain_image_loader(num_records=256):
    """A checkpointable source that yields distinguishable image batches."""
    import grain.python as pygrain

    class ToImage(pygrain.MapTransform):
        def map(self, index):
            return {"image": np.full((RES, RES, 3), index, np.float32)}

    return pygrain.DataLoader(
        data_source=pygrain.RangeDataSource(0, num_records, 1),
        sampler=pygrain.IndexSampler(num_records=num_records, shuffle=False, seed=0,
                                     num_epochs=4, shard_options=pygrain.NoSharding()),
        operations=[ToImage(), pygrain.Batch(BATCH, drop_remainder=True)],
        worker_count=0,
    )


def test_resume_continues_mid_epoch(tmp_path):
    """A resumed run must not replay the batches it already trained on."""
    steps = 3
    data = {"train": grain_image_loader, "train_len": BATCH * 64,
            "local_batch_size": BATCH, "global_batch_size": BATCH}

    trainer = make_trainer(tmp_path, "resume", distributed_training=True)
    trainer.fit(data, training_steps_per_epoch=steps, epochs=1, val_steps_per_epoch=0)
    assert trainer.dataset_state is not None, "iterator position was never captured"
    trainer.wait_for_checkpoints()

    resumed = make_trainer(tmp_path, "resume", distributed_training=True,
                           load_from_checkpoint=trainer.checkpoint_path())
    assert resumed.dataset_state == trainer.dataset_state

    # The next batch after resuming is the one the first run would have seen next
    reference = DevicePrefetchIterator(
        iter(grain_image_loader()), trainer.batch_sharding,
        source_state=trainer.dataset_state)
    resumed_iter = DevicePrefetchIterator(
        iter(grain_image_loader()), resumed.batch_sharding,
        source_state=resumed.dataset_state)
    np.testing.assert_array_equal(np.asarray(next(resumed_iter)["image"]),
                                  np.asarray(next(reference)["image"]))

    # and it is past the batches already consumed
    fresh = DevicePrefetchIterator(iter(grain_image_loader()), resumed.batch_sharding)
    first = np.asarray(next(fresh)["image"])
    assert not np.array_equal(np.asarray(next(resumed_iter)["image"]), first)


def test_load_tolerates_checkpoints_without_iterator_state(tmp_path):
    """Checkpoints written before iterator tracking must still restore."""
    trainer = make_trainer(tmp_path, "legacy", distributed_training=True)
    assert trainer.dataset_state is None
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    restored = make_trainer(tmp_path, "legacy", distributed_training=True,
                            load_from_checkpoint=trainer.checkpoint_path())
    assert int(restored.state.step) == 0
    assert restored.dataset_state is None


def test_checkpoint_restores_onto_a_different_mesh(tmp_path):
    """A run saved with FSDP must be resumable on a replicated mesh."""
    trainer = make_trainer(tmp_path, "mesh", distributed_training=True,
                           fsdp_size=2, fsdp_min_param_size=TINY)
    grads = jax.tree.map(jnp.ones_like, trainer.state.params)
    trainer.state = trainer.state.apply_gradients(grads=grads)
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    restored = make_trainer(tmp_path, "mesh", distributed_training=True, fsdp_size=1,
                            load_from_checkpoint=trainer.checkpoint_path())
    assert int(restored.state.step) == 1
    for leaf in jax.tree.leaves(restored.state.params):
        assert leaf.sharding.spec == P()
    for before, after in zip(jax.tree.leaves(trainer.state.params),
                             jax.tree.leaves(restored.state.params)):
        np.testing.assert_allclose(np.asarray(before), np.asarray(after))
