"""Throughput accounting, divergence detection and the profiler hook.

None of the performance work is evaluable without these numbers, so they get
the same treatment as the training maths.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flaxdiff.inputs import DiffusionInputConfig
from flaxdiff.models.simple_dit import SimpleDiT
from flaxdiff.predictors import get_diffusion_preset
from flaxdiff.trainer import GeneralDiffusionTrainer
from flaxdiff.utils import (
    DevicePrefetchIterator, enable_compilation_cache, model_flops_utilization, step_flops,
)

RES = 8
BATCH = 8


def make_trainer(tmp_path, **kwargs):
    train_schedule, _, transform = get_diffusion_preset("edm")
    return GeneralDiffusionTrainer(
        model=SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
        optimizer=optax.adam(1e-3),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image", sample_data_shape=(RES, RES, 3), conditions=[]),
        rngs=jax.random.PRNGKey(0),
        name="instr",
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
        **kwargs,
    )


def batches():
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (BATCH, 1, RES, 3))
    while True:
        yield {"image": images}


def data_dict():
    return {"train": batches, "train_len": BATCH * 8,
            "local_batch_size": BATCH, "global_batch_size": BATCH}


def test_step_flops_reports_a_positive_count(tmp_path):
    trainer = make_trainer(tmp_path)
    step = trainer._define_train_step(batch_size=BATCH)
    source = DevicePrefetchIterator(batches(), trainer.batch_sharding)
    flops = step_flops(step, trainer.state, trainer.rngstate, next(source))
    assert flops is not None and flops > 0


def test_throughput_metrics_are_consistent(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.global_batch_size = 64
    metrics = trainer._throughput_metrics(elapsed=2.0, steps=10)
    assert metrics["train/step_time_ms"] == pytest.approx(200.0)
    assert metrics["train/samples_per_sec"] == pytest.approx(320.0)


def test_throughput_metrics_ignore_a_zero_interval(tmp_path):
    assert make_trainer(tmp_path)._throughput_metrics(elapsed=0.0, steps=0) == {}


def test_mfu_is_skipped_on_unknown_hardware():
    # CPU is deliberately absent from the peak-FLOPs table
    assert model_flops_utilization(1e12, 1.0, 8) is None


def test_mfu_scales_with_time_and_devices(monkeypatch):
    import flaxdiff.utils as utils
    monkeypatch.setitem(utils.PEAK_FLOPS_PER_DEVICE, jax.devices()[0].device_kind, 100.0)
    assert utils.model_flops_utilization(50.0, 1.0, 1) == pytest.approx(0.5)
    assert utils.model_flops_utilization(50.0, 1.0, 2) == pytest.approx(0.25)
    assert utils.model_flops_utilization(50.0, 2.0, 1) == pytest.approx(0.25)


def test_fit_reports_throughput_to_wandb(tmp_path):
    """The logging tick must actually carry the numbers, not just the loss."""
    logged = []

    class FakeWandb:
        def log(self, payload, step=None):
            logged.append(payload)

        def define_metric(self, *args, **kwargs):
            pass

    trainer = make_trainer(tmp_path, log_every=1)
    trainer.wandb = FakeWandb()
    trainer.fit(data_dict(), training_steps_per_epoch=3, epochs=1, val_steps_per_epoch=0)

    ticks = [p for p in logged if "train/samples_per_sec" in p]
    assert ticks, "no throughput was logged"
    assert all(p["train/step_time_ms"] > 0 for p in ticks)
    assert all(p["train/samples_per_sec"] > 0 for p in ticks)


def test_compilation_cache_directory_is_configured(tmp_path):
    path = str(tmp_path / "xla-cache")
    enable_compilation_cache(path)
    assert os.path.isdir(path)
    assert jax.config.jax_compilation_cache_dir == path


def test_profiler_writes_a_trace(tmp_path):
    trainer = make_trainer(tmp_path, profile_steps=2)
    trainer.fit(data_dict(), training_steps_per_epoch=3, epochs=1, val_steps_per_epoch=0)
    assert os.path.isdir(trainer.profile_path())
    assert any(files for _, _, files in os.walk(trainer.profile_path()))


# --------------------------------------------------------------------------
# Divergence
# --------------------------------------------------------------------------

def test_sustained_non_finite_loss_stops_the_run(tmp_path):
    trainer = make_trainer(
        tmp_path, log_every=1, max_bad_loss_steps=3,
        loss_fn=lambda pred, target: jnp.full_like(pred, jnp.nan))
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit(data_dict(), training_steps_per_epoch=8, epochs=1, val_steps_per_epoch=0)


def test_healthy_run_does_not_trip_the_detector(tmp_path):
    trainer = make_trainer(tmp_path, log_every=1, max_bad_loss_steps=3)
    trainer.fit(data_dict(), training_steps_per_epoch=6, epochs=1, val_steps_per_epoch=0)
    assert int(trainer.state.step) == 6
