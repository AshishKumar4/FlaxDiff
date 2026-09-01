"""Trainer smoke tests: training without wandb, and checkpoint save/restore.

These run the real GeneralDiffusionTrainer on CPU with a tiny DiT and an
unconditional synthetic dataset - the same code path a real run takes, minus
wandb and conditioning encoders.
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
from flaxdiff.trainer import general_diffusion_trainer as gdt
from flaxdiff.utils import get_latest_checkpoint

RES = 8


def make_trainer(tmp_path, name="smoke", load_from_checkpoint=None):
    train_schedule, _, transform = get_diffusion_preset("edm")
    return GeneralDiffusionTrainer(
        model=SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
        optimizer=optax.adam(1e-3),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image",
            sample_data_shape=(RES, RES, 3),
            conditions=[],
        ),
        rngs=jax.random.PRNGKey(0),
        name=name,
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
        load_from_checkpoint=load_from_checkpoint,
    )


def batch_iterator():
    # uint8-range images, as the data pipeline provides them
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None], (8, 1, RES, 3))
    while True:
        yield {"image": jnp.asarray(images)}


def test_fit_without_wandb(tmp_path):
    trainer = make_trainer(tmp_path)
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    state = trainer.fit(data, training_steps_per_epoch=4, epochs=1, val_steps_per_epoch=0)
    assert state is not None
    assert int(state.step) == 4


def test_save_writes_checkpoint(tmp_path):
    """The checkpoint has to land under the step it was asked for."""
    trainer = make_trainer(tmp_path)
    trainer.save(epoch=0, step=1)
    # Saving is async to keep it off the training loop; this is the barrier
    trainer.wait_for_checkpoints()
    assert trainer.checkpointer.latest_step() == 1


def test_restore_preserves_optimizer_state(tmp_path):
    trainer = make_trainer(tmp_path)

    # One real update so the adam moments, step counter and EMA are non-trivial
    grads = jax.tree.map(jnp.ones_like, trainer.state.params)
    trainer.state = trainer.state.apply_gradients(grads=grads).apply_ema(0.99)
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    restored = make_trainer(tmp_path, load_from_checkpoint=trainer.checkpoint_path())
    assert int(restored.state.step) == 1, "optimizer step counter was reset"

    old_opt = jax.tree.leaves(trainer.state.opt_state)
    new_opt = jax.tree.leaves(restored.state.opt_state)
    assert all(np.allclose(a, b) for a, b in zip(old_opt, new_opt)), "adam moments were reset"

    old_ema = jax.tree.leaves(trainer.state.ema_params)
    new_ema = jax.tree.leaves(restored.state.ema_params)
    assert all(np.allclose(a, b) for a, b in zip(old_ema, new_ema)), "ema params were reset"


def test_fit_video_model(tmp_path):
    """Video end to end: 5D batches through the real trainer and the video DiT."""
    from flaxdiff.models.video_dit import VideoDiT

    train_schedule, _, transform = get_diffusion_preset("edm")
    trainer = GeneralDiffusionTrainer(
        model=VideoDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1),
        optimizer=optax.adam(1e-3),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="video",
            sample_data_shape=(3, RES, RES, 3),
            conditions=[],
        ),
        rngs=jax.random.PRNGKey(0),
        name="video-smoke",
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
    )

    def video_batches():
        frames = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, None, :, None, None],
                         (4, 3, 1, RES, 3))
        while True:
            yield {"video": jnp.asarray(frames)}

    data = {"train": video_batches, "train_len": 16, "local_batch_size": 4}
    state = trainer.fit(data, training_steps_per_epoch=2, epochs=1, val_steps_per_epoch=0)
    assert int(state.step) == 2


# --------------------------------------------------------------------------
# The end-of-run checkpoint
# --------------------------------------------------------------------------

def test_fit_checkpoints_the_final_step(tmp_path):
    """The last save of a run must carry the step the run ended on.

    The in-loop save only fires on an epoch that improved the best loss, so
    pinning the best loss out of reach leaves the end-of-run save as the only
    one - which is the case where it used to write the final weights under
    step 0, behind whatever older checkpoint a resume would then pick.
    """
    trainer = make_trainer(tmp_path)
    trainer.best_loss = -1.0
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    trainer.fit(data, training_steps_per_epoch=2, epochs=2, val_steps_per_epoch=0)

    assert trainer.checkpointer.latest_step() == 4
    assert set(trainer.checkpointer.all_steps()) == {4}
    assert not os.path.exists(os.path.join(trainer.checkpoint_path(), "0"))


def test_fit_skips_the_final_save_when_the_loop_already_wrote_that_step(tmp_path):
    trainer = make_trainer(tmp_path, name="skip")
    written = []
    real_save = trainer.save

    def spy(epoch=0, step=0, state=None, rngstate=None):
        written.append(step)
        return real_save(epoch=epoch, step=step, state=state, rngstate=rngstate)

    trainer.save = spy
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    trainer.fit(data, training_steps_per_epoch=4, epochs=1, val_steps_per_epoch=0)

    # One save, from the epoch's best-loss branch; the tail must not add a second
    assert written == [4]


def test_fit_that_never_trains_checkpoints_step_zero(tmp_path):
    """A run that really ends at step 0 is the one case where step 0 is honest."""
    trainer = make_trainer(tmp_path, name="nosteps")
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    trainer.fit(data, training_steps_per_epoch=2, epochs=0, val_steps_per_epoch=0)

    assert trainer.checkpointer.latest_step() == 0


def test_get_latest_checkpoint_ignores_non_step_entries(tmp_path):
    """Orbax leaves locks, metadata and interrupted tmp directories behind, and
    a step directory can exist while still being empty."""
    for name in ("1", "10", "2"):
        os.makedirs(tmp_path / name)
        (tmp_path / name / "manifest.ocdbt").write_bytes(b"x")
    os.makedirs(tmp_path / "11")                       # written, then interrupted
    os.makedirs(tmp_path / "12.orbax-checkpoint-tmp")
    (tmp_path / "_CHECKPOINT_METADATA").write_text("{}")
    (tmp_path / "descriptor").write_text("")

    assert get_latest_checkpoint(str(tmp_path)) == str(tmp_path / "10")


def test_get_latest_checkpoint_reports_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_latest_checkpoint(str(tmp_path))


class ExplodingCheckpointer:
    """Orbax when the filesystem refuses the write.

    Stubbed rather than provoked with a read-only directory: a genuinely
    failed *async* orbax write leaves a background thread that never joins,
    which hangs interpreter exit and with it the whole test session.
    """

    def save(self, *args, **kwargs):
        raise OSError("No space left on device")

    def wait_until_finished(self):
        pass


def test_save_propagates_persistence_failures(tmp_path):
    """A checkpoint that did not get written is data loss, not a log line."""
    trainer = make_trainer(tmp_path, name="unwritable")
    trainer.checkpointer = ExplodingCheckpointer()
    with pytest.raises(OSError):
        trainer.save(epoch=0, step=1)
    assert trainer.last_saved_step is None


def test_registry_failure_leaves_the_checkpoint_alone(tmp_path, monkeypatch):
    """Publishing to wandb is reporting, not persistence: it may fail, but it
    must neither abort the save nor delete the local checkpoint."""
    trainer = make_trainer(tmp_path, name="registry")
    trainer.wandb = object()  # truthy: enough to enter the publishing branch

    def unavailable(_path):
        raise FileNotFoundError("checkpoint already handed to the registry")

    monkeypatch.setattr(gdt, "get_latest_checkpoint", unavailable)
    trainer.save(epoch=0, step=3)
    trainer.wait_for_checkpoints()

    assert trainer.checkpointer.latest_step() == 3
    assert os.path.exists(os.path.join(trainer.checkpoint_path(), "3"))
