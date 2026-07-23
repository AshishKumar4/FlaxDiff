"""Trainer smoke tests: training without wandb, and checkpoint save/restore.

These run the real GeneralDiffusionTrainer on CPU with a tiny DiT and an
unconditional synthetic dataset - the same code path a real run takes, minus
wandb and conditioning encoders.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flaxdiff.inputs import DiffusionInputConfig
from flaxdiff.models.simple_dit import SimpleDiT
from flaxdiff.predictors import get_diffusion_preset
from flaxdiff.trainer import GeneralDiffusionTrainer

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
    """save() swallows exceptions, so assert the checkpoint actually landed."""
    trainer = make_trainer(tmp_path)
    trainer.save(epoch=0, step=1)
    assert trainer.checkpointer.latest_step() == 1


def test_restore_preserves_optimizer_state(tmp_path):
    trainer = make_trainer(tmp_path)

    # One real update so the adam moments, step counter and EMA are non-trivial
    grads = jax.tree.map(jnp.ones_like, trainer.state.params)
    trainer.state = trainer.state.apply_gradients(grads=grads).apply_ema(0.99)
    trainer.save(epoch=0, step=1)

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
