"""Trainer smoke tests: training without wandb, and checkpoint save/restore.

These run on CPU with a tiny regression model through SimpleTrainer, which
owns the state/checkpoint/loop machinery shared by all trainers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from flaxdiff.trainer.simple_trainer import SimpleTrainer


class TinyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x)


def make_trainer(tmp_path, name="smoke", load_from_checkpoint=None):
    return SimpleTrainer(
        model=TinyModel(),
        input_shapes={"x": (4,)},
        optimizer=optax.adam(1e-3),
        rngs=jax.random.PRNGKey(0),
        name=name,
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
        load_from_checkpoint=load_from_checkpoint,
    )


def batch_iterator():
    while True:
        yield {"image": jnp.ones((8, 4)), "label": jnp.zeros((8, 2))}


def test_fit_without_wandb(tmp_path):
    trainer = make_trainer(tmp_path)
    data = {"train": batch_iterator, "train_len": 32}
    state = trainer.fit(data, train_steps_per_epoch=4, epochs=2, val_steps_per_epoch=0)
    # the tiny regression must actually learn something
    final_loss = float(jnp.mean(optax.l2_loss(
        state.apply_fn(state.params, jnp.ones((8, 4))), jnp.zeros((8, 2)))))
    initial = make_trainer(tmp_path, name="fresh")
    initial_loss = float(jnp.mean(optax.l2_loss(
        initial.state.apply_fn(initial.state.params, jnp.ones((8, 4))), jnp.zeros((8, 2)))))
    assert final_loss < initial_loss


def test_save_writes_checkpoint(tmp_path):
    """save() swallows exceptions, so assert the checkpoint actually landed."""
    trainer = make_trainer(tmp_path)
    trainer.save(epoch=0, step=1)
    assert trainer.checkpointer.latest_step() == 1


def test_restore_preserves_optimizer_state(tmp_path):
    trainer = make_trainer(tmp_path)

    # One real update so the adam moments and step counter are non-trivial
    grads = jax.tree.map(jnp.ones_like, trainer.state.params)
    trainer.state = trainer.state.apply_gradients(grads=grads)
    trainer.save(epoch=0, step=1)

    restored = make_trainer(tmp_path, load_from_checkpoint=trainer.checkpoint_path())
    assert int(restored.state.step) == 1, "optimizer step counter was reset"

    old_mu = jax.tree.leaves(trainer.state.opt_state)
    new_mu = jax.tree.leaves(restored.state.opt_state)
    assert all(np.allclose(a, b) for a, b in zip(old_mu, new_mu)), "adam moments were reset"
