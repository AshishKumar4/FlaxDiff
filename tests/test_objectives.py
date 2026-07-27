"""The Objective seam: the trainer owns mechanics, the objective owns the loss.

The diffusion objective was lifted out of the trainer's train step, so the
first test here is a golden fingerprint of the parameters, EMA and optimizer
state after five real steps, captured from the implementation that inlined the
objective. It must not move.
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
from flaxdiff.trainer.general_diffusion_trainer import TrainState
from flaxdiff.trainer.objectives import EMASpec, Objective

RES = 8


def tree_fingerprint(tree):
    # per-leaf sums accumulated in python floats, so the golden values below
    # do not depend on float32 reduction order
    return sum(float(jnp.sum(leaf)) for leaf in jax.tree.leaves(tree)
               if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating))


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
        name="objective-parity",
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
        **kwargs,
    )


def batch_iterator():
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (8, 1, RES, 3))
    while True:
        yield {"image": jnp.asarray(images)}


def test_diffusion_objective_reproduces_the_inlined_train_step(tmp_path):
    trainer = make_trainer(tmp_path)
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    state = trainer.fit(data, training_steps_per_epoch=5, epochs=1, val_steps_per_epoch=0)

    assert tree_fingerprint(state.params) == pytest.approx(8.4679913880548, abs=1e-9)
    assert tree_fingerprint(state.ema_params) == pytest.approx(8.220795648946352, abs=1e-9)
    assert tree_fingerprint(state.opt_state) == pytest.approx(-0.004871756572780005, abs=1e-12)


def test_train_step_returns_auxiliary_metrics(tmp_path):
    """value_and_grad runs with has_aux so objectives can report diagnostics."""
    trainer = make_trainer(tmp_path)
    train_step = trainer._define_train_step(batch_size=8)
    batch = next(batch_iterator())

    new_state, loss, aux, rng_state = train_step(
        trainer.state, trainer.rngstate, batch, jnp.asarray(0))
    assert jnp.isfinite(loss)
    assert isinstance(aux, dict)


class ConstantObjective(Objective):
    """Two independent parameter subtrees, so EMA scoping is observable."""

    def __init__(self, ema):
        self.ema = ema

    def init_params(self, rng):
        return {"params": {"tracked": {"w": jnp.ones((2,))},
                           "untracked": {"w": jnp.ones((2,))}}}

    def loss(self, params, ema_params, batch, rng, step):
        total = sum(jnp.sum(leaf ** 2) for leaf in jax.tree.leaves(params))
        return total, {"probe": jnp.asarray(1.0)}

    def make_validation_step(self, **kwargs):
        return lambda val_state, batch: None


def make_state(objective, params=None):
    params = objective.init_params(jax.random.PRNGKey(0)) if params is None else params
    return TrainState.create(
        apply_fn=lambda *a, **k: None, params=params, ema_params=params,
        tx=optax.sgd(0.0), rngs=jax.random.PRNGKey(0), metrics=None, dynamic_scale=None)


def test_ema_over_a_subtree_leaves_its_siblings_alone():
    objective = ConstantObjective(EMASpec(decay=optax.constant_schedule(0.5),
                                          path=("params", "tracked")))
    state = make_state(objective)
    state = state.replace(params=jax.tree.map(lambda p: p * 3.0, state.params))

    updated = state.apply_ema(0.5, objective.ema.path)
    assert jnp.allclose(updated.ema_params["params"]["tracked"]["w"], 2.0)
    assert jnp.allclose(updated.ema_params["params"]["untracked"]["w"], 1.0)


def test_ema_over_the_whole_tree_is_the_default():
    state = make_state(ConstantObjective(EMASpec(decay=optax.constant_schedule(0.5))))
    state = state.replace(params=jax.tree.map(lambda p: p * 3.0, state.params))

    updated = state.apply_ema(0.5)
    assert jnp.allclose(updated.ema_params["params"]["tracked"]["w"], 2.0)
    assert jnp.allclose(updated.ema_params["params"]["untracked"]["w"], 2.0)


def test_ema_decay_follows_the_step_schedule():
    """I-JEPA's momentum ramp: the trainer must read decay at the current step."""
    ema = EMASpec(decay=optax.linear_schedule(0.996, 1.0, transition_steps=100))
    assert float(ema.decay(0)) == pytest.approx(0.996)
    assert float(ema.decay(100)) == pytest.approx(1.0)
    assert float(ema.decay(50)) == pytest.approx(0.998)


def test_trainer_drives_an_arbitrary_objective(tmp_path):
    """The seam is real: a non-diffusion objective trains through the same loop."""
    objective = ConstantObjective(EMASpec(decay=optax.constant_schedule(0.9),
                                          path=("params", "tracked")))
    trainer = make_trainer(tmp_path, objective=objective)
    data = {"train": batch_iterator, "train_len": 32, "local_batch_size": 8}
    state = trainer.fit(data, training_steps_per_epoch=3, epochs=1, val_steps_per_epoch=0)

    assert int(state.step) == 3
    # sum of squares under gradient descent must shrink
    assert tree_fingerprint(state.params) < tree_fingerprint(objective.init_params(None))
