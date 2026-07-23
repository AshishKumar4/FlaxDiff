"""Round-trip tests for the prediction transforms (parameterizations).

forward_diffusion produces (x_t, c_in, target); backward_diffusion must
recover x0 and epsilon exactly from that target. This must hold for every
parameterization on both VP and VE schedules.
"""

import jax
import jax.numpy as jnp
import pytest

from flaxdiff.predictors import (
    EpsilonPredictionTransform,
    DirectPredictionTransform,
    VPredictionTransform,
    KarrasPredictionTransform,
)
from flaxdiff.schedulers import CosineNoiseScheduler, KarrasVENoiseScheduler
from flaxdiff.schedulers.common import get_coeff_shapes_tuple

TRANSFORMS = [
    ("epsilon", EpsilonPredictionTransform()),
    ("x0", DirectPredictionTransform()),
    ("v", VPredictionTransform()),
    ("karras", KarrasPredictionTransform(sigma_data=0.5)),
]
SCHEDULES = [
    ("cosine", CosineNoiseScheduler(1000), jnp.array([10, 300, 600, 900])),
    ("karras_ve", KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5), jnp.array([0.2, 0.4, 0.6, 0.8])),
]


@pytest.mark.parametrize("tname,transform", TRANSFORMS)
@pytest.mark.parametrize("sname,schedule,steps", SCHEDULES)
def test_forward_backward_roundtrip(tname, transform, sname, schedule, steps, rng):
    key0, key1 = jax.random.split(rng)
    # one timestep per batch element
    x0 = jax.random.normal(key0, (4, 8, 8, 3))
    noise = jax.random.normal(key1, (4, 8, 8, 3))
    rates = schedule.get_rates(steps, get_coeff_shapes_tuple(x0))

    xt, c_in, target = transform.forward_diffusion(x0, noise, rates)
    # A model that outputs the exact target must recover x0 and noise
    recovered_x0, recovered_noise = transform.backward_diffusion(xt, target, rates)

    assert jnp.max(jnp.abs(recovered_x0 - x0)) < 5e-3, f"{tname} on {sname}: x0 not recovered"
    assert jnp.max(jnp.abs(recovered_noise - noise)) < 5e-3, f"{tname} on {sname}: noise not recovered"


def test_karras_preconditioning_matches_paper(rng):
    """c_in, c_skip and c_out must match Karras et al. 2022 Table 1."""
    transform = KarrasPredictionTransform(sigma_data=0.5)
    schedule = KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)
    steps = jnp.array([0.3])
    rates = schedule.get_rates(steps, (-1, 1, 1, 1))
    _, sigma = rates
    sd = 0.5

    c_in = transform.get_input_scale(rates)
    assert jnp.allclose(c_in, 1 / jnp.sqrt(sigma**2 + sd**2), rtol=1e-4)

    x_t = jax.random.normal(rng, (1, 8, 8, 3))
    raw = jax.random.normal(jax.random.fold_in(rng, 1), (1, 8, 8, 3))
    c_skip = sd**2 / (sd**2 + sigma**2)
    c_out = sigma * sd / jnp.sqrt(sd**2 + sigma**2)
    expected = c_skip * x_t + c_out * raw
    assert jnp.allclose(transform.pred_transform(x_t, raw, rates), expected, rtol=1e-4)
