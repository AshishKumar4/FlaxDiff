"""Invariant tests for the noise schedulers.

These encode the properties the rest of the library relies on:
variance preservation for VP schedules, alpha=1 for the generalized (VE)
schedules, and exact invertibility of the forward diffusion.
"""

import jax
import jax.numpy as jnp
import pytest

from flaxdiff.schedulers import (
    CosineNoiseScheduler,
    LinearNoiseSchedule,
    KarrasVENoiseScheduler,
    EDMNoiseScheduler,
)
from flaxdiff.schedulers.common import get_coeff_shapes_tuple

VP_SCHEDULES = [
    ("cosine", lambda: CosineNoiseScheduler(1000), jnp.array([10, 300, 600, 900])),
    ("linear", lambda: LinearNoiseSchedule(1000), jnp.array([10, 300, 600, 900])),
]
VE_SCHEDULES = [
    ("karras_ve", lambda: KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5), jnp.array([0.05, 0.3, 0.6, 0.95])),
    ("edm", lambda: EDMNoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5), jnp.array([0.05, 0.3, 0.6, 0.95])),
]


@pytest.mark.parametrize("name,make,steps", VP_SCHEDULES)
def test_vp_variance_preserving(name, make, steps):
    schedule = make()
    alpha, sigma = schedule.get_rates(steps, shape=(-1,))
    assert jnp.allclose(alpha**2 + sigma**2, 1.0, atol=1e-5)


@pytest.mark.parametrize("name,make,steps", VE_SCHEDULES)
def test_ve_signal_rate_is_one(name, make, steps):
    schedule = make()
    alpha, sigma = schedule.get_rates(steps, shape=(-1,))
    assert jnp.allclose(alpha, 1.0)
    # Noise grows monotonically with t
    assert jnp.all(jnp.diff(sigma) > 0)


@pytest.mark.parametrize("name,make,steps", VP_SCHEDULES + VE_SCHEDULES)
def test_forward_diffusion_invertible(name, make, steps, rng):
    schedule = make()
    key0, key1 = jax.random.split(rng)
    x0 = jax.random.normal(key0, (4, 8, 8, 3))
    noise = jax.random.normal(key1, (4, 8, 8, 3))
    xt = schedule.add_noise(x0, noise, steps)
    recovered = schedule.remove_all_noise(xt, noise, steps)
    assert jnp.max(jnp.abs(recovered - x0)) < 1e-4


def test_karras_weights_match_edm_lambda():
    """KarrasVE loss weights must equal EDM's lambda(sigma) = (s^2+sd^2)/(s*sd)^2.
    The current epsilon guard halves the weight at sigma_min."""
    schedule = KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)
    # steps away from sigma_min; the sigma_min case is the xfail test below
    steps = jnp.array([0.3, 0.6, 1.0])
    sigma = schedule.get_sigmas(steps)
    expected = (sigma**2 + 0.5**2) / ((sigma * 0.5) ** 2)
    got = schedule.get_weights(steps, shape=(-1,))
    assert jnp.allclose(got, expected, rtol=1e-3)


def test_karras_weights_at_sigma_min():
    schedule = KarrasVENoiseScheduler(1, sigma_min=0.002, sigma_max=80, rho=7, sigma_data=0.5)
    sigma = jnp.array([0.002])
    expected = (sigma**2 + 0.5**2) / ((sigma * 0.5) ** 2)
    # steps=0 maps to sigma_min under the karras rho spacing
    got = schedule.get_weights(jnp.array([0.0]), shape=(-1,))
    assert jnp.allclose(got, expected, rtol=1e-2)


def test_edm_lognormal_sigma_distribution(rng):
    """EDM training sigmas must follow exp(N(-1.2, 1.2^2)) when timesteps=1."""
    from flaxdiff.utils import RandomMarkovState

    schedule = EDMNoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)
    steps, _ = schedule.generate_timesteps(20000, RandomMarkovState(rng))
    log_sigma = jnp.log(schedule.get_sigmas(steps))
    assert abs(float(jnp.mean(log_sigma)) - (-1.2)) < 0.05
    assert abs(float(jnp.std(log_sigma)) - 1.2) < 0.05
