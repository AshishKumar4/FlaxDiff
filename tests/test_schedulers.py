"""Invariant tests for the noise schedulers.

These encode the properties the rest of the library relies on:
variance preservation for VP schedules, alpha=1 for the generalized (VE)
schedules, and exact invertibility of the forward diffusion.
"""

import jax
import jax.numpy as jnp
import pytest

from flaxdiff.predictors import (
    EpsilonPredictionTransform,
    VPredictionTransform,
    get_diffusion_preset,
)
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


@pytest.mark.parametrize("P_mean,P_std", [(-0.4, 1.0), (-1.2, 1.2)])
def test_edm_lognormal_sigma_distribution(rng, P_mean, P_std):
    """EDM training sigmas follow exp(N(P_mean, P_std^2)), defaulting to EDM2."""
    from flaxdiff.utils import RandomMarkovState

    schedule = EDMNoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5, P_mean=P_mean, P_std=P_std)
    steps, _ = schedule.generate_timesteps(20000, RandomMarkovState(rng))
    log_sigma = jnp.log(schedule.get_sigmas(steps))
    assert abs(float(jnp.mean(log_sigma)) - P_mean) < 0.05
    assert abs(float(jnp.std(log_sigma)) - P_std) < 0.05


def test_edm_defaults_to_edm2_distribution():
    schedule = EDMNoiseScheduler(1)
    assert (schedule.P_mean, schedule.P_std) == (-0.4, 1.0)


############################################################################################################
# min-SNR-gamma loss weighting (Hang et al. 2023)
############################################################################################################

MIN_SNR_STEPS = jnp.array([10, 200, 400, 600, 800, 990])


def make_min_snr_schedule(transform, gamma):
    return CosineNoiseScheduler(1000, min_snr_gamma=gamma, prediction_transform=transform)


def test_min_snr_needs_a_parameterization():
    with pytest.raises(ValueError):
        CosineNoiseScheduler(1000, min_snr_gamma=5.0)


def test_min_snr_weights_keep_the_requested_shape():
    schedule = make_min_snr_schedule(EpsilonPredictionTransform(), 5.0)
    assert schedule.get_weights(MIN_SNR_STEPS, shape=(-1, 1, 1, 1)).shape == (len(MIN_SNR_STEPS), 1, 1, 1)
    assert schedule.get_weights(MIN_SNR_STEPS, shape=(-1,)).shape == (len(MIN_SNR_STEPS),)


def test_min_snr_epsilon_weights_match_the_paper():
    schedule = make_min_snr_schedule(EpsilonPredictionTransform(), 5.0)
    snr = schedule.get_snr(MIN_SNR_STEPS)
    expected = jnp.minimum(snr, 5.0) / snr
    assert jnp.allclose(schedule.get_weights(MIN_SNR_STEPS, shape=(-1,)), expected, rtol=1e-5)


def test_min_snr_v_weights_match_the_paper():
    schedule = make_min_snr_schedule(VPredictionTransform(), 5.0)
    snr = schedule.get_snr(MIN_SNR_STEPS)
    expected = jnp.minimum(snr, 5.0) / (snr + 1)
    assert jnp.allclose(schedule.get_weights(MIN_SNR_STEPS, shape=(-1,)), expected, rtol=1e-5)


def test_min_snr_weights_are_capped_and_non_increasing_in_snr():
    """The whole point: high-SNR (low noise) steps stop dominating the gradient."""
    schedule = make_min_snr_schedule(EpsilonPredictionTransform(), 5.0)
    # ascending timesteps are descending SNR, so weights must be non-decreasing
    weights = schedule.get_weights(jnp.arange(1, 1000, 10), shape=(-1,))
    assert jnp.all(jnp.diff(weights) >= -1e-6)
    assert jnp.all(weights <= 1.0 + 1e-6)


def test_min_snr_gamma_infinity_is_the_unweighted_case():
    schedule = make_min_snr_schedule(EpsilonPredictionTransform(), float('inf'))
    assert jnp.allclose(schedule.get_weights(MIN_SNR_STEPS, shape=(-1,)), 1.0, atol=1e-6)


def test_min_snr_is_off_by_default():
    schedule = CosineNoiseScheduler(1000)
    assert jnp.allclose(
        schedule.get_weights(MIN_SNR_STEPS, shape=(-1,)),
        schedule.get_schedule_weights(MIN_SNR_STEPS, shape=(-1,)),
    )


@pytest.mark.parametrize("name", ['cosine', 'edm', 'karras', 'flow'])
def test_preset_wires_min_snr_into_the_training_schedule_only(name):
    train, sample, transform = get_diffusion_preset(name, min_snr_gamma=5.0)
    assert train.min_snr_gamma == 5.0
    assert train.prediction_transform is transform
    assert sample.min_snr_gamma is None
