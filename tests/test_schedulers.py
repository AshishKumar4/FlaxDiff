"""Invariant tests for the noise schedulers.

These encode the properties the rest of the library relies on:
variance preservation for VP schedules, alpha=1 for the generalized (VE)
schedules, monotone SNR along the trajectory, rate shapes that broadcast
against image and video batches, and exact invertibility of the forward
diffusion.

SCHEDULES below is the single table every shared invariant runs over, and
test_every_exported_scheduler_is_covered fails if a scheduler is exported
without being added to it, so new schedulers inherit the invariants.
"""

import jax
import jax.numpy as jnp
import pytest

import flaxdiff.schedulers as schedulers
from flaxdiff.predictors import (
    EpsilonPredictionTransform,
    VPredictionTransform,
    get_diffusion_preset,
)
from flaxdiff.schedulers import (
    CosineContinuousNoiseScheduler,
    CosineGeneralNoiseScheduler,
    CosineNoiseScheduler,
    EDMNoiseScheduler,
    ExpNoiseScheduler,
    FlowMatchingScheduler,
    KarrasVENoiseScheduler,
    LinearNoiseScheduler,
    SqrtContinuousNoiseScheduler,
)
from flaxdiff.schedulers.common import get_coeff_shapes_tuple

# Timesteps ascending from low to high noise, in each schedule's own domain:
# an index into the beta table for the discrete schedules, [0, 1] for the
# continuous ones.
DISCRETE_STEPS = jnp.array([10, 300, 600, 900])
CONTINUOUS_STEPS = jnp.array([0.05, 0.3, 0.6, 0.95])

# (class, factory, probe steps, family) — family picks the extra rate identity:
# 'vp' is variance preserving, 've' keeps alpha=1 and scales the input,
# 'flow' is the rectified-flow linear path.
SCHEDULES = [
    (CosineNoiseScheduler, lambda: CosineNoiseScheduler(1000), DISCRETE_STEPS, 'vp'),
    (LinearNoiseScheduler, lambda: LinearNoiseScheduler(1000), DISCRETE_STEPS, 'vp'),
    (ExpNoiseScheduler, lambda: ExpNoiseScheduler(1000), DISCRETE_STEPS, 'vp'),
    (CosineContinuousNoiseScheduler, lambda: CosineContinuousNoiseScheduler(), CONTINUOUS_STEPS, 'vp'),
    (SqrtContinuousNoiseScheduler, lambda: SqrtContinuousNoiseScheduler(), CONTINUOUS_STEPS, 'vp'),
    (CosineGeneralNoiseScheduler, lambda: CosineGeneralNoiseScheduler(), CONTINUOUS_STEPS, 've'),
    (KarrasVENoiseScheduler, lambda: KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5), CONTINUOUS_STEPS, 've'),
    (EDMNoiseScheduler, lambda: EDMNoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5), CONTINUOUS_STEPS, 've'),
    (FlowMatchingScheduler, lambda: FlowMatchingScheduler(), CONTINUOUS_STEPS, 'flow'),
]

# Base classes: no rates of their own, so nothing to assert invariants against
ABSTRACT_SCHEDULERS = {
    'NoiseScheduler',
    'GeneralizedNoiseScheduler',
    'DiscreteNoiseScheduler',
    'ContinuousNoiseScheduler',
}
# Exported helpers that are not schedulers
EXPORTED_HELPERS = {
    'get_coeff_shapes_tuple',
    'reshape_rates',
    'compute_resolution_shift',
    'linear_beta_schedule',
    'cosine_beta_schedule',
    'exp_beta_schedule',
}

ALL_CASES = [(cls, make, steps, family) for cls, make, steps, family in SCHEDULES]
ALL_IDS = [cls.__name__ for cls, _, _, _ in SCHEDULES]
VP_CASES = [case for case in ALL_CASES if case[3] == 'vp']
VP_IDS = [case[0].__name__ for case in VP_CASES]
VE_CASES = [case for case in ALL_CASES if case[3] == 've']
VE_IDS = [case[0].__name__ for case in VE_CASES]


def test_every_exported_scheduler_is_covered():
    """Exporting a scheduler without adding it to SCHEDULES fails here."""
    exported = set(schedulers.__all__) - EXPORTED_HELPERS - ABSTRACT_SCHEDULERS
    assert exported == {cls.__name__ for cls, _, _, _ in SCHEDULES}


def test_abstract_schedulers_stay_exported():
    """They are the documented subclassing surface."""
    assert ABSTRACT_SCHEDULERS <= set(schedulers.__all__)


@pytest.mark.parametrize("cls,make,steps,family", ALL_CASES, ids=ALL_IDS)
def test_snr_decreases_along_the_trajectory(cls, make, steps, family):
    """t runs from clean to noisy, so the signal-to-noise ratio must fall."""
    snr = make().get_snr(steps)
    assert jnp.all(jnp.diff(snr) < 0), snr


@pytest.mark.parametrize("cls,make,steps,family", ALL_CASES, ids=ALL_IDS)
@pytest.mark.parametrize("sample_shape", [(8, 8, 3), (2, 8, 8, 3)], ids=['image', 'video'])
def test_rates_broadcast_against_the_sample(cls, make, steps, family, sample_shape):
    """get_coeff_shapes_tuple is how every caller shapes the rates: the result
    must broadcast against the batch it came from, for images and for video."""
    x = jnp.zeros((len(steps),) + sample_shape)
    shape = get_coeff_shapes_tuple(x)
    assert shape == (-1,) + (1,) * (x.ndim - 1)

    alpha, sigma = make().get_rates(steps, shape=shape)
    assert alpha.shape == sigma.shape == (len(steps),) + (1,) * (x.ndim - 1)
    assert (alpha * x).shape == x.shape


@pytest.mark.parametrize("cls,make,steps,family", ALL_CASES, ids=ALL_IDS)
@pytest.mark.parametrize("sample_shape", [(8, 8, 3), (2, 8, 8, 3)], ids=['image', 'video'])
def test_forward_diffusion_invertible(cls, make, steps, family, sample_shape, rng):
    """remove_all_noise inverts add_noise exactly at the same timestep."""
    schedule = make()
    key0, key1 = jax.random.split(rng)
    full_shape = (len(steps),) + sample_shape
    x0 = jax.random.normal(key0, full_shape)
    noise = jax.random.normal(key1, full_shape)
    xt = schedule.add_noise(x0, noise, steps)
    recovered = schedule.remove_all_noise(xt, noise, steps)
    assert xt.shape == x0.shape
    assert jnp.max(jnp.abs(recovered - x0)) < 1e-4


@pytest.mark.parametrize("cls,make,steps,family", VP_CASES, ids=VP_IDS)
def test_vp_variance_preserving(cls, make, steps, family):
    alpha, sigma = make().get_rates(steps, shape=(-1,))
    assert jnp.allclose(alpha**2 + sigma**2, 1.0, atol=1e-5)


@pytest.mark.parametrize("cls,make,steps,family", VE_CASES, ids=VE_IDS)
def test_ve_signal_rate_is_one(cls, make, steps, family):
    alpha, sigma = make().get_rates(steps, shape=(-1,))
    assert jnp.allclose(alpha, 1.0)
    # Noise grows monotonically with t
    assert jnp.all(jnp.diff(sigma) > 0)


def test_flow_matching_stays_on_the_linear_path():
    """Rectified flow: x_t = (1 - t) x_0 + t eps, so the rates sum to one."""
    alpha, sigma = FlowMatchingScheduler().get_rates(CONTINUOUS_STEPS, shape=(-1,))
    assert jnp.allclose(alpha + sigma, 1.0, atol=1e-6)


def test_sqrt_schedule_matches_the_diffusion_lm_formula():
    """Li et al. 2022: alpha = sqrt(1 - t), sigma = sqrt(t)."""
    alpha, sigma = SqrtContinuousNoiseScheduler().get_rates(CONTINUOUS_STEPS, shape=(-1,))
    assert jnp.allclose(alpha, jnp.sqrt(1 - CONTINUOUS_STEPS), atol=1e-6)
    assert jnp.allclose(sigma, jnp.sqrt(CONTINUOUS_STEPS), atol=1e-6)


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
