"""End-to-end sampler tests against an analytic oracle denoiser.

For gaussian data x0 ~ N(0, s^2 I), the optimal denoiser has a closed form:
E[x0 | x_t] = alpha * s^2 / (alpha^2 s^2 + sigma^2) * x_t. A sampler
integrating the reverse process with this oracle must produce samples with
mean 0 and std s. This catches integrator bugs, step-domain bugs, and
shape bugs in one place, with no training involved.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from flaxdiff.inputs import DiffusionInputConfig
from flaxdiff.predictors import EpsilonPredictionTransform, KarrasPredictionTransform
from flaxdiff.samplers.ddim import DDIMSampler
from flaxdiff.samplers.ddpm import DDPMSampler, SimpleDDPMSampler
from flaxdiff.samplers.euler import EulerSampler, EulerAncestralSampler
from flaxdiff.samplers.heun_sampler import HeunSampler
from flaxdiff.samplers.multistep_dpm import MultiStepDPM
from flaxdiff.samplers.rk4_sampler import RK4Sampler
from flaxdiff.schedulers import CosineNoiseScheduler, KarrasVENoiseScheduler
from flaxdiff.schedulers.common import get_coeff_shapes_tuple
from flaxdiff.utils import RandomMarkovState

DATA_STD = 0.3

input_config = DiffusionInputConfig(
    sample_data_key="image",
    sample_data_shape=(8, 8, 3),
    conditions=[],
)


class VPOracle(nn.Module):
    """Optimal epsilon-predictor for x0 ~ N(0, DATA_STD^2) on a VP schedule.

    The sampler feeds raw timesteps for discrete schedules, so rates are
    recomputed from the schedule inside the module.
    """
    schedule: CosineNoiseScheduler

    @nn.compact
    def __call__(self, x, temb):
        alpha, sigma = self.schedule.get_rates(temb, get_coeff_shapes_tuple(x))
        return x * sigma / (alpha**2 * DATA_STD**2 + sigma**2)


class KarrasOracle(nn.Module):
    """Optimal raw-F predictor under the Karras preconditioning (VE, alpha=1).

    Receives x * c_in and temb = log(sigma)/4; must output F such that
    c_skip * x + c_out * F = E[x0 | x] = s^2/(s^2 + sigma^2) * x.
    """
    sigma_data: float = 0.5

    @nn.compact
    def __call__(self, x_scaled, temb):
        sigma = jnp.exp(4.0 * temb).reshape(get_coeff_shapes_tuple(x_scaled))
        sd = self.sigma_data
        c_in = 1 / jnp.sqrt(sigma**2 + sd**2)
        c_skip = sd**2 / (sd**2 + sigma**2)
        c_out = sigma * sd / jnp.sqrt(sd**2 + sigma**2)
        x = x_scaled / c_in
        x0 = DATA_STD**2 / (DATA_STD**2 + sigma**2) * x
        return (x0 - c_skip * x) / c_out


def make_vp_sampler(sampler_class, **kwargs):
    schedule = CosineNoiseScheduler(1000)
    model = VPOracle(schedule=schedule)
    return model, sampler_class(
        model=model,
        noise_schedule=schedule,
        model_output_transform=EpsilonPredictionTransform(),
        input_config=input_config,
        guidance_scale=0.0,
        **kwargs,
    )


def make_karras_sampler(sampler_class, **kwargs):
    schedule = KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5)
    model = KarrasOracle()
    return model, sampler_class(
        model=model,
        noise_schedule=schedule,
        model_output_transform=KarrasPredictionTransform(sigma_data=0.5),
        input_config=input_config,
        guidance_scale=0.0,
        **kwargs,
    )


def assert_gaussian_stats(samples, std=DATA_STD, tol=0.05):
    assert jnp.all(jnp.isfinite(samples)), "sampler produced non-finite values"
    assert abs(float(jnp.mean(samples))) < tol
    assert abs(float(jnp.std(samples)) - std) < tol


def generate(model, sampler, **kwargs):
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))
    defaults = dict(
        num_samples=256,
        resolution=8,
        diffusion_steps=100,
        start_step=1000,
        rngstate=RandomMarkovState(jax.random.PRNGKey(2)),
    )
    defaults.update(kwargs)
    return sampler.generate_samples(params, **defaults)


@pytest.mark.parametrize("sampler_class", [EulerSampler, DDIMSampler, SimpleDDPMSampler])
def test_vp_sampler_converges(sampler_class):
    model, sampler = make_vp_sampler(sampler_class)
    samples = generate(model, sampler)
    assert_gaussian_stats(samples)


@pytest.mark.parametrize("sampler_class", [EulerSampler, EulerAncestralSampler, DDIMSampler, HeunSampler, MultiStepDPM])
def test_karras_sampler_converges(sampler_class):
    model, sampler = make_karras_sampler(sampler_class)
    samples = generate(model, sampler)
    assert_gaussian_stats(samples)


@pytest.mark.xfail(strict=True, reason="bug: default start_step uses max_timesteps but the step domain is hardcoded to [0, 1000]")
def test_karras_sampler_default_start_step():
    """With no explicit start_step the sampler must still denoise from sigma_max.
    Today max_timesteps=1 collapses the schedule to a single no-op step and
    the 'samples' are just the initial noise."""
    model, sampler = make_karras_sampler(EulerSampler)
    samples = generate(model, sampler, start_step=None)
    assert_gaussian_stats(samples)


@pytest.mark.xfail(strict=True, reason="bug: sample_model reshapes rates with the default (-1,1,1,1), breaking 5D video shapes for every sampler")
def test_ddim_video_samples():
    """DDIM's own take_next_step is 5D-safe, but the shared sample_model wrapper
    still scales the input with 4D-reshaped rates, so video is broken everywhere."""
    model, sampler = make_karras_sampler(DDIMSampler)
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))
    samples = sampler.generate_samples(
        params,
        num_samples=2,
        resolution=8,
        sequence_length=3,
        diffusion_steps=50,
        start_step=1000,
        rngstate=RandomMarkovState(jax.random.PRNGKey(2)),
    )
    assert samples.shape == (2, 3, 8, 8, 3)
    assert jnp.all(jnp.isfinite(samples))


@pytest.mark.xfail(strict=True, reason="bug: samplers reshape rates with the default (-1,1,1,1), breaking 5D video shapes")
def test_euler_video_samples():
    model, sampler = make_karras_sampler(EulerSampler)
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))
    samples = sampler.generate_samples(
        params,
        num_samples=2,
        resolution=8,
        sequence_length=3,
        diffusion_steps=50,
        start_step=1000,
        rngstate=RandomMarkovState(jax.random.PRNGKey(2)),
    )
    assert samples.shape == (2, 3, 8, 8, 3)


@pytest.mark.xfail(strict=True, reason="bug: DDPMSampler casts the step array through int(), crashing for batch > 1")
def test_ddpm_sampler_converges():
    model, sampler = make_vp_sampler(DDPMSampler)
    samples = generate(model, sampler, diffusion_steps=1000)
    assert_gaussian_stats(samples)


@pytest.mark.xfail(strict=True, reason="bug: DDIM eta > 0 calls .sqrt() on a jax array and uses the wrong variance split")
def test_ddim_eta_converges():
    model, sampler = make_vp_sampler(DDIMSampler, eta=0.5)
    samples = generate(model, sampler)
    assert_gaussian_stats(samples)


@pytest.mark.xfail(strict=True, reason="bug: RK4Sampler jits over its python-callable argument")
def test_rk4_sampler_runs():
    model, sampler = make_karras_sampler(RK4Sampler)
    samples = generate(model, sampler, diffusion_steps=20)
    assert jnp.all(jnp.isfinite(samples))


@pytest.mark.xfail(strict=True, reason="bug: MultiStepDPM keeps stale derivative history across generate calls")
def test_multistep_dpm_reentrant():
    model, sampler = make_karras_sampler(MultiStepDPM)
    first = generate(model, sampler)
    second = generate(model, sampler)
    assert jnp.allclose(first, second, atol=1e-5)


@pytest.mark.parametrize("spacing", ["quadratic", "karras", "exponential"])
@pytest.mark.xfail(strict=True, reason="bug: non-linear timestep spacings produce log(0) or duplicate int16 steps (dt=0)")
def test_timestep_spacing_produces_finite_samples(spacing):
    model, sampler = make_karras_sampler(EulerSampler, timestep_spacing=spacing)
    samples = generate(model, sampler, diffusion_steps=50)
    assert_gaussian_stats(samples, tol=0.1)
