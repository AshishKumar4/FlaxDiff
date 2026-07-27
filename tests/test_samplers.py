"""End-to-end sampler tests against an analytic oracle denoiser.

For gaussian data x0 ~ N(0, s^2 I), the optimal denoiser has a closed form:
E[x0 | x_t] = alpha * s^2 / (alpha^2 s^2 + sigma^2) * x_t. A sampler
integrating the reverse process with this oracle must produce samples with
mean 0 and std s. This catches integrator bugs, step-domain bugs, and
shape bugs in one place, with no training involved.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn

from flaxdiff.inputs import ConditionalInputConfig, DiffusionInputConfig
from flaxdiff.inputs.encoders import ConditioningEncoder
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


def test_karras_sampler_default_start_step():
    """With no explicit start_step the sampler must denoise from sigma_max."""
    model, sampler = make_karras_sampler(EulerSampler)
    samples = generate(model, sampler, start_step=None)
    assert_gaussian_stats(samples)


def test_ddim_video_samples():
    model, sampler = make_karras_sampler(DDIMSampler)
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))
    samples = sampler.generate_samples(
        params,
        num_samples=2,
        resolution=8,
        sequence_length=3,
        diffusion_steps=50,
        rngstate=RandomMarkovState(jax.random.PRNGKey(2)),
    )
    assert samples.shape == (2, 3, 8, 8, 3)
    assert jnp.all(jnp.isfinite(samples))


def test_euler_video_samples():
    model, sampler = make_karras_sampler(EulerSampler)
    params = model.init(jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)))
    samples = sampler.generate_samples(
        params,
        num_samples=2,
        resolution=8,
        sequence_length=3,
        diffusion_steps=50,
        rngstate=RandomMarkovState(jax.random.PRNGKey(2)),
    )
    assert samples.shape == (2, 3, 8, 8, 3)


def test_ddpm_sampler_converges():
    model, sampler = make_vp_sampler(DDPMSampler)
    samples = generate(model, sampler, diffusion_steps=1000)
    assert_gaussian_stats(samples)


def test_ddim_eta_converges():
    model, sampler = make_vp_sampler(DDIMSampler, eta=0.5)
    samples = generate(model, sampler)
    assert_gaussian_stats(samples)


def test_rk4_sampler_runs():
    model, sampler = make_karras_sampler(RK4Sampler)
    samples = generate(model, sampler, diffusion_steps=20)
    assert jnp.all(jnp.isfinite(samples))


def test_multistep_dpm_reentrant():
    model, sampler = make_karras_sampler(MultiStepDPM)
    first = generate(model, sampler)
    second = generate(model, sampler)
    assert jnp.allclose(first, second, atol=1e-5)


############################################################################################################
# Interval-limited classifier-free guidance (Kynkaanniemi et al. 2024)
############################################################################################################

@dataclass
class LabelEncoder(ConditioningEncoder):
    """Smallest conditioning seam that still gives CFG a real conditional and
    unconditional branch to interpolate between."""
    @property
    def key(self):
        return "label"

    def tokenize(self, data):
        return jnp.asarray(data, dtype=jnp.float32).reshape(-1, 1)

    def encode_from_tokens(self, tokens):
        return tokens

    def serialize(self):
        return {}

    @staticmethod
    def deserialize(serialized_config):
        raise NotImplementedError


conditional_input_config = DiffusionInputConfig(
    sample_data_key="image",
    sample_data_shape=(8, 8, 3),
    conditions=[
        ConditionalInputConfig(
            encoder=LabelEncoder(model=None, tokenizer=None),
            conditioning_data_key="label",
            unconditional_input=0.0,
        )
    ],
)


class ConditionalVPOracle(nn.Module):
    """VPOracle offset by the label, so the guided output reads back the scale
    that was actually applied."""
    schedule: CosineNoiseScheduler

    @nn.compact
    def __call__(self, x, temb, label):
        alpha, sigma = self.schedule.get_rates(temb, get_coeff_shapes_tuple(x))
        eps = x * sigma / (alpha**2 * DATA_STD**2 + sigma**2)
        return eps + label.reshape(get_coeff_shapes_tuple(x))


def make_guided_sampler(**kwargs):
    schedule = CosineNoiseScheduler(1000)
    model = ConditionalVPOracle(schedule=schedule)
    sampler = DDIMSampler(
        model=model,
        noise_schedule=schedule,
        model_output_transform=EpsilonPredictionTransform(),
        input_config=conditional_input_config,
        **kwargs,
    )
    params = model.init(
        jax.random.PRNGKey(1), jnp.ones((1, 8, 8, 3)), jnp.ones((1,)), jnp.ones((1, 1))
    )
    return params, sampler


@pytest.mark.parametrize("progress,inside", [(0.1, False), (0.5, True), (0.9, False)])
def test_interval_cfg_applies_only_inside_the_interval(progress, inside):
    params, full = make_guided_sampler(guidance_scale=3.0)
    _, interval = make_guided_sampler(guidance_scale=3.0, guidance_start=0.4, guidance_stop=0.6)
    _, unguided = make_guided_sampler(guidance_scale=0.0)

    x_t = jax.random.normal(jax.random.PRNGKey(3), (4, 8, 8, 3))
    labels = jnp.full((4, 1), 0.7)
    t = jnp.full((4,), (1.0 - progress) * 1000)

    def output(sampler):
        return sampler.sample_model(params, x_t, t, labels)[2]

    matches_full = bool(jnp.allclose(output(interval), output(full), atol=1e-5))
    matches_unguided = bool(jnp.allclose(output(interval), output(unguided), atol=1e-5))
    assert matches_full is inside
    assert matches_unguided is not inside


def test_interval_cfg_defaults_to_the_full_range():
    params, default = make_guided_sampler(guidance_scale=3.0)
    _, explicit = make_guided_sampler(guidance_scale=3.0, guidance_start=0.0, guidance_stop=1.0)

    x_t = jax.random.normal(jax.random.PRNGKey(3), (4, 8, 8, 3))
    labels = jnp.full((4, 1), 0.7)
    t = jnp.full((4,), 500.0)
    assert jnp.allclose(
        default.sample_model(params, x_t, t, labels)[2],
        explicit.sample_model(params, x_t, t, labels)[2],
        atol=1e-6,
    )


def test_empty_guidance_interval_generates_the_unguided_samples():
    params, empty = make_guided_sampler(guidance_scale=3.0, guidance_start=0.9, guidance_stop=0.1)
    _, unguided = make_guided_sampler(guidance_scale=0.0)

    def run(sampler):
        return sampler.generate_samples(
            params, num_samples=16, resolution=8, diffusion_steps=25,
            rngstate=RandomMarkovState(jax.random.PRNGKey(2)),
            model_conditioning_inputs=(jnp.full((16, 1), 0.7),),
        )

    assert jnp.allclose(run(empty), run(unguided), atol=1e-5)


