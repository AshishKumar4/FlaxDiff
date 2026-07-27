"""Flow matching on the linear (rectified flow) path.

Covers the schedule invariants, the exact velocity round-trip, the claim that
the existing DDIM/Euler samplers already integrate the flow ODE, and a toy
end-to-end run proving the objective actually learns a distribution.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from flaxdiff.inputs import DiffusionInputConfig
from flaxdiff.predictors import FlowMatchPredictionTransform, get_diffusion_preset
from flaxdiff.samplers.ddim import DDIMSampler
from flaxdiff.samplers.euler import EulerSampler
from flaxdiff.schedulers import FlowMatchingScheduler
from flaxdiff.schedulers.flow import compute_resolution_shift
from flaxdiff.schedulers.common import get_coeff_shapes_tuple
from flaxdiff.utils import RandomMarkovState

STEPS = jnp.array([0.05, 0.3, 0.6, 0.95])


def test_linear_path_rates():
    schedule = FlowMatchingScheduler()
    alpha, sigma = schedule.get_rates(STEPS, shape=(-1,))
    assert jnp.allclose(alpha + sigma, 1.0, atol=1e-6)
    assert jnp.allclose(sigma, STEPS, atol=1e-6)
    # No input preconditioning on the linear path
    assert FlowMatchPredictionTransform().get_input_scale((alpha, sigma)) == 1


def test_endpoints_are_data_and_noise(rng):
    schedule = FlowMatchingScheduler()
    key0, key1 = jax.random.split(rng)
    x0 = jax.random.normal(key0, (4, 8, 8, 3))
    noise = jax.random.normal(key1, (4, 8, 8, 3))
    assert jnp.allclose(schedule.add_noise(x0, noise, jnp.zeros((4,))), x0, atol=1e-6)
    assert jnp.allclose(schedule.add_noise(x0, noise, jnp.ones((4,))), noise, atol=1e-6)


def test_timesteps_are_logit_normal(rng):
    schedule = FlowMatchingScheduler(logit_mean=-0.3, logit_std=1.4)
    steps, _ = schedule.generate_timesteps(50000, RandomMarkovState(rng))
    assert jnp.all((steps > 0) & (steps < 1))
    logits = jnp.log(steps) - jnp.log1p(-steps)
    assert abs(float(jnp.mean(logits)) - (-0.3)) < 0.05
    assert abs(float(jnp.std(logits)) - 1.4) < 0.05


def test_resolution_shift_is_identity_at_one():
    schedule = FlowMatchingScheduler(shift=1.0)
    assert jnp.allclose(schedule.shift_timesteps(STEPS), STEPS, atol=1e-7)


@pytest.mark.parametrize("shift", [0.5, 1.0, 3.0])
def test_resolution_shift_is_monotonic_and_fixes_endpoints(shift):
    schedule = FlowMatchingScheduler(shift=shift)
    t = jnp.linspace(0.0, 1.0, 101)
    shifted = schedule.shift_timesteps(t)
    assert jnp.all(jnp.diff(shifted) > 0)
    assert float(shifted[0]) == pytest.approx(0.0, abs=1e-7)
    assert float(shifted[-1]) == pytest.approx(1.0, abs=1e-7)
    # A shift above 1 moves every interior timestep towards higher noise
    if shift >= 1:
        assert jnp.all(shifted >= t - 1e-7)
    else:
        assert jnp.all(shifted <= t + 1e-7)


def test_resolution_shift_grows_with_sequence_length():
    shifts = [compute_resolution_shift(n) for n in (256, 1024, 4096)]
    assert shifts == sorted(shifts)
    assert shifts[0] == pytest.approx(np.exp(0.5))
    assert shifts[-1] == pytest.approx(np.exp(1.15))


def test_timestep_conditioning_is_scaled_to_the_embedding_range():
    schedule = FlowMatchingScheduler(shift=2.0)
    x = jnp.zeros((4, 8, 8, 3))
    _, temb = schedule.transform_inputs(x, STEPS)
    assert jnp.allclose(temb, schedule.shift_timesteps(STEPS) * 1000)


def test_velocity_roundtrip_is_exact(rng):
    schedule = FlowMatchingScheduler()
    transform = FlowMatchPredictionTransform()
    key0, key1 = jax.random.split(rng)
    x0 = jax.random.normal(key0, (4, 8, 8, 3))
    noise = jax.random.normal(key1, (4, 8, 8, 3))
    rates = schedule.get_rates(STEPS, get_coeff_shapes_tuple(x0))

    xt, _, target = transform.forward_diffusion(x0, noise, rates)
    assert jnp.allclose(target, noise - x0, atol=1e-6)

    recovered_x0, recovered_noise = transform.backward_diffusion(xt, target, rates)
    assert jnp.max(jnp.abs(recovered_x0 - x0)) < 1e-5
    assert jnp.max(jnp.abs(recovered_noise - noise)) < 1e-5


def test_preset_wires_flow_matching():
    for name in ('flow', 'flow_matching'):
        train, sample, transform = get_diffusion_preset(name, shift=2.0)
        assert isinstance(train, FlowMatchingScheduler)
        assert isinstance(sample, FlowMatchingScheduler)
        assert isinstance(transform, FlowMatchPredictionTransform)
        assert train.shift == 2.0 and sample.shift == 2.0


############################################################################################################
# The existing samplers already integrate the flow ODE
############################################################################################################

class ConstantVelocity(nn.Module):
    """Stands in for a trained model; take_next_step never calls it."""
    @nn.compact
    def __call__(self, x, temb):
        return x


def _flow_sampler(sampler_class):
    schedule = FlowMatchingScheduler()
    return sampler_class(
        model=ConstantVelocity(),
        noise_schedule=schedule,
        model_output_transform=FlowMatchPredictionTransform(),
        input_config=DiffusionInputConfig(sample_data_key="image", sample_data_shape=(8, 8, 3), conditions=[]),
        guidance_scale=0.0,
    )


@pytest.mark.parametrize("sampler_class", [EulerSampler, DDIMSampler])
def test_sampler_step_is_the_flow_euler_step(sampler_class, rng):
    """x_{t+dt} = x_t + u * dt exactly, for the unmodified samplers."""
    sampler = _flow_sampler(sampler_class)
    transform = FlowMatchPredictionTransform()
    schedule = sampler.noise_schedule

    key0, key1 = jax.random.split(rng)
    x_t = jax.random.normal(key0, (4, 8, 8, 3))
    velocity = jax.random.normal(key1, (4, 8, 8, 3))
    current_step = jnp.full((4,), 0.8)
    next_step = jnp.full((4,), 0.6)

    rates = schedule.get_rates(current_step, get_coeff_shapes_tuple(x_t))
    x0, eps = transform.backward_diffusion(x_t, velocity, rates)

    stepped, _ = sampler.take_next_step(
        current_samples=x_t,
        reconstructed_samples=x0,
        model_conditioning_inputs=(),
        pred_noise=eps,
        current_step=current_step,
        state=RandomMarkovState(rng),
        sample_model_fn=None,
        next_step=next_step,
    )
    expected = x_t + velocity * (0.6 - 0.8)
    assert jnp.max(jnp.abs(stepped - expected)) < 1e-5


############################################################################################################
# Toy end-to-end: a two-mode gaussian mixture in the plane
############################################################################################################

MODE_CENTERS = jnp.array([[-0.5, -0.5], [0.5, 0.5]])
MODE_STD = 0.08


def sample_mixture(key, n):
    """Two well-separated modes in the leading two channels; the third channel
    carries no mode information, so the sampler's fixed channel count does not
    turn this into a harder problem."""
    mode_key, noise_key = jax.random.split(key)
    modes = jax.random.bernoulli(mode_key, 0.5, (n,)).astype(jnp.int32)
    centers = jnp.concatenate([MODE_CENTERS[modes], jnp.zeros((n, 1))], axis=-1)
    return (centers + MODE_STD * jax.random.normal(noise_key, (n, 3))).reshape(n, 1, 1, 3)


class ToyVelocityMLP(nn.Module):
    features: int = 128

    @nn.compact
    def __call__(self, x, temb):
        t = jnp.reshape(temb, (-1, 1)) / 1000.0
        freqs = jnp.arange(1, 5, dtype=jnp.float32) * jnp.pi
        h = jnp.concatenate([x.reshape(x.shape[0], -1), t, jnp.sin(t * freqs), jnp.cos(t * freqs)], axis=-1)
        h = nn.swish(nn.Dense(self.features)(h))
        h = nn.swish(nn.Dense(self.features)(h))
        return nn.Dense(3)(h).reshape(x.shape)


def test_flow_matching_learns_a_two_mode_mixture():
    schedule, sampling_schedule, transform = get_diffusion_preset('flow')
    model = ToyVelocityMLP()
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((1, 1, 1, 3)), jnp.zeros((1,)))
    optimizer = optax.adam(3e-3)
    opt_state = optimizer.init(params)

    def loss_fn(params, x0, noise, steps):
        rates = schedule.get_rates(steps, get_coeff_shapes_tuple(x0))
        x_t, c_in, target = transform.forward_diffusion(x0, noise, rates)
        preds = model.apply(params, *schedule.transform_inputs(x_t * c_in, steps))
        weights = schedule.get_weights(steps, get_coeff_shapes_tuple(x0))
        return jnp.mean(weights * (preds - target) ** 2)

    @jax.jit
    def train_step(params, opt_state, rng_state):
        rng_state, data_key = rng_state.get_random_key()
        rng_state, noise_key = rng_state.get_random_key()
        x0 = sample_mixture(data_key, 512)
        noise = jax.random.normal(noise_key, x0.shape)
        steps, rng_state = schedule.generate_timesteps(512, rng_state)
        loss, grads = jax.value_and_grad(loss_fn)(params, x0, noise, steps)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, rng_state, loss

    rng_state = RandomMarkovState(jax.random.PRNGKey(1))
    for _ in range(1500):
        params, opt_state, rng_state, loss = train_step(params, opt_state, rng_state)
    assert float(loss) < 1.0, "flow matching loss did not come down"

    sampler = EulerSampler(
        model=model,
        noise_schedule=sampling_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(sample_data_key="image", sample_data_shape=(1, 1, 3), conditions=[]),
        guidance_scale=0.0,
    )
    samples = sampler.generate_samples(
        params, num_samples=2048, resolution=1, diffusion_steps=64,
        rngstate=RandomMarkovState(jax.random.PRNGKey(2)),
    ).reshape(-1, 3)

    assignment = samples[:, 0] > 0
    fraction = float(jnp.mean(assignment))
    assert 0.35 < fraction < 0.65, f"modes not balanced: {fraction:.2f}"

    for mode, center in enumerate(MODE_CENTERS):
        members = samples[assignment == bool(mode)]
        assert jnp.max(jnp.abs(jnp.mean(members[:, :2], axis=0) - center)) < 0.04
        assert abs(float(jnp.std(members[:, :2])) - MODE_STD) < 0.04
    # The third channel is a single zero-centred gaussian, not a mixture
    assert abs(float(jnp.mean(samples[:, 2]))) < 0.04
    assert abs(float(jnp.std(samples[:, 2])) - MODE_STD) < 0.04
