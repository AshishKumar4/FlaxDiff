import os

# Tests must run identically on any machine, CPU is enough
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Enough simulated devices to exercise a 4x2 data/fsdp mesh. Must be set before
# jax initialises its backend.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
).strip()

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def text_context():
    # Shape of the default CLIP-L/14 text context, no need for the actual encoder
    return jnp.ones((2, 77, 768), dtype=jnp.float32)
