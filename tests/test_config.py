"""Train -> wandb config -> inference reconstruction round-trip tests.

parse_config must rebuild exactly the model that was trained. Any key that
is silently dropped or remapped means the inference pipeline runs a
different model than the one in the checkpoint.
"""

import jax
import jax.numpy as jnp
import pytest

from flaxdiff.inference.utils import parse_config


def make_config(model_overrides=None, arguments_overrides=None):
    """A minimal wandb-style config, the same shape training.py logs."""
    model = {
        "emb_features": 64,
        "dtype": "bfloat16",
        "precision": "high",
        "activation": "swish",
        "output_channels": 3,
        "norm_groups": 4,
        "patch_size": 4,
        "num_layers": 2,
        "num_heads": 2,
        "dropout_rate": 0.0,
        "mlp_ratio": 2,
        "use_hilbert": False,
        "use_zigzag": False,
    }
    model.update(model_overrides or {})
    arguments = {
        "architecture": "simple_dit",
        "image_size": 32,
        "noise_schedule": "edm",
    }
    arguments.update(arguments_overrides or {})
    return {
        "model": model,
        "architecture": arguments["architecture"],
        "arguments": arguments,
        "input_config": {
            "sample_data_key": "image",
            "sample_data_shape": (32, 32, 3),
            "conditions": [],
        },
    }


def test_parse_config_rebuilds_model():
    result = parse_config(make_config())
    model = result["model"]
    assert type(model).__name__ == "SimpleDiT"
    assert model.emb_features == 64
    assert model.num_layers == 2
    assert model.dtype == jnp.bfloat16
    assert model.precision == jax.lax.Precision.HIGH
    assert model.activation is jax.nn.swish


def test_parse_config_noise_schedule_selection():
    edm = parse_config(make_config())
    assert type(edm["noise_schedule"]).__name__ == "KarrasVENoiseScheduler"

    cosine = parse_config(make_config(arguments_overrides={"noise_schedule": "cosine"}))
    assert type(cosine["noise_schedule"]).__name__ == "CosineNoiseScheduler"


@pytest.mark.xfail(strict=True, reason="bug: parse_config silently drops any string value containing a dot")
def test_parse_config_preserves_dotted_values():
    """String config values with a dot in them (module paths, filenames, version
    strings) must survive the round trip, not vanish into class defaults."""
    result = parse_config(make_config(model_overrides={"activation": "jax.nn.mish"}))
    model = result["model"]
    # a dropped key falls back to the class default (swish) instead of erroring
    assert model.activation is jax.nn.mish, "activation was silently dropped"
