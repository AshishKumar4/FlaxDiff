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
        "output_channels": 3,
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


def test_parse_config_drops_removed_fields():
    """Configs from older runs carry since-removed flags; reconstruction must
    drop them instead of crashing."""
    result = parse_config(make_config(model_overrides={
        "use_flash_attention": False, "norm_groups": 8, "activation": "swish"}))
    assert type(result["model"]).__name__ == "SimpleDiT"


def test_parse_config_noise_schedule_selection():
    edm = parse_config(make_config())
    assert type(edm["noise_schedule"]).__name__ == "KarrasVENoiseScheduler"

    cosine = parse_config(make_config(arguments_overrides={"noise_schedule": "cosine"}))
    assert type(cosine["noise_schedule"]).__name__ == "CosineNoiseScheduler"


def test_parse_config_resolves_dotted_values():
    """Function paths like 'jax.nn.mish' (and the 'jax._src.nn.functions.silu'
    that old configs contain from the aliasing bug) must resolve to the actual
    function, not silently vanish into class defaults."""
    config = make_config(arguments_overrides={"architecture": "uvit"})
    config["model"]["activation"] = "jax.nn.mish"
    result = parse_config(config)
    assert result["model"].activation is jax.nn.mish

    config = make_config(arguments_overrides={"architecture": "uvit"})
    config["model"]["activation"] = "jax._src.nn.functions.silu"
    result = parse_config(config)
    assert result["model"].activation is jax.nn.silu


def test_training_and_inference_share_schedule_presets():
    """--noise_schedule must mean the same thing at train and inference time.
    Both sides now build from get_diffusion_preset."""
    from flaxdiff.predictors import get_diffusion_preset

    train, sample, transform = get_diffusion_preset("edm")
    assert type(train).__name__ == "EDMNoiseScheduler"
    assert type(sample).__name__ == "KarrasVENoiseScheduler"
    assert type(transform).__name__ == "KarrasPredictionTransform"

    train, sample, transform = get_diffusion_preset("cosine")
    assert type(train).__name__ == "CosineNoiseScheduler"
    assert type(sample).__name__ == "CosineNoiseScheduler"
    assert type(transform).__name__ == "VPredictionTransform"

    # parse_config must agree with the preset for every name
    for name in ("edm", "karras", "cosine"):
        _, sample, transform = get_diffusion_preset(name)
        result = parse_config(make_config(arguments_overrides={"noise_schedule": name}))
        assert type(result["noise_schedule"]) is type(sample)
        assert type(result["prediction_transform"]) is type(transform)


def test_registry_builds_every_architecture():
    """Every registry entry must construct from a minimal string config -
    the same call path training and inference share."""
    from flaxdiff.models.registry import MODEL_REGISTRY, build_model

    base = {"emb_features": 64, "dtype": "float32", "precision": "default",
            "output_channels": 3, "patch_size": 4, "num_layers": 2, "num_heads": 2}
    per_arch = {
        "unet": {"feature_depths": [16, 32], "attention_configs": [None, None],
                 "num_res_blocks": 1, "num_middle_res_blocks": 1,
                 "activation": "swish", "norm_groups": 4},
        "uvit": {"num_layers": 4},
        "simple_udit": {"num_layers": 4},
        "hierarchical_mmdit": {"emb_features": (32, 64, 96), "num_layers": (1, 1, 1),
                               "num_heads": (2, 2, 2), "base_patch_size": 2},
    }
    for name in MODEL_REGISTRY:
        config = {**base, **per_arch.get(name, {})}
        model = build_model(name, config)
        assert type(model).__name__ == MODEL_REGISTRY[name].__name__


def test_registry_suffix_canonicalization():
    from flaxdiff.models.registry import canonicalize_architecture

    name, flags = canonicalize_architecture("hybrid_dit+2d+hilbert")
    assert name == "hybrid_dit"
    assert flags == {"use_2d_fusion": True, "use_hilbert": True}
