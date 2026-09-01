"""Inference pipeline tests: a checkpoint on disk through to a sample.

Everything below the artifact download works offline, so these drive the real
pipeline from a checkpoint the trainer has just written - the same path a
wandb run takes once the artifact has landed - and cover the wandb entry
points only where they report a failed lookup.

parse_config is exercised against both generations of logged config: the
current one, which carries a serialized input_config, and the older one, which
has none and has to have its text conditioning reconstructed from `arguments`.
"""

import jax
import numpy as np
import optax
import pytest

from flaxdiff import inference
from flaxdiff.inference import (
    DiffusionInferencePipeline, InferencePipeline, load_from_checkpoint, parse_config,
)
from flaxdiff.inference import utils as inference_utils
from flaxdiff.inputs import (
    CONDITIONAL_ENCODERS_REGISTRY, ConditioningEncoder, DiffusionInputConfig,
)
from flaxdiff.models.simple_dit import SimpleDiT
from flaxdiff.predictors import get_diffusion_preset
from flaxdiff.samplers.euler import EulerAncestralSampler
from flaxdiff.trainer import GeneralDiffusionTrainer
from flaxdiff.utils import get_latest_checkpoint

RES = 8
MODEL_KWARGS = dict(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)


def make_trainer(tmp_path, name="inference"):
    """The tiny unconditional DiT of tests/test_trainer.py, on one device."""
    train_schedule, _, transform = get_diffusion_preset("edm")
    return GeneralDiffusionTrainer(
        model=SimpleDiT(**MODEL_KWARGS),
        optimizer=optax.adam(1e-3),
        noise_schedule=train_schedule,
        model_output_transform=transform,
        input_config=DiffusionInputConfig(
            sample_data_key="image",
            sample_data_shape=(RES, RES, 3),
            conditions=[],
        ),
        rngs=jax.random.PRNGKey(0),
        name=name,
        wandb_config=None,
        distributed_training=False,
        checkpoint_base_path=str(tmp_path),
    )


def current_config():
    """A config in the shape training.py logs today, for the model above."""
    return {
        "model": {
            "emb_features": 16,
            "dtype": "None",
            "precision": "default",
            "output_channels": 3,
            "attention_impl": None,
            "remat": False,
            "patch_size": 4,
            "num_layers": 1,
            "num_heads": 2,
            "dropout_rate": 0.0,
            "mlp_ratio": 1,
            "use_hilbert": False,
            "use_zigzag": False,
        },
        "architecture": "simple_dit",
        "arguments": {
            "architecture": "simple_dit",
            "image_size": RES,
            "noise_schedule": "edm",
        },
        "input_config": {
            "sample_data_key": "image",
            "sample_data_shape": (RES, RES, 3),
            "conditions": [],
        },
    }


def legacy_config():
    """A run from before input_config was logged.

    No 'input_config' and no top-level 'architecture': parse_config has to
    read the architecture out of `arguments` and synthesize the text
    conditioning that every run of that era used.
    """
    return {
        "model": {
            "emb_features": 64,
            "dtype": "bfloat16",
            "precision": "high",
            "output_channels": 3,
            "feature_depths": [16, 32],
            "attention_configs": [None, None],
            "num_res_blocks": 1,
            "num_middle_res_blocks": 1,
            "activation": "swish",
            "norm_groups": 4,
            "named_norms": False,
            # a flag the model no longer accepts, as old configs carry
            "use_flash_attention": False,
        },
        "arguments": {
            "architecture": "unet",
            "image_size": 128,
            "noise_schedule": "cosine",
            "autoencoder": None,
            "autoencoder_opts": "{}",
        },
    }


class StubTextEncoder(ConditioningEncoder):
    """Stands in for CLIP-L/14, which parse_config would otherwise download."""

    @property
    def key(self):
        return "text"

    def tokenize(self, data):
        return {"tokens": np.zeros((len(data), 77), np.int32)}

    def encode_from_tokens(self, tokens):
        return np.zeros((len(tokens["tokens"]), 77, 768), np.float32)

    def serialize(self):
        return {"modelname": self.model}

    @staticmethod
    def deserialize(serialized_config):
        return StubTextEncoder(model=serialized_config["modelname"], tokenizer=None)


# --------------------------------------------------------------------------
# Checkpoint -> pipeline -> samples
# --------------------------------------------------------------------------

def test_pipeline_generates_from_a_local_checkpoint(tmp_path):
    """The full offline path: a trainer checkpoint, the config that run logged,
    and a sample out of the reconstructed model."""
    trainer = make_trainer(tmp_path)
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    state, best_state = load_from_checkpoint(
        get_latest_checkpoint(trainer.checkpoint_path()))
    assert state is not None and {"params", "ema_params"} <= set(state)

    pipeline = DiffusionInferencePipeline.create(
        config=parse_config(current_config()), state=state, best_state=best_state)
    samples = pipeline.generate_samples(
        num_samples=2,
        resolution=RES,
        diffusion_steps=3,
        guidance_scale=0.0,
        sampler_class=EulerAncestralSampler,
        seed=0,
    )

    assert samples.shape == (2, RES, RES, 3)
    assert np.all(np.isfinite(samples))
    # post_process clips into the image range
    assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_pipeline_can_sample_from_the_ema_parameters(tmp_path):
    """The EMA copy is what a real run publishes, so it has to be reachable
    through the same call."""
    trainer = make_trainer(tmp_path, name="inference-ema")
    trainer.save(epoch=0, step=1)
    trainer.wait_for_checkpoints()

    state, _ = load_from_checkpoint(get_latest_checkpoint(trainer.checkpoint_path()))
    pipeline = DiffusionInferencePipeline.create(
        config=parse_config(current_config()), state=state)
    samples = pipeline.generate_samples(
        num_samples=2, resolution=RES, diffusion_steps=3, guidance_scale=0.0,
        sampler_class=EulerAncestralSampler, seed=0, use_ema=True)

    assert samples.shape == (2, RES, RES, 3)
    assert np.all(np.isfinite(samples))


def test_missing_checkpoint_is_reported(tmp_path):
    """A load that failed must say so rather than hand back an empty pair that
    reads as a successful one."""
    with pytest.raises(FileNotFoundError):
        load_from_checkpoint(str(tmp_path / "nothing-here"))


# --------------------------------------------------------------------------
# parse_config across config generations
# --------------------------------------------------------------------------

def test_parse_config_current_generation():
    parsed = parse_config(current_config())

    assert type(parsed["model"]).__name__ == "SimpleDiT"
    assert parsed["model"].emb_features == 16
    assert parsed["architecture"] == "simple_dit"
    assert parsed["autoencoder"] is None
    assert isinstance(parsed["input_config"], DiffusionInputConfig)
    assert parsed["input_config"].sample_data_shape == (RES, RES, 3)
    assert parsed["input_config"].conditions == []

    _, sampling_schedule, transform = get_diffusion_preset("edm")
    assert type(parsed["noise_schedule"]) is type(sampling_schedule)
    assert type(parsed["prediction_transform"]) is type(transform)


def test_parse_config_current_generation_rebuilds_conditions(monkeypatch):
    """A serialized condition has to come back with its pretokenization flag,
    which decides whether inference tokenizes or embeds directly."""
    monkeypatch.setitem(CONDITIONAL_ENCODERS_REGISTRY, "text", StubTextEncoder)
    config = current_config()
    config["input_config"]["conditions"] = [{
        "encoder": {"modelname": "openai/clip-vit-large-patch14"},
        "encoder_key": "text",
        "conditioning_data_key": "text",
        "pretokenized": True,
        "unconditional_input": "",
        "model_key_override": "textcontext",
    }]

    condition = parse_config(config)["input_config"].conditions[0]
    assert condition.pretokenized is True
    assert condition.model_key_override == "textcontext"
    assert condition.get_unconditional().shape == (1, 77, 768)


def test_parse_config_legacy_generation(monkeypatch):
    """No input_config at all: the text conditioning is reconstructed from the
    image size and the era's fixed conventions."""
    monkeypatch.setattr(inference_utils, "defaultTextEncodeModel",
                        lambda: StubTextEncoder(model="clip", tokenizer=None))
    parsed = parse_config(legacy_config())

    assert type(parsed["model"]).__name__ == "Unet"
    assert parsed["architecture"] == "unet"
    assert parsed["autoencoder"] is None

    input_config = parsed["input_config"]
    assert input_config.sample_data_key == "image"
    assert input_config.sample_data_shape == (128, 128, 3)

    condition, = input_config.conditions
    assert condition.conditioning_data_key == "text"
    assert condition.pretokenized is True
    assert condition.model_key_override == "textcontext"
    assert condition.unconditional_input == ""

    _, sampling_schedule, transform = get_diffusion_preset("cosine")
    assert type(parsed["noise_schedule"]) is type(sampling_schedule)
    assert type(parsed["prediction_transform"]) is type(transform)


def test_parse_config_overrides_reach_both_levels():
    parsed = parse_config(current_config(), overrides={"noise_schedule": "cosine"})

    _, sampling_schedule, _ = get_diffusion_preset("cosine")
    assert type(parsed["noise_schedule"]) is type(sampling_schedule)
    assert parsed["raw_config"]["arguments"]["noise_schedule"] == "cosine"


# --------------------------------------------------------------------------
# Public surface and the wandb entry points
# --------------------------------------------------------------------------

def test_package_exports_the_public_names():
    for name in ("InferencePipeline", "DiffusionInferencePipeline", "parse_config",
                 "load_from_checkpoint", "load_from_wandb_run",
                 "load_from_wandb_registry", "get_wandb_run"):
        assert hasattr(inference, name), name


def test_base_pipeline_advertises_no_loader():
    """Loading is the concrete pipeline's business; the base class used to
    declare a from_wandb that only ever raised."""
    assert not hasattr(InferencePipeline, "from_wandb")
    assert hasattr(DiffusionInferencePipeline, "from_wandb_run")
    assert hasattr(DiffusionInferencePipeline, "from_wandb_registry")


def test_wandb_run_loader_reports_a_miss(monkeypatch):
    """A run that cannot be found returns Nones. The artifact name used to be
    referenced in the return before it was ever bound."""
    monkeypatch.setattr(inference_utils, "get_wandb_run", lambda *a, **k: None)

    assert inference_utils.load_from_wandb_run(
        "no-such-run", project="p", entity="e") == (None, None, None, None)


def test_wandb_registry_loader_reports_a_miss(monkeypatch):
    def unreachable(*args, **kwargs):
        raise RuntimeError("wandb is unreachable")

    monkeypatch.setattr(inference_utils.wandb, "Api", unreachable)

    assert inference_utils.load_from_wandb_registry(
        "no-such-model", project="p") == (None, None, None, None)
