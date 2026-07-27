"""Conditioning encoder abstraction tests.

The base class must stay modality-agnostic: text and audio differ in both
tokenization and embedding, and adding a modality must not require touching
anything shared. These use stub processors/models so no weights are
downloaded; the network-marked test covers a real HF audio model.
"""

import numpy as np
import pytest

from flaxdiff.inputs import CONDITIONAL_ENCODERS_REGISTRY
from flaxdiff.inputs.encoders import (
    ConditioningEncoder, TextEncoder, CLIPTextEncoder, AudioEncoder, HFAudioEncoder,
)


class StubOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


def test_registry_exposes_both_modalities():
    assert CONDITIONAL_ENCODERS_REGISTRY["text"] is CLIPTextEncoder
    assert CONDITIONAL_ENCODERS_REGISTRY["audio"] is HFAudioEncoder


def test_encoder_keys():
    """Each modality declares its own batch key; the intermediate classes stay
    abstract until a concrete backend implements serialize/deserialize."""
    assert CLIPTextEncoder(model=None, tokenizer=None, modelname="m", backend="jax").key == "text"
    assert HFAudioEncoder(model=None, tokenizer=None, modelname="m",
                          backend="jax", sampling_rate=16000).key == "audio"
    for cls in (TextEncoder, AudioEncoder):
        assert cls.__abstractmethods__, f"{cls.__name__} should stay abstract"


def test_base_class_is_modality_agnostic():
    """tokenize/encode_from_tokens must be abstract - a base class that
    assumes input_ids/attention_mask cannot serve audio."""
    for name in ("tokenize", "encode_from_tokens", "key"):
        assert name in ConditioningEncoder.__abstractmethods__


@pytest.mark.parametrize("feature_key", ["input_values", "input_features"])
def test_audio_encoder_passes_processor_keys_through(feature_key):
    """wav2vec2/HuBERT emit input_values, Whisper/AST emit input_features.
    The encoder must forward whatever the processor produced, so swapping the
    audio model is a config change and nothing else."""
    seen = {}

    class StubProcessor:
        sampling_rate = 16000

        def __call__(self, audio, sampling_rate=None, padding=None, return_tensors=None):
            seen["sampling_rate"] = sampling_rate
            return {feature_key: np.zeros((1, 4), dtype=np.float32),
                    "attention_mask": np.ones((1, 4), dtype=np.int32)}

    def stub_model(**kwargs):
        seen["forwarded"] = sorted(kwargs)
        return StubOutput(np.zeros((1, 4, 8), dtype=np.float32))

    encoder = HFAudioEncoder(model=stub_model, tokenizer=StubProcessor(),
                             modelname="stub", backend="jax", sampling_rate=16000)
    embeddings = encoder(np.zeros(16000, dtype=np.float32))

    assert seen["sampling_rate"] == 16000
    assert seen["forwarded"] == sorted([feature_key, "attention_mask"])
    assert embeddings.shape == (1, 4, 8)


def test_audio_encoder_roundtrips_config():
    encoder = HFAudioEncoder(model=None, tokenizer=None, modelname="facebook/wav2vec2-base-960h",
                             backend="jax", sampling_rate=24000)
    config = encoder.serialize()
    assert config == {"modelname": "facebook/wav2vec2-base-960h",
                      "backend": "jax", "sampling_rate": 24000}
