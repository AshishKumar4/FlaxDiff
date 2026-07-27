import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ConditioningEncoder(ABC):
    """A modality's path from raw data to conditioning embeddings.

    `tokenize` turns raw data into model inputs (runs on CPU in the data
    pipeline workers), `encode_from_tokens` turns those into embeddings on
    device. Modalities differ in both, so both are abstract here - nothing in
    this base class assumes text.
    """
    model: Any
    tokenizer: Callable

    @property
    @abstractmethod
    def key(self):
        """Batch/config key this encoder reads and writes."""
        pass

    def __call__(self, data):
        tokens = self.tokenize(data)
        outputs = self.encode_from_tokens(tokens)
        return outputs

    @abstractmethod
    def encode_from_tokens(self, tokens):
        """Embed already-tokenized inputs."""
        pass

    @abstractmethod
    def tokenize(self, data):
        """Turn raw data into model inputs."""
        pass

    @abstractmethod
    def serialize(self):
        """Serialize the encoder configuration."""
        pass
    
    @staticmethod
    @abstractmethod
    def deserialize(serialized_config):
        """Deserialize the encoder configuration."""
        pass
    
@dataclass
class TextEncoder(ConditioningEncoder):
    """Text Encoder."""
    @property
    def key(self):
        return "text"

    def tokenize(self, data):
        tokens = self.tokenizer(data, padding="max_length",
                        max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="np")
        return tokens

    def encode_from_tokens(self, tokens):
        outputs = self.model(input_ids=tokens['input_ids'],
                        attention_mask=tokens['attention_mask'])
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state
    
@dataclass
class CLIPTextEncoder(TextEncoder):
    """CLIP Text Encoder."""
    modelname: str
    backend: str
    
    @staticmethod
    def from_modelname(modelname: str = "openai/clip-vit-large-patch14", backend: str="jax"):
        from transformers import (
            CLIPTextModel,
            FlaxCLIPTextModel,
            AutoTokenizer,
        )
        if backend == "jax":
            model = FlaxCLIPTextModel.from_pretrained(
                modelname, dtype=jnp.bfloat16)
        else:
            model = CLIPTextModel.from_pretrained(modelname)
        tokenizer = AutoTokenizer.from_pretrained(modelname, dtype=jnp.float16)
        return CLIPTextEncoder(
            model=model,
            tokenizer=tokenizer,
            modelname=modelname,
            backend=backend
        )
    
    def serialize(self):
        """Serialize the encoder configuration."""
        serialized_config = {
            "modelname": self.modelname,
            "backend": self.backend,
        }
        return serialized_config
    
    @staticmethod
    def deserialize(serialized_config):
        """Deserialize the encoder configuration."""
        modelname = serialized_config["modelname"]
        backend = serialized_config["backend"]
        return CLIPTextEncoder.from_modelname(modelname=modelname, backend=backend)
    
@dataclass
class AudioEncoder(ConditioningEncoder):
    """Audio conditioning."""
    @property
    def key(self):
        return "audio"


@dataclass
class HFAudioEncoder(AudioEncoder):
    """Any HuggingFace audio model, resolved through the Auto* classes.

    Works with wav2vec2/HuBERT (`input_values`), Whisper/AST
    (`input_features`) and anything else whose processor and model are
    registered with transformers - whatever keys the processor emits are
    forwarded to the model unchanged, so swapping the audio model is a config
    change and nothing more. Audio *file* formats are handled upstream by
    `flaxdiff.data.sources.audio_utils.read_audio`, which decodes via ffmpeg
    and resamples to `sampling_rate`.
    """
    modelname: str
    backend: str
    sampling_rate: int

    @staticmethod
    def from_modelname(modelname: str = "facebook/wav2vec2-base-960h",
                       backend: str = "jax", sampling_rate: int = None):
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(modelname)
        if backend == "jax":
            from transformers import FlaxAutoModel
            model = FlaxAutoModel.from_pretrained(modelname, dtype=jnp.bfloat16)
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(modelname)
        if sampling_rate is None:
            # The processor knows the rate its model was trained at
            sampling_rate = getattr(processor, "sampling_rate", 16000)
        return HFAudioEncoder(
            model=model,
            tokenizer=processor,
            modelname=modelname,
            backend=backend,
            sampling_rate=sampling_rate,
        )

    def tokenize(self, data):
        return dict(self.tokenizer(
            data, sampling_rate=self.sampling_rate,
            padding=True, return_tensors="np"))

    def encode_from_tokens(self, tokens):
        outputs = self.model(**tokens)
        return outputs.last_hidden_state

    def serialize(self):
        return {
            "modelname": self.modelname,
            "backend": self.backend,
            "sampling_rate": self.sampling_rate,
        }

    @staticmethod
    def deserialize(serialized_config):
        return HFAudioEncoder.from_modelname(
            modelname=serialized_config["modelname"],
            backend=serialized_config["backend"],
            sampling_rate=serialized_config.get("sampling_rate"),
        )

CONDITIONAL_ENCODERS_REGISTRY = {
    "text": CLIPTextEncoder,
    "audio": HFAudioEncoder,
}
