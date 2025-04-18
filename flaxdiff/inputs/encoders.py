import jax.numpy as jnp
import flax.linen as nn
from typing import Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ConditioningEncoder(ABC):
    model: nn.Module
    tokenizer: Callable
    
    @property
    def key(self):
        name = self.tokenizer.__name__
        # Remove the 'Encoder' suffix from the name and lowercase it
        if name.endswith("Encoder"):
            name = name[:-7].lower()
        return name

    def __call__(self, data):
        tokens = self.tokenize(data)
        outputs = self.encode_from_tokens(tokens)
        return outputs
        
    def encode_from_tokens(self, tokens):
        outputs = self.model(input_ids=tokens['input_ids'],
                        attention_mask=tokens['attention_mask'])
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state
    
    def tokenize(self, data):
        tokens = self.tokenizer(data, padding="max_length",
                        max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="np")
        return tokens
    
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
        modelname = "openai/clip-vit-large-patch14"
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
    
CONDITIONAL_ENCODERS_REGISTRY = {
    "text": CLIPTextEncoder,
}
