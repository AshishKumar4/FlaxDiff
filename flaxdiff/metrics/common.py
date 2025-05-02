from typing import Callable
from dataclasses import dataclass

@dataclass
class EvaluationMetric:
    """
    Evaluation metrics for the diffusion model.
    The function is given generated samples batch [B, H, W, C] and the original batch.
    """
    function: Callable
    name: str