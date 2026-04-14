from typing import Callable
from dataclasses import dataclass

@dataclass
class EvaluationMetric:
    """
    Evaluation metrics for the diffusion model.
    The function is given generated samples batch [B, H, W, C] and the original batch.
    Set higher_is_better for score/similarity metrics so the trainer tracks max instead of min.
    """
    function: Callable
    name: str
    higher_is_better: bool = False