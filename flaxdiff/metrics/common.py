from typing import Callable
from dataclasses import dataclass

@dataclass
class EvaluationMetric:
    """
    Evaluation metrics for the diffusion model.
    The function is given generated samples batch [B, H, W, C] and the original batch.

    Args:
        function: Callable taking (generated, batch) and returning a scalar metric value.
        name: Metric name used as the wandb key (will be prefixed with `val/`).
        higher_is_better: If True, the trainer will track the maximum across epochs.
            If False (default), the trainer tracks the minimum (legacy behavior).
            Set this to True for similarity/score metrics where larger = better.
    """
    function: Callable
    name: str
    higher_is_better: bool = False