"""
Data structures for Neural Style Transfer (NST) results.

This module defines the `StyleTransferResult` data class, which
encapsulates the outputs of a style transfer optimization run.
"""

from dataclasses import dataclass
from typing import List

from torch import Tensor


@dataclass
class StyleTransferResult:
    """
    Stores the result of a neural style transfer experiment.

    This object captures the optimized image along with
    recorded loss histories if `return_history=True` was used
    in `run_style_transfer`.

    Attributes:
        image (Tensor):
            The final stylized image tensor of shape (1, C, H, W).

        content_losses (List[float]):
            Recorded content loss values at each optimization step.

        style_losses (List[float]):
            Recorded style loss values at each optimization step.

        total_losses (List[float]):
            Recorded total loss values (content + style) at each optimization step.
    """
    image: Tensor
    content_losses: List[float]
    style_losses: List[float]
    total_losses: List[float]
