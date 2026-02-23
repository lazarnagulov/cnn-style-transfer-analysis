"""
Experiment runner for Neural Style Transfer (NST).

This module provides the `run_experiment` function, which executes a full
style transfer experiment using a provided `ExperimentConfig`. It handles:

- Device setup (CPU or GPU)
- Image loading
- NST optimization using `run_style_transfer`
- Saving the output image to disk
"""
from typing import cast

import torch

from nst.model import StyleTransferResult, run_style_transfer
from nst.utils import load_image, save_image, save_result

from .config import ExperimentConfig


def run_experiment(config: ExperimentConfig, return_history: bool = True) -> torch.Tensor:
    """
    Execute a full style transfer experiment.

    Args:
        config (ExperimentConfig): Configuration object containing
            content/style image paths, optimization parameters,
            layer selections, and output path.
        return_history (bool, optional):
            If True, record content, style, and total loss values at each
            optimization step and save them alongside the final image.
            Defaults to True.
    Returns:
        Tensor: The optimized image Tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    imsize = config.image_size
    
    style_img   = load_image(config.style_image, imsize, device)
    content_img = load_image(config.content_image, imsize, device)
    input_img   = content_img.clone()

    result = run_style_transfer(
        content_img=content_img, 
        style_img=style_img, 
        input_img=input_img,
        steps=config.steps,
        alpha=config.alpha,
        beta=config.beta,
        style_layers=config.style_layers,
        content_layers=config.content_layers,
        return_history=return_history,
        log_every=20,
    )
    
    if isinstance(result, StyleTransferResult):
        save_image(result.image, config.output_path)
        save_result(result, config.output_path)
        return result.image
    else:
        return result
