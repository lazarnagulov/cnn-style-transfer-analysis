"""
Experiment runner for Neural Style Transfer (NST).

This module provides the `run_experiment` function, which executes a full
style transfer experiment using a provided `ExperimentConfig`. It handles:

- Device setup (CPU or GPU)
- Image loading
- NST optimization using `run_style_transfer`
- Saving the output image to disk
"""
import torch

from nst.model import run_style_transfer
from nst.utils import load_image, save_image

from .config import ExperimentConfig


def run_experiment(config: ExperimentConfig) -> None:
    """
    Execute a full style transfer experiment.

    Args:
        config (ExperimentConfig): Configuration object containing
            content/style image paths, optimization parameters,
            layer selections, and output path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    imsize = config.image_size
    
    style_img   = load_image(config.style_image, imsize, device)
    content_img = load_image(config.content_image, imsize, device)
    input_img   = content_img.clone()

    output = run_style_transfer(
        content_img=content_img, 
        style_img=style_img, 
        input_img=input_img,
        steps=config.steps,
        alpha=config.alpha,
        beta=config.beta,
        log_every=20,
    )
    save_image(output, config.output_path)
