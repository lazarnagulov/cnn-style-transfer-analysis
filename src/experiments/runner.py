import torch

from nst.model import run_style_transfer
from nst.utils import load_image, save_image

from .config import ExperimentConfig


def run_experiment(config: ExperimentConfig) -> None:
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
        debug=True
    )
    save_image(output, config.output_path)
