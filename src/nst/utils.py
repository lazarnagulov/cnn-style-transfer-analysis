"""
Utility functions and constants for neural style transfer.

This module provides:
- Image normalization constants for pretrained CNNs (e.g., ImageNet models)
- Utility functions for loading and preprocessing images

Typical usage:
    from utils import CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
"""
import csv
import os
from typing import Union
import torch
import torchvision.transforms as transforms
from PIL import Image

from nst.result import StyleTransferResult

CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406])
CNN_NORMALIZATION_STD  = torch.tensor([0.229, 0.224, 0.225])


def load_image(
    image_input: Union[str, Image.Image],
    image_size: int,
    device: torch.device
) -> torch.Tensor:
    """
    Load an image from disk or from a PIL Image object and convert it to a
    normalized tensor suitable for neural style transfer processing.

    Args:
        image_input (Union[str, Image.Image]): Path to the input image file
            or a preloaded PIL Image instance.
        image_size (int): Desired image size.
        device (torch.device): Device to load the image onto (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: Image tensor of shape (1, 3, H, W) on the specified device,
                      with pixel values scaled to [0,1].
    """
    loader = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}. Expected str or PIL.Image.Image.")

    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def save_image(tensor: torch.Tensor, path: str, filename: str = "output.png") -> None:
    """
    Saves a PyTorch tensor as an image file.

    Args:
        tensor (torch.Tensor): Image tensor of shape (1, C, H, W) or (C, H, W)
                               with values in [0, 1].
        path (str): Directory or full file path.
        filename (str): Optional filename if path is a directory.
    """
    if os.path.isdir(path) or not os.path.splitext(path)[1]:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)

    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)

    tensor = torch.clamp(tensor, 0, 1)
    pil_image = transforms.ToPILImage()(tensor.cpu())
    pil_image.save(path)

    print(f"Image saved to: {os.path.abspath(path)}")
    
def save_result(result: StyleTransferResult, path: str, name: str = "nst_result") -> None:
    """
    Save the loss history of a neural style transfer experiment.

    Args:
        result (StyleTransferResult): The result object returned by `run_style_transfer`.
        path (str): Directory or full file path.
        name (Optional[str]): Optional prefix name for files. Defaults to 'nst_result'.

    Saves:
        - Losses as {name}_losses.csv with columns: step, content_loss, style_loss, total_loss
    """
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, f"{name}_losses.csv")
    steps = len(result.total_losses)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "content_loss", "style_loss", "total_loss"])
        for i in range(steps):
            writer.writerow([
                i,
                result.content_losses[i],
                result.style_losses[i],
                result.total_losses[i]
            ])
        print(f"Saved loss history to: {csv_path}")