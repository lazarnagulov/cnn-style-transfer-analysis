"""
Model utilities for neural style transfer.

This module provides:

- `Normalization`: a module to normalize images using ImageNet mean and std.
- `create_style_transfer_model`: builds the style transfer model by inserting content and style loss layers
  at specified positions.

"""
from typing import List, Tuple
import torch
from torch import Tensor
import torch.nn as nn

from torchvision.models import vgg19, VGG19_Weights, VGG

from losses import ContentLoss, StyleLoss
from utils import CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD

class Normalization(nn.Module):
    """
    Normalizes a batch of images using ImageNet mean and std.

    This is required when using pretrained CNNs.
    """
    
    def __init__(self) -> None:
        super(Normalization, self).__init__()
        self.mean = torch.tensor(CNN_NORMALIZATION_MEAN).view(-1, 1, 1)
        self.std = torch.tensor(CNN_NORMALIZATION_STD).view(-1, 1, 1)

    def forward(self, img: Tensor) -> Tensor:
        return (img - self.mean) / self.std

def create_style_transfer_model(
    style_img: Tensor,
    content_img: Tensor,
    content_layers: List[str] = ["conv4_2"],
    style_layers: List[str] = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
) -> Tuple[nn.Sequential, List[ContentLoss], List[StyleLoss]]:
    """
    Build a style transfer model with inserted content and style loss layers.

    Args:
        style_img (Tensor): Preprocessed style image tensor (C, H, W) or (1, C, H, W).
        content_img (Tensor): Preprocessed content image tensor (C, H, W) or (1, C, H, W).
        content_layers (List[str], optional): Names of layers to compute content loss.
        style_layers (List[str], optional): Names of layers to compute style loss.

    Returns:
        model (nn.Sequential): VGG-based model with Normalization + loss layers.
        content_losses (List[ContentLoss]): List of ContentLoss modules.
        style_losses (List[StyleLoss]): List of StyleLoss modules.
    """
    raise NotImplementedError


def _load_model(weights: VGG19_Weights = VGG19_Weights.DEFAULT) -> VGG:
    """
    Loads pretrained VGG19 features for style transfer.

    Args:
        weights: Pretrained weights to use (default: ImageNet).

    Returns:
        VGG feature extractor in eval mode.
    """
    return vgg19(weights=weights).features.eval()
