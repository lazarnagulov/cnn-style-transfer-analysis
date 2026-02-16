"""
Utility functions and constants for neural style transfer.

This module provides:
- Image normalization constants for pretrained CNNs (e.g., ImageNet models)

Typical usage:
    from utils import CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
"""
import torch

CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406])
CNN_NORMALIZATION_STD  = torch.tensor([0.229, 0.224, 0.225])
