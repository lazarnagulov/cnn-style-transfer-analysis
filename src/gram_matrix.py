"""
Gram matrix computation utility for neural style transfer.

This module provides a helper function to compute the Gram matrix
of convolutional feature maps. The Gram matrix encodes channel-wise
feature correlations and is commonly used to represent image style
in neural style transfer methods.
"""
from torch import Tensor
import torch


def create_gram_matrix(features: Tensor) -> Tensor:
    """
    Compute the Gram matrix of feature maps.

    The Gram matrix captures correlations between feature channels and is
    commonly used in neural style transfer to represent image style.

    Args:
        features (Tensor): Feature tensor of shape (B, C, H, W),
            where:
                B = batch size
                C = number of channels
                H = height
                W = width

    Returns:
        Tensor: Normalized Gram matrix of shape (B*C, B*C).
    """
    b, c, h, w = features.size()
    matrix_features = features.view(b * c, h * w)
    gram = torch.mm(matrix_features, matrix_features.t())
    return gram.div(b * c * h * w)

