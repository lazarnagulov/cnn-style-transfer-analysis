"""
Loss modules for neural style transfer.

This module implements content and style loss layers as described in:

    Gatys et al., "Image Style Transfer Using Convolutional Neural Networks"

The losses are designed to be inserted into a CNN model. Each module
computes its respective loss during the forward pass while returning
the input features unchanged.

Classes:
    ContentLoss: Computes MSE loss between feature maps.
    StyleLoss: Computes MSE loss between Gram matrices of feature maps.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import gram_matrix as G

class ContentLoss(nn.Module):
    """
    Content loss module used in neural style transfer.

    Computes the Mean Squared Error (MSE) between the input feature maps
    and a target feature representation extracted from a content image.

    Args:
        target_feature (Tensor): Feature map of shape (B, C, H, W)
            extracted from the content image.
    """

    def __init__(self, target_feature: Tensor) -> None:
        super(ContentLoss, self).__init__()
        self.target = target_feature.detach()

    def forward(self, features: Tensor) -> Tensor:
        """
        Computes content loss.

        Args:
            features (Tensor): Feature map of shape (B, C, H, W)
                from the generated image.

        Returns:
            Tensor: The input features (unchanged).
        """
        self.loss = F.mse_loss(features, self.target)
        return features
    
class StyleLoss(nn.Module):
    """
    Style loss module used in neural style transfer.

    Computes the Mean Squared Error (MSE) between the Gram matrix of
    the input feature maps and the Gram matrix of a target style image.

    Args:
        target_feature (Tensor): Feature map of shape (B, C, H, W)
            extracted from the style image.
    """
    
    def __init__(self, target_feature: Tensor) -> None:
        super(StyleLoss, self).__init__()
        self.target = G.create_gram_matrix(target_feature).detach()

    def forward(self, features: Tensor) -> Tensor:
        """
        Computes style loss.

        Args:
            features (Tensor): Feature map of shape (B, C, H, W)
                from the generated image.

        Returns:
            Tensor: The input features (unchanged).
        """
        gram_matrix = G.create_gram_matrix(features)
        self.loss = F.mse_loss(gram_matrix, self.target)
        return features