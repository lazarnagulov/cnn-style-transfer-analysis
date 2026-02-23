"""
Model utilities for neural style transfer.

This module provides:

- `Normalization`: a module to normalize images using ImageNet mean and std.
- `create_style_transfer_model`: builds the style transfer model by inserting content and style loss layers
  at specified positions.
- `create_optimizer`: Returns a PyTorch optimizer configured to update the input image.
  Supports LBFGS by default; will be extended to other optimizers (Adam, SGD, RMSProp) for experimentation.
- `run_style_transfer`: Performs neural style transfer on the input image.
  Optimizes the image to minimize style and content losses using the selected optimizer,
  returning the stylized output image.

"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer

from torchvision.models import vgg19, VGG19_Weights

from nst.result import StyleTransferResult

from .losses import ContentLoss, StyleLoss
from .utils import CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD


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
        """
        Apply normalization to the input image tensor.

        Args:
            img (Tensor): Input image tensor of shape (C, H, W) or
                          (N, C, H, W).

        Returns:
            Tensor: Normalized image tensor with same shape as input.
        """
        return (img - self.mean) / self.std

def create_style_transfer_model(
    style_img: Tensor,
    content_img: Tensor,
    content_layers: Optional[List[str]] = None,
    style_layers: Optional[List[str]] = None,
) -> Tuple[nn.Sequential, List[ContentLoss], List[StyleLoss]]:
    """
    Build a style transfer model with inserted content and style loss layers.

    The model is constructed from a pretrained VGG19 feature extractor.
    Content and style loss modules are inserted immediately after specified
    convolutional layers.

    Args:
        style_img (Tensor): Preprocessed style image tensor of shape
            (C, H, W) or (1, C, H, W).
        content_img (Tensor): Preprocessed content image tensor of shape
            (C, H, W) or (1, C, H, W).
        content_layers (Optional[List[str]]): Layer names at which to compute
            content loss. Defaults to ["conv4_2"].
        style_layers (Optional[List[str]]): Layer names at which to compute
            style loss. Defaults to
            ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"].

    Returns:
        Tuple:
            - model (nn.Sequential): Modified VGG-based model with inserted
              loss layers.
            - content_losses (List[ContentLoss]): Content loss modules.
            - style_losses (List[StyleLoss]): Style loss modules.
    """
    if content_layers is None:
        content_layers = ["conv4_2"]
        
    if style_layers is None:
        style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        
    model = nn.Sequential(Normalization())
    cnn: nn.Sequential = _load_model()
    
    content_losses: List[ContentLoss] = []
    style_losses: List[StyleLoss] = []
    block_count: int = 1
    conv_in_block_count: int = 0
 
    for layer in cnn.children():
        name: str
        match layer:
            case nn.Conv2d():
                conv_in_block_count += 1
                name = f"conv{block_count}_{conv_in_block_count}"
            case nn.ReLU():
                name = f"relu{block_count}_{conv_in_block_count}"
                layer = nn.ReLU(inplace=False)
            case nn.MaxPool2d():
                name = f"pool{block_count}_{conv_in_block_count}"
                block_count += 1
                conv_in_block_count = 0
            case nn.BatchNorm2d():
                name = f"batch_norm{block_count}_{conv_in_block_count}"
            case _:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
        
        model.add_module(name, layer)
        if name in content_layers:
            content_losses.append(_insert_content_loss_layer(model, content_img, f"content_loss{block_count}_{conv_in_block_count}"))
        
        if name in style_layers:
            style_losses.append(_insert_style_loss_layer(model, style_img, f"style_loss{block_count}_{conv_in_block_count}"))

    return _trim_model(model), content_losses, style_losses

def create_optimizer(input_img: Tensor) -> Optimizer:
    """
    Create the optimizer used for neural style transfer.

    Args:
        input_img (Tensor): The image tensor to be optimized. Must have
            ``requires_grad=True``.

    Returns:
        Optimizer: An instance of ``torch.optim.LBFGS`` configured to
        update the input image.
    """
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(
    content_img: Tensor,
    style_img: Tensor,
    input_img: Tensor,
    steps: int = 400,
    alpha: float = 1.0,
    beta: float = 1_000_000.0,
    style_layers: Optional[List[str]] = None,
    content_layers: Optional[List[str]] = None,
    return_history: bool = False,
    log_every: int = 0,
) -> Union[Tensor, StyleTransferResult]:
    """
    Run neural style transfer optimization.

    This function constructs the style transfer model and optimizes the
    `input_img` to simultaneously minimize weighted content and style losses.
    Optimization is performed in-place on `input_img` using L-BFGS.

    Optionally, loss values can be recorded during optimization for later
    analysis or plotting.

    Args:
        content_img (Tensor): Preprocessed content image tensor of shape (1, C, H, W).
        style_img (Tensor): Preprocessed style image tensor of shape (1, C, H, W).
        input_img (Tensor):
            Initial image tensor to optimize. Typically initialized as a copy
            of the content image or as random noise. This tensor is modified
            in-place.
        steps (int, optional): Number of optimization iterations. Defaults to 400.
        alpha (float, optional): Weight assigned to the content loss. Defaults to 1.0.
        beta (float, optional): Weight assigned to the style loss. Defaults to 1_000_000.0.
        content_layers (Optional[List[str]]): Layer names at which to compute
            content loss. Defaults to ["conv4_2"].
        style_layers (Optional[List[str]]): Layer names at which to compute
            style loss. Defaults to
            ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"].
        return_history (bool, optional):
            If True, record content, style, and total loss values at each
            optimization step and return them alongside the final image.
            Defaults to False.
        log_every (int, optional):
            If greater than 0, prints progress every `log_every` iterations.
            If 0, disables progress logging. Defaults to 0.

    Returns:
        - If `return_history` is False, returns the optimized image `Tensor` otherwise 
        returns a `StyleTransferResult` 
    """
    
    if content_layers is None:
        content_layers = ["conv4_2"]
        
    if style_layers is None:
        style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    
    model, content_losses, style_losses = create_style_transfer_model(
        style_img=style_img, 
        content_img=content_img,
        style_layers=style_layers,
        content_layers=content_layers
    )
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = create_optimizer(input_img)
    
    content_history: List[float] = []
    style_history: List[float]   = []
    total_history: List[float]   = []
    
    def style_transfer_step():
        """
        Perform a single optimization step for L-BFGS.

        This closure:
            - Clamps the image to valid pixel range
            - Computes forward pass
            - Accumulates content and style losses
            - Performs backward pass

        Returns:
            Tensor: Scalar total loss tensor used by L-BFGS.
        """
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        optimizer.zero_grad()
        model(input_img)
        content_loss = alpha * sum(cl.loss for cl in content_losses)
        style_loss   = beta  * sum(sl.loss for sl in style_losses)
        
        total_loss = style_loss + content_loss
        total_loss.backward() # ty: ignore

        if return_history:
            content_history.append(content_loss.item())
            style_history.append(style_loss.item())
            total_history.append(total_loss.item())
            
        return total_loss

    for i in range(steps):
        if log_every > 0 and  i % log_every == 0:
            print(f"Step {i}/{steps}")
        optimizer.step(style_transfer_step)
    
    with torch.no_grad():
        input_img.clamp_(0,1)
        input_img.requires_grad_(False)
    
    if return_history:
        return StyleTransferResult(
            image=input_img,
            content_losses=content_history,
            style_losses=style_history,
            total_losses=total_history,
        )
    else:
        return input_img

def _insert_content_loss_layer(model: nn.Sequential, content_img: Tensor, name: str) -> ContentLoss:
    """
    Insert a ContentLoss module into the model.

    The target feature map is computed by forwarding the content image
    through the model up to the current layer. The resulting activation
    is detached to prevent gradient tracking.

    Args:
        model (nn.Sequential): Partially constructed model.
        content_img (Tensor): Preprocessed content image tensor.
        name (str): Name assigned to the inserted loss module.

    Returns:
        ContentLoss: The created content loss module.
    """
    
    target = model(content_img).detach()
    content_loss = ContentLoss(target)
    model.add_module(name, content_loss)
    return content_loss

def _insert_style_loss_layer(model: nn.Sequential, style_img: Tensor, name: str) -> StyleLoss:
    """
    Insert a StyleLoss module into the model.

    The target feature map is computed by forwarding the style image
    through the model up to the current layer. The resulting activation
    is detached before being used to initialize the StyleLoss module.

    Args:
        model (nn.Sequential): Partially constructed model.
        style_img (Tensor): Preprocessed style image tensor.
        name (str): Name assigned to the inserted loss module.

    Returns:
        StyleLoss: The created style loss module.
    """
    target_feature = model(style_img).detach()
    style_loss = StyleLoss(target_feature)
    model.add_module(name, style_loss)
    return style_loss

def _trim_model(model: nn.Sequential) -> nn.Sequential:
    """
    Trim the model after the last loss layer.

    Removes all layers that appear after the final ContentLoss or
    StyleLoss module.

    Args:
        model (nn.Sequential): Constructed model with inser.ted loss modules.

    Returns:
        nn.Sequential: Trimmed model ending at the last loss module.
    """
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            return nn.Sequential(*list(model.children())[: i + 1])
    return model

def _load_model(weights: VGG19_Weights = VGG19_Weights.DEFAULT) -> nn.Sequential:
    """
    Loads pretrained VGG19 features for style transfer.

    Args:
        weights: Pretrained weights to use (default: ImageNet).

    Returns:
        nn.Sequential: VGG19 feature extractor in evaluation mode.
    """
    return vgg19(weights=weights).features.eval()
