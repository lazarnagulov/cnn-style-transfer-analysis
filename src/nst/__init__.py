"""
Neural Style Transfer (NST) implementation using PyTorch.

This package provides a modular implementation of the neural style transfer
algorithm introduced by Gatys et al., enabling the synthesis of images that
combine the content of one image with the artistic style of another.

The implementation is structured into reusable components including
Gram matrix computation, content and style loss modules, model construction
utilities, optimization routines and image preprocessing helpers.

This module provides:
- Gram matrix computation for feature correlation extraction
- Content and style loss modules compatible with CNN feature extractors
- Model construction utilities based on pretrained VGG19 features
- Optimization pipeline for performing style transfer
- Image loading, normalization, and saving utilities
"""
