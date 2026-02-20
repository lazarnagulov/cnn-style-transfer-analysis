"""
Experiment configuration utilities for Neural Style Transfer (NST).

This module provides:

- `ExperimentConfig`: A dataclass that stores all configuration
  parameters for an NST experiment, including content/style images,
  optimization hyperparameters, and layer selections.
- YAML support: Load experiment configuration from a YAML file
  containing required and optional parameters.
- CLI support: Override YAML settings or provide configuration entirely
  via command-line arguments.
- Merging logic: CLI arguments take precedence over YAML defaults,
  which in turn fallback to dataclass defaults.
"""
import argparse
from dataclasses import dataclass
from typing import List, Optional

import yaml

from .paths import BASE_DIR, RESULTS_DIR


@dataclass
class ExperimentConfig:
    """
    Configuration container for a Neural Style Transfer experiment.

    This class manages all parameters needed for a single NST run,
    including content and style images, optimization settings, and
    which layers to use for computing content and style losses.

    Supports both:
        - YAML-based configuration
        - CLI argument overrides
    """
    content_image: str
    style_image: str
    image_size: int = 512
    steps: int = 400
    alpha: float = 1.0
    beta: float = 1_000_000.0
    output_path: str = RESULTS_DIR
    content_layers: Optional[List[str]] = None
    style_layers: Optional[List[str]] = None

    @classmethod
    def parse(cls) -> "ExperimentConfig":
        """
        Parse configuration from CLI arguments or YAML file.

        The precedence is:
            1. CLI arguments override
            2. YAML configuration
            3. Dataclass defaults

        Returns:
            ExperimentConfig: Fully populated experiment configuration.

        Raises:
            SystemExit: if required arguments are missing.
            ValueError: if required fields are missing from YAML.
        """
        parser = cls.create_args_parser()
        args = parser.parse_args()

        if args.config:
            yaml_config = cls.from_yaml(args.config)
            cli_args = {
                k: v for k, v in vars(args).items()
                if k in cls.__dataclass_fields__ and v is not None
            }
            return cls(**{**yaml_config.__dict__, **cli_args})

        if not args.content_image or not args.style_image:
            parser.error("either --config or both --content_image and --style_image must be provided")

        return cls(
            content_image=BASE_DIR + args.content_image,
            style_image=BASE_DIR + args.style_image,
            image_size=args.image_size if args.image_size is not None else 512,
            steps=args.steps if args.steps is not None else 400,
            alpha=args.alpha if args.alpha is not None else 1.0,
            beta=args.beta if args.beta is not None else 1_000_000.0,
            output_path=BASE_DIR  + args.output_path,
            content_layers=args.content_layers,
            style_layers=args.style_layers
        )


    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """
        Load experiment configuration from a YAML file.

        Required fields in YAML:
            - content_image
            - style_image

        Optional fields have default values:
            - image_size, steps, alpha, beta
            - content_layers, style_layers
            - output_path

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            ExperimentConfig: Populated configuration object.

        Raises:
            ValueError: If required fields are missing from the YAML file.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        required_fields = ["content_image", "style_image"]

        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"missing required fields in config file '{path}': {missing}")

        return cls(
            content_image=data["content_image"],
            style_image=data["style_image"],
            image_size=data.get("image_size", 512),
            steps=data.get("steps", 400),
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1_000_000.0),
            output_path=data.get("output_path", RESULTS_DIR),
            content_layers=data.get("content_layers"),
            style_layers=data.get("style_layers"),
        )
        
    @staticmethod
    def create_args_parser() -> argparse.ArgumentParser:
        """
        Create an argument parser for CLI usage.

        Returns:
            argparse.ArgumentParser: Configured parser for NST experiment.
        """
        parser = argparse.ArgumentParser(description="Neural Style Transfer experiment configuration.")

        parser.add_argument("--config", type=str, help="path to YAML config file.")
        parser.add_argument("--content_image", type=str, help="path to the content image.")
        parser.add_argument("--style_image", type=str, help="path to the style image.")
        parser.add_argument("--image_size", type=int, help="image resize dimension (default: 512).")
        parser.add_argument("--steps", type=int, help="optimization steps (default: 400).")
        parser.add_argument("--alpha", type=float, help="content loss weight (default: 1.0).")
        parser.add_argument("--beta", type=float, help="style loss weight (default: 1e6).")
        parser.add_argument("--output_path", type=str, help="output path. (default: ./results/)")
        parser.add_argument("--content_layers", nargs="+", help="content loss layers (e.g. conv4_2).")
        parser.add_argument("--style_layers", nargs="+", help="style loss layers.")
        return parser