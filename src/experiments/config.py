from dataclasses import dataclass
import os
from typing import List, Optional

import yaml

@dataclass
class ExperimentConfig:
    content_image: str
    style_image: str
    image_size: int = 512
    steps: int = 400
    alpha: float = 1.0
    beta: float = 1_000_000.0
    content_layers: Optional[List[str]] = None
    style_layers: Optional[List[str]] = None
    
    @staticmethod
    def parse_from_arguments() -> 'ExperimentConfig':
        import argparse
    
        parser = argparse.ArgumentParser()
        parser.add_argument("--content", required=True)
        parser.add_argument("--style", required=True)
        parser.add_argument("--steps", type=int, default=400)
        parser.add_argument("--alpha", type=float, default=1.0)
        parser.add_argument("--beta", type=float, default=1e6)
        
        args = parser.parse_args()

        return ExperimentConfig(
            content_image=args.content,
            style_image=args.style,
            steps=args.steps,
            alpha=args.alpha,
            beta=args.beta,
        )

