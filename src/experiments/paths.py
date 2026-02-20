"""
Filesystem helpers and paths for Neural Style Transfer experiments.

This module defines important directories used across experiments,
including the base project directory and results output directory.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")