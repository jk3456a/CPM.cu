"""
MiniCPM4 Example Package

Essential utilities for MiniCPM4 model configuration.
"""

from .utils import (
    setup_minicpm4_paths,
    create_minicpm4_parser,
    apply_minicpm4_yarn_config,
    create_minicpm4_yarn_callback
)

__all__ = [
    'setup_minicpm4_paths',
    'create_minicpm4_parser',
    'apply_minicpm4_yarn_config',
    'create_minicpm4_yarn_callback'
] 