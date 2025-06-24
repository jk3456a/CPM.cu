"""
MiniCPM4 Example Package

This package contains MiniCPM4-specific configurations and examples.
"""

from .utils import (
    apply_minicpm4_defaults,
    get_minicpm4_model_paths,
    apply_minicpm4_yarn_config,
    get_minicpm4_yarn_factors,
    create_minicpm4_config,
    create_minicpm4_server_parser,
    create_minicpm4_test_parser,
    generate_haystack_prompt
)

__all__ = [
    'apply_minicpm4_defaults',
    'get_minicpm4_model_paths', 
    'apply_minicpm4_yarn_config',
    'get_minicpm4_yarn_factors',
    'create_minicpm4_config',
    'create_minicpm4_server_parser',
    'create_minicpm4_test_parser',
    'generate_haystack_prompt'
] 