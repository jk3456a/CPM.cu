#!/usr/bin/env python3
"""
MiniCPM4 Model Configuration and Utilities

Specialized configuration and utilities for MiniCPM4 models with YARN support and optimized defaults.
"""

import torch
from typing import Dict, Any, Tuple


def get_minicpm4_yarn_factors():
    """Get MiniCPM4 YARN factors for long context support"""
    return [
        0.9977997200264581, 1.014658295992452, 1.0349680404997148, 1.059429246056193,
        1.0888815016813513, 1.1243301355211495, 1.166977103606075, 1.2182568066927284,
        1.2798772354275727, 1.3538666751582975, 1.4426259039919596, 1.5489853358570191,
        1.6762658237220625, 1.8283407612492941, 2.0096956085876183, 2.225478927469756,
        2.481536379650452, 2.784415934557119, 3.1413289096347365, 3.560047844772632,
        4.048719380066383, 4.752651957515948, 5.590913044973868, 6.584005926629993,
        7.7532214876576155, 9.119754865903639, 10.704443927019176, 12.524994176518703,
        14.59739595363613, 16.93214476166354, 19.53823297353041, 22.417131025031697,
        25.568260840911098, 28.991144156566317, 32.68408069090375, 36.65174474170465,
        40.90396065611201, 45.4664008671033, 50.37147343433591, 55.6804490772103,
        61.470816952306556, 67.8622707390618, 75.00516023410414, 83.11898235973767,
        92.50044360202462, 103.57086856690864, 116.9492274587385, 118.16074567836519,
        119.18497548708795, 120.04810876261652, 120.77352815196981, 121.38182790207875,
        121.89094985353891, 122.31638758099915, 122.6714244963338, 122.9673822552567,
        123.21386397019609, 123.41898278254268, 123.58957065488238, 123.73136519024158,
        123.84917421274221, 123.94701903496814, 124.02825801299717, 124.09569231686116
    ]


def get_minicpm4_model_paths(path_prefix="openbmb"):
    """Get MiniCPM4 model paths based on configuration"""
    return {
        'base_model': f"{path_prefix}/MiniCPM4-8B",
        'quantized_model': f"{path_prefix}/MiniCPM4-8B-marlin-cpmcu", 
        'draft_model': f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec",
        'quantized_draft_model': f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu"
    }


def apply_minicpm4_defaults(config, override_sparse=True):
    """Apply MiniCPM4-specific default configuration overrides
    
    These are MiniCPM4 optimizations that override CPM.cu defaults.
    
    Args:
        config: Configuration dictionary to modify
        override_sparse: Whether to override sparse setting
    """
    # Force MiniCPM4 model type
    config["model_type"] = "minicpm4"
    
    # MiniCPM4 optimized for sparse attention (only if allowed to override)
    if override_sparse:
        config["apply_sparse"] = True
    
    # MiniCPM4-specific YARN configuration
    config["minicpm4_yarn"] = True
    
    return config


def setup_minicpm4_paths(path_prefix="openbmb", apply_quant=None, apply_eagle=None, apply_eagle_quant=None):
    """Setup MiniCPM4 model paths based on configuration
    
    Args:
        path_prefix: Model repository prefix
        apply_quant: Whether to use quantized base model (None=auto-detect)
        apply_eagle: Whether to use Eagle draft model (None=auto-detect)
        apply_eagle_quant: Whether to use quantized Eagle draft model (None=auto-detect)
    """
    paths = get_minicpm4_model_paths(path_prefix)
    
    # Determine base model path
    if apply_quant is True:
        model_path = paths['quantized_model']
    else:
        model_path = paths['base_model']
    
    # Determine draft model path
    draft_model_path = None
    if apply_eagle is True:
        if apply_eagle_quant is True:
            draft_model_path = paths['quantized_draft_model']
        else:
            draft_model_path = paths['draft_model']
    else:
        draft_model_path = None
    
    return model_path, draft_model_path


def apply_minicpm4_yarn_config(llm):
    """Apply MiniCPM4 YARN configuration to model"""
    yarn_factors = get_minicpm4_yarn_factors()
    
    # Create or modify rope_scaling configuration
    if not hasattr(llm.config, 'rope_scaling') or llm.config.rope_scaling is None:
        llm.config.rope_scaling = {}
    
    llm.config.rope_scaling['rope_type'] = 'longrope'
    llm.config.rope_scaling['long_factor'] = yarn_factors
    llm.config.rope_scaling['short_factor'] = yarn_factors
    print("Applied MiniCPM4 YARN rope_scaling parameters")


def create_minicpm4_yarn_callback():
    """Create a callback function for applying MiniCPM4 YARN configuration
    
    This allows the generic cpmcu module to apply MiniCPM4-specific configurations
    without directly depending on MiniCPM4 code.
    """
    def yarn_callback(llm):
        apply_minicpm4_yarn_config(llm)
    return yarn_callback


def create_minicpm4_parser(parser_type='server'):
    """Create argument parser with MiniCPM4 optimized defaults
    
    Args:
        parser_type: Type of parser to create ('server' or 'test')
    
    Returns:
        ArgumentParser with MiniCPM4-specific arguments added
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cpmcu.args import create_server_parser, create_test_parser, str2bool
    
    # Create base parser based on type
    if parser_type == 'server':
        parser = create_server_parser()
        parser.description = 'MiniCPM4 Optimized Server'
    elif parser_type == 'test':
        parser = create_test_parser()
        parser.description = 'MiniCPM4 Test Generation'
    else:
        raise ValueError(f"Invalid parser_type: {parser_type}. Must be 'server' or 'test'")
    
    # Add MiniCPM4-specific arguments
    minicpm4_group = parser.add_argument_group('MiniCPM4 Configuration', 'MiniCPM4-specific parameters')
    
    # Model path configuration for MiniCPM4
    minicpm4_group.add_argument('--path-prefix', '--path_prefix', type=str, default='openbmb',
                               help='Model repository prefix (default: openbmb)')
    
    # Quantization control parameters
    minicpm4_group.add_argument('--apply-quant', '--apply_quant', default=True,
                               type=str2bool, nargs='?', const=True,
                               help='Enable quantization for base model (auto-detected from model path if not specified)')
    
    minicpm4_group.add_argument('--apply-eagle', '--apply_eagle', default=True,
                               type=str2bool, nargs='?', const=True,
                               help='Enable Eagle speculative decoding (auto-detected when draft model provided if not specified)')
    
    minicpm4_group.add_argument('--apply-eagle-quant', '--apply_eagle_quant', default=True,
                               type=str2bool, nargs='?', const=True,
                               help='Enable quantization for Eagle draft model (auto-detected from draft model path if not specified)')
    
    # MiniCPM4 YARN configuration
    minicpm4_group.add_argument('--minicpm4-yarn', '--minicpm4_yarn', default=True,
                               type=str2bool, nargs='?', const=True,
                               help='Enable MiniCPM4 YARN for long context support (default: True)')
    
    # Make model-path optional for MiniCPM4 (since we use path-prefix)
    for action in parser._actions:
        if action.dest == 'model_path':
            action.required = False
            action.help = 'Path to the main model (optional for MiniCPM4, uses path-prefix if not specified)'
            break
    
    return parser


def create_minicpm4_server_parser():
    """Create argument parser with MiniCPM4 optimized defaults by extending base server parser"""
    return create_minicpm4_parser('server')


def create_minicpm4_test_parser():
    """Create argument parser with MiniCPM4 optimized defaults by extending base test parser"""
    return create_minicpm4_parser('test')


def generate_haystack_prompt(target_length_k):
    """Generate haystack prompt with pass key hidden in context
    
    Args:
        target_length_k: Target length in thousands of tokens
    
    Returns:
        Generated haystack prompt string
    """
    # Simple calculation based on target length
    a = target_length_k * 16  # Scale factor for before text
    b = target_length_k * 33  # Scale factor for after text
    
    # Fixed pass key from original implementation
    digits = 681725493
    
    head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
    before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * a
    needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
    after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * b
    query = "Now, give me the exact number of the pass key. The pass key is "
    
    return head + before + needle + after + query


def create_minicpm4_config(args=None, path_prefix="openbmb", minicpm4_yarn=True, **overrides):
    """Create complete MiniCPM4 configuration with model paths
    
    Args:
        args: Parsed arguments from argparse (contains all standard CPM.cu parameters)
        path_prefix: Model repository prefix for MiniCPM4 models
        minicpm4_yarn: Enable MiniCPM4 YARN configuration
        **overrides: Additional configuration overrides
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cpmcu.args import args_to_config
    
    # Convert arguments to configuration
    if args is not None:
        # Use standard CPM.cu argument processing
        config = args_to_config(args, is_server=False)
        
        # Check which parameters were explicitly set by user
        import sys
        user_args = ' '.join(sys.argv[1:])  # Get command line arguments as string
        user_set_sparse = ('--apply-sparse' in user_args)
        
        # Apply MiniCPM4-specific defaults only for parameters not explicitly set
        config = apply_minicpm4_defaults(config, override_sparse=not user_set_sparse)
        
        # Extract MiniCPM4-specific parameters from args
        apply_quant = getattr(args, 'apply_quant', None)
        apply_eagle = getattr(args, 'apply_eagle', None)
        apply_eagle_quant = getattr(args, 'apply_eagle_quant', None)
        if hasattr(args, 'path_prefix'):
            path_prefix = args.path_prefix
        if hasattr(args, 'minicpm4_yarn'):
            minicpm4_yarn = args.minicpm4_yarn
    else:
        # Create minimal config with MiniCPM4 defaults
        config = {}
        config = apply_minicpm4_defaults(config)
        apply_quant = None
        apply_eagle = None
        apply_eagle_quant = None
    
    # Setup MiniCPM4 model paths
    if not config.get('model_path') or args is None:
        # Use path_prefix and quantization parameters to set up MiniCPM4 model paths
        model_path, draft_model_path = setup_minicpm4_paths(
            path_prefix=path_prefix,
            apply_quant=apply_quant,
            apply_eagle=apply_eagle,
            apply_eagle_quant=apply_eagle_quant
        )
        config['model_path'] = model_path
        if draft_model_path:
            config['draft_model_path'] = draft_model_path
            config['frspec_path'] = draft_model_path
    
    # Handle MiniCPM4-specific parameters
    config['minicpm4_yarn'] = minicpm4_yarn
    
    # Apply any additional overrides
    config.update(overrides)
    
    return config 