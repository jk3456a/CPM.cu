#!/usr/bin/env python3
"""
MiniCPM4 Utilities

Essential utilities for MiniCPM4 model configuration and path management.
"""


def setup_minicpm4_paths(apply_quant=None, apply_eagle=None, apply_eagle_quant=None):
    """Setup MiniCPM4 model paths based on configuration"""
    # Determine base model path
    model_path = f"openbmb/MiniCPM4-8B-marlin-cpmcu" if apply_quant else f"openbmb/MiniCPM4-8B"
    
    # Determine draft model path
    draft_model_path = None
    if apply_eagle:
        draft_model_path = f"openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu" if apply_eagle_quant else f"openbmb/MiniCPM4-8B-Eagle-FRSpec"
    
    return model_path, draft_model_path


def create_minicpm4_parser():
    """Create argument parser with MiniCPM4-specific parameters extended from test parser"""
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from cpmcu.args import create_test_parser, str2bool
    
    # Create base test parser with all standard arguments
    parser = create_test_parser()
    parser.description = 'MiniCPM4 Optimized Tool'
    
    # Override the required model-path argument to make it optional for MiniCPM4
    # since we can automatically determine paths based on configuration
    for action in parser._actions:
        if action.dest == 'model_path' or action.dest == 'draft_model_path' or action.dest == 'frspec_path':
            action.required = False
            action.default = None
            action.help = 'auto-selected based on configuration'
            break
    
    # Add MiniCPM4-specific arguments (reuse existing apply-quant and apply-spec-quant)
    group = parser.add_argument_group('MiniCPM4 Configuration', 'MiniCPM4-specific parameters')
    
    group.add_argument('--apply-quant', '--apply_quant', default=True,
                      type=str2bool, nargs='?', const=True,
                      help='Enable quantization for base model')
    
    group.add_argument('--apply-eagle', '--apply_eagle', default=True,
                      type=str2bool, nargs='?', const=True,
                      help='Enable Eagle speculative decoding')
    
    group.add_argument('--apply-eagle-quant', '--apply_eagle_quant', default=True,
                      type=str2bool, nargs='?', const=True,
                      help='Enable quantization for Eagle draft model')
    
    group.add_argument('--minicpm4-yarn', '--minicpm4_yarn', default=True,
                      type=str2bool, nargs='?', const=True,
                      help='Enable MiniCPM4 YARN for long context support')
    
    return parser


# === YARN Configuration (for core module usage) ===

def apply_minicpm4_yarn_config(llm):
    """Apply MiniCPM4 YARN configuration to model"""
    yarn_factors = [
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
    
    if not hasattr(llm.config, 'rope_scaling') or llm.config.rope_scaling is None:
        llm.config.rope_scaling = {}
    
    llm.config.rope_scaling['rope_type'] = 'longrope'
    llm.config.rope_scaling['long_factor'] = yarn_factors
    llm.config.rope_scaling['short_factor'] = yarn_factors
    print("Applied MiniCPM4 YARN rope_scaling parameters")


def create_minicpm4_yarn_callback():
    """Create a callback function for applying MiniCPM4 YARN configuration"""
    return apply_minicpm4_yarn_config 