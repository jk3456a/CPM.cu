#!/usr/bin/env python3
"""
MiniCPM4 Utilities

Essential utilities for MiniCPM4 model configuration and path management.
"""

import sys
import subprocess
from pathlib import Path


def launch_minicpm4_module(module_name):
    """
    Universal MiniCPM4 module launcher
    
    Args:
        module_name: Target module name (e.g., "cpmcu.launch_server", "cpmcu.generate")
    
    Returns:
        int: Process return code
    """
    # Add current directory to sys.path for imports
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Determine parser type based on module name
    is_server = "server" in module_name
    
    # Parse MiniCPM4-specific arguments and get unknown arguments
    parser = create_minicpm4_parser(is_server=is_server)
    # Check if help is requested
    if '--help' in sys.argv or '-h' in sys.argv:
        cmd_args = [sys.executable, "-m", module_name, "--help"]
        subprocess.run(cmd_args)
        parser.print_help()
        return 0
    args, unknown_args = parser.parse_known_args()
    
    # Extract MiniCPM4-specific parameters
    apply_sparse = getattr(args, 'apply_sparse', True)
    apply_quant = getattr(args, 'apply_quant', True)
    apply_eagle = getattr(args, 'apply_eagle', True)
    apply_eagle_quant = getattr(args, 'apply_eagle_quant', True)
    minicpm4_yarn = getattr(args, 'minicpm4_yarn', True)
    
    # Generate model paths based on MiniCPM4 parameters
    model_path, draft_model_path = setup_minicpm4_paths(
        apply_quant=apply_quant,
        apply_eagle=apply_eagle,
        apply_eagle_quant=apply_eagle_quant
    )
    
    # Build command arguments for subprocess
    cmd_args = [sys.executable, "-m", module_name]
    
    # Add model configuration
    cmd_args.extend(["--model-path", model_path])
    if draft_model_path:
        cmd_args.extend(["--draft-model-path", draft_model_path])
        cmd_args.extend(["--frspec-path", draft_model_path])
    
    # Add model type
    cmd_args.extend(["--model-type", "minicpm4" if apply_sparse else "minicpm"])
    
    # Pass through all other arguments (non-MiniCPM4 specific)
    cmd_args.extend(unknown_args)
    
    try:
        # Run the target module with processed arguments
        process = subprocess.run(cmd_args, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        action = "Server failed to start" if "server" in module_name else "Generation failed"
        print(f"{action}: {e}")
        return e.returncode
    except KeyboardInterrupt:
        action = "Server" if "server" in module_name else "Generation"
        print(f"\n{action} interrupted by user")
        return 0


def setup_minicpm4_paths(apply_quant=None, apply_eagle=None, apply_eagle_quant=None):
    """Setup MiniCPM4 model paths based on configuration"""
    # Determine base model path
    model_path = f"openbmb/MiniCPM4-8B-marlin-cpmcu" if apply_quant else f"openbmb/MiniCPM4-8B"
    
    # Determine draft model path
    draft_model_path = None
    if apply_eagle:
        draft_model_path = f"openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu" if apply_eagle_quant else f"openbmb/MiniCPM4-8B-Eagle-FRSpec"
    
    return model_path, draft_model_path


def create_minicpm4_parser(is_server=False):
    """Create argument parser with MiniCPM4-specific parameters only"""
    from cpmcu.args import str2bool
    import argparse
    
    parser = argparse.ArgumentParser(description='MiniCPM4 Configuration')
    
    # Add MiniCPM4-specific arguments
    group = parser.add_argument_group('MiniCPM4 Test Configuration', 'Parameters for MiniCPM4 test')
    
    group.add_argument('--apply-sparse', '--apply_sparse', default=True,
                      type=str2bool, nargs='?', const=True,
                      help='Enable sparse attention (default: True). Values: true/false, yes/no, 1/0, or just --apply-sparse for True')
    
    group.add_argument('--apply-quant', '--apply_quant', default=True,
                      type=str2bool, nargs='?', const=True,
                      help='Enable quantization for base model (default: True). Values: true/false, yes/no, 1/0, or just --apply-quant for True')
    
    group.add_argument('--apply-eagle', '--apply_eagle', default=True,
                      type=str2bool, nargs='?', const=True,
                      help='Enable Eagle speculative decoding (default: True). Values: true/false, yes/no, 1/0, or just --apply-eagle for True')
    
    group.add_argument('--apply-eagle-quant', '--apply_eagle_quant', default=True,
                      type=str2bool, nargs='?', const=True,
                      help='Enable quantization for Eagle draft model (default: True). Values: true/false, yes/no, 1/0, or just --apply-eagle-quant for True')
    
    group.add_argument('--minicpm4-yarn', '--minicpm4_yarn', default=True,
                      type=str2bool, nargs='?', const=True,
                      help='Enable MiniCPM4 YARN for long context support (default: True). Values: true/false, yes/no, 1/0, or just --minicpm4-yarn for True')
    
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