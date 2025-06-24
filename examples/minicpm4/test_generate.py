#!/usr/bin/env python3
"""
MiniCPM4 Test Generation Launcher

MiniCPM4-optimized test generation launcher that parses MiniCPM4-specific parameters 
and forwards all other arguments to the core generation module.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Launch the core generation module with MiniCPM4 parameter processing"""
    # Add parent directory to sys.path for absolute imports
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    from utils import create_minicpm4_parser, setup_minicpm4_paths
    
    # Parse MiniCPM4-specific arguments and get unknown arguments
    parser = create_minicpm4_parser()
    args, unknown_args = parser.parse_known_args()
    
    # Extract MiniCPM4-specific parameters (map to existing test parser args)
    apply_quant = getattr(args, 'apply_quant', True)  # Use existing apply-quant
    apply_eagle = getattr(args, 'apply_eagle', True)  # MiniCPM4 specific
    apply_eagle_quant = getattr(args, 'apply_eagle_quant', True)  # Map to existing apply-spec-quant  
    minicpm4_yarn = getattr(args, 'minicpm4_yarn', True)
    
    # Generate model paths based on MiniCPM4 parameters
    model_path, draft_model_path = setup_minicpm4_paths(
        apply_quant=apply_quant,
        apply_eagle=apply_eagle,
        apply_eagle_quant=apply_eagle_quant
    )
    
    # Build command arguments for subprocess
    cmd_args = [sys.executable, "-m", "cpmcu.generate"]
    
    # Add model configuration
    cmd_args.extend(["--model-path", model_path])
    if draft_model_path:
        cmd_args.extend(["--draft-model-path", draft_model_path])
        cmd_args.extend(["--frspec-path", draft_model_path])
    
    # Add model type
    cmd_args.extend(["--model-type", "minicpm4"])
    
    # Add default prompt if none provided in unknown args
    has_prompt = any(arg in unknown_args for arg in ["--prompt-text", "--prompt-file"])
    if not has_prompt:
        cmd_args.extend(["--prompt-text", "Who are you?"])
        print("No prompt specified, using default: 'Who are you?'")
    
    # Pass through all other arguments
    cmd_args.extend(unknown_args)
    
    try:
        # Run the core generation module with processed arguments
        process = subprocess.run(cmd_args, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Generation failed: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        return 0


if __name__ == "__main__":
    sys.exit(main()) 