#!/usr/bin/env python3
"""
MiniCPM4 Test Generation Script

Optimized test script for MiniCPM4 models with YARN support and default configurations.
"""

import argparse
import sys
from pathlib import Path

# Smart import handling - supports both execution modes
if __package__ is None:
    # Direct script execution - add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from minicpm4.utils import create_minicpm4_config, create_minicpm4_test_parser, generate_haystack_prompt
    from cpmcu.generate import run_generation
else:
    # Module execution - use relative import
    from .utils import create_minicpm4_config, create_minicpm4_test_parser, generate_haystack_prompt
    from ...cpmcu.generate import run_generation


def main():
    """MiniCPM4 test generation main entry point"""
    parser = create_minicpm4_test_parser()
    args = parser.parse_args()
    
    # Use default prompt if none specified
    if not args.prompt_file and not args.prompt_text:
        args.prompt_text = "Who are you?"
        print("No prompt specified, using default: 'Who are you?'")
    
    # Create MiniCPM4 configuration using the args
    try:
        config = create_minicpm4_config(
            args=args,
            path_prefix=args.path_prefix,
            minicpm4_yarn=args.minicpm4_yarn
        )
    except Exception as e:
        print(f"Error creating MiniCPM4 configuration: {e}")
        return 1
    
    # Add prompt to configuration
    if args.prompt_file:
        config['prompt_file'] = args.prompt_file
    else:
        config['prompt_text'] = args.prompt_text
    
    print("=" * 60)
    print("MiniCPM4 Test Generation Configuration:")
    print("=" * 60)
    print(f"Model: {config['model_path']}")
    print(f"Draft Model: {config.get('draft_model_path', 'None')}")
    print(f"Features: speculative=auto, quant=auto, sparse={config.get('apply_sparse', False)}")
    print(f"YARN: {config.get('minicpm4_yarn', False)}")
    print(f"Generation: num_tokens={config.get('num_generate', 256)}, temperature={config.get('temperature', 0.0)}")
    prompt_text = args.prompt_text or (open(args.prompt_file, 'r').read().strip() if args.prompt_file else "")
    print(f"Prompt: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
    print("=" * 60)
    
    try:
        print("Starting MiniCPM4 generation...")
        
        # Apply MiniCPM4-specific configurations before generation
        if config.get('minicpm4_yarn', False):
            # Import and apply YARN configuration callback
            if __package__ is None:
                from minicpm4.utils import create_minicpm4_yarn_callback
            else:
                from .utils import create_minicpm4_yarn_callback
            config['model_init_callback'] = create_minicpm4_yarn_callback()
        
        # Direct function call instead of subprocess
        generated_text = run_generation(config)
        return 0
    except Exception as e:
        print(f"Generation failed: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        return 0


if __name__ == "__main__":
    sys.exit(main()) 