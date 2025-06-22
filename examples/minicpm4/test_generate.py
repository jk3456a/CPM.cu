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
    from minicpm4.default_config import create_minicpm4_config
    from cpmcu.generate import run_generation
else:
    # Module execution - use relative import
    from .default_config import create_minicpm4_config
    from ...cpmcu.generate import run_generation


def create_minicpm4_test_parser():
    """Create argument parser with MiniCPM4 optimized defaults"""
    parser = argparse.ArgumentParser(description='MiniCPM4 Test Generation')
    
    # Prompt configuration
    parser.add_argument('--prompt-file', '--prompt_file', type=str, default=None,
                       help='Path to prompt file')
    parser.add_argument('--prompt-text', '--prompt_text', type=str, default=None,
                       help='Direct prompt text')
    parser.add_argument('--prompt-haystack', '--prompt_haystack', type=int, default=None,
                       help='Generate haystack prompt with specified length in thousands')
    
    # Model configuration with MiniCPM4 defaults
    parser.add_argument('--path-prefix', '--path_prefix', type=str, default='openbmb',
                       help='Model repository prefix (default: openbmb)')
    parser.add_argument('--apply-quant', '--apply_quant', type=lambda x: x.lower() == 'true', default=True,
                       help='Use quantized model (default: True)')
    parser.add_argument('--apply-spec-quant', '--apply_spec_quant', type=lambda x: x.lower() == 'true', default=True,
                       help='Use quantized speculative model (default: True)')
    parser.add_argument('--minicpm4-yarn', '--minicpm4_yarn', type=lambda x: x.lower() == 'true', default=True,
                       help='Enable MiniCPM4 YARN (default: True)')
    parser.add_argument('--apply-sparse', '--apply_sparse', type=lambda x: x.lower() == 'true', default=True,
                       help='Use sparse attention (default: True)')
    
    # Generation parameters
    parser.add_argument('--temperature', '--temp', type=float, default=0.0,
                       help='Generation temperature (default: 0.0)')
    parser.add_argument('--num-generate', '--num_generate', type=int, default=256,
                       help='Number of tokens to generate (default: 256)')
    parser.add_argument('--use-stream', '--use_stream', type=lambda x: x.lower() == 'true', default=True,
                       help='Use stream generation (default: True)')
    parser.add_argument('--chunk-length', '--chunk_length', type=int, default=2048,
                       help='Chunk length (default: 2048)')
    
    # Interactive features
    parser.add_argument('--use-enter', '--use_enter', type=lambda x: x.lower() == 'true', default=False,
                       help='Use enter to generate (default: False)')
    parser.add_argument('--use-decode-enter', '--use_decode_enter', type=lambda x: x.lower() == 'true', default=False,
                       help='Use enter before decode phase (default: False)')
    
    # Optional overrides
    parser.add_argument('--frspec-path', '--frspec_path', type=str, default=None,
                       help='Path to frequency speculative vocabulary file')
    
    return parser


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


def main():
    """MiniCPM4 test generation main entry point"""
    parser = create_minicpm4_test_parser()
    args = parser.parse_args()
    
    # Determine if we should use chat template based on prompt type
    use_chat_template = True
    
    # Create MiniCPM4 configuration
    config = create_minicpm4_config(
        path_prefix=args.path_prefix,
        apply_quant=args.apply_quant,
        apply_spec_quant=args.apply_spec_quant,
        minicpm4_yarn=args.minicpm4_yarn,
        apply_sparse=args.apply_sparse,
        temperature=args.temperature,
        num_generate=args.num_generate,
        use_stream=args.use_stream,
        chunk_length=args.chunk_length,
        use_enter=args.use_enter,
        use_decode_enter=args.use_decode_enter
    )
    
    # Add prompt parameters to config
    if args.prompt_file:
        config['prompt_file'] = args.prompt_file
        # Use chat template for file prompts
        use_chat_template = True
    elif args.prompt_text:
        config['prompt_text'] = args.prompt_text
        # Use chat template for text prompts
        use_chat_template = True
    elif args.prompt_haystack:
        # Generate haystack prompt
        haystack_prompt = generate_haystack_prompt(args.prompt_haystack)
        config['prompt_text'] = haystack_prompt
        # Don't use chat template for haystack prompts
        use_chat_template = False
        print(f"Generated haystack prompt with {args.prompt_haystack}k tokens (using pass key 681725493)")
        print("Note: Chat template disabled for haystack prompt")
    
    # Set chat template usage in config
    config['use_chat_template'] = use_chat_template
    
    # Add optional FRSpec path
    if args.frspec_path:
        config['frspec_path'] = args.frspec_path
    
    print("=" * 60)
    print("MiniCPM4 Test Generation Configuration:")
    print("=" * 60)
    print(f"Model: {config['model_path']}")
    print(f"Draft Model: {config['draft_model_path']}")
    print(f"Features: speculative={config['apply_speculative']}, quant={config['apply_quant']}, sparse={config['apply_sparse']}")
    print(f"YARN: {config['minicpm4_yarn']}")
    print(f"Generation: num_tokens={config['num_generate']}, stream={config['use_stream']}")
    print(f"Chat template: {'enabled' if use_chat_template else 'disabled'}")
    print("=" * 60)
    
    try:
        print("Starting MiniCPM4 generation...")
        
        # Apply MiniCPM4-specific configurations before generation
        if config.get('minicpm4_yarn', False):
            # Import and apply YARN configuration callback
            if __package__ is None:
                from minicpm4.default_config import create_minicpm4_yarn_callback
            else:
                from .default_config import create_minicpm4_yarn_callback
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