#!/usr/bin/env python3
"""
MiniCPM4 Test Generation Script

This script provides MiniCPM4-specific test generation with YARN configuration
and optimized settings for MiniCPM4 models.
"""

import sys
import os
import argparse
import torch

# Import from installed cpmcu package
from cpmcu.utils import setup_model_paths, create_model, setup_frspec_vocab
from cpmcu.args import display_config_summary
from transformers import AutoTokenizer

# Import local config module
from config import (
    get_minicpm4_default_config,
    get_minicpm4_model_paths,
    apply_minicpm4_yarn_config
)


def make_input(tokenizer, args):
    """Create input for generation"""
    if args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    elif args.prompt_text:
        prompt = args.prompt_text
    elif hasattr(args, 'prompt_haystack') and args.prompt_haystack:
        # Generate haystack prompt
        needle = "The secret key is 42"
        haystack_length = args.prompt_haystack * 1000
        haystack = "Random text. " * (haystack_length // 13)  # Approximate
        prompt = f"{haystack}\n{needle}\nQuestion: What is the secret key?"
    else:
        prompt = "Hello, how are you today?"
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input_ids.to("cuda", dtype=torch.int32)

def run_stream_generation(llm, input_ids, config, terminators, tokenizer):
    """Run streaming generation"""
    print("Starting streaming generation...")
    results = llm.generate(
        input_ids=input_ids,
        generation_length=config['num_generate'],
        teminators=terminators,
        use_stream=True
    )
    
    generated_text = ""
    for result in results:
        if 'text' in result:
            print(result['text'], end='', flush=True)
            generated_text += result['text']
    print("\nGeneration completed!")
    return generated_text

def run_non_stream_generation(llm, input_ids, config, terminators, tokenizer):
    """Run non-streaming generation"""
    print("Starting non-streaming generation...")
    results = llm.generate(
        input_ids=input_ids,
        generation_length=config['num_generate'],
        teminators=terminators,
        use_stream=False
    )
    
    generated_text = ""
    for result in results:
        if 'text' in result:
            generated_text += result['text']
    
    print(f"Generated text: {generated_text}")
    return generated_text




def create_minicpm4_test_parser():
    """Create MiniCPM4-specific test argument parser"""
    parser = argparse.ArgumentParser(description='MiniCPM4 Test Generation')
    
    # MiniCPM4 specific parameters
    parser.add_argument('--path-prefix', type=str, default='openbmb',
                       help='HuggingFace model path prefix (default: openbmb)')
    parser.add_argument('--enable-yarn', action='store_true', default=True,
                       help='Enable YARN rope scaling for long context (default: True)')
    parser.add_argument('--use-quant', action='store_true', default=True,
                       help='Use quantized models (default: True)')
    parser.add_argument('--use-eagle', action='store_true', default=True,
                       help='Use Eagle speculative decoding (default: True)')
    parser.add_argument('--use-eagle-quant', action='store_true', default=True,
                       help='Use quantized Eagle model (default: True)')
    
    # Test parameters
    parser.add_argument('--prompt-file', type=str, default=None,
                       help='Path to prompt file')
    parser.add_argument('--prompt-text', type=str, default=None,
                       help='Direct prompt text')
    parser.add_argument('--prompt-haystack', type=int,
                       help='Generate haystack prompt with specified length in thousands')
    parser.add_argument('--use-stream', action='store_true', default=True,
                       help='Use stream generation (default: True)')
    parser.add_argument('--num-generate', type=int, default=256,
                       help='Number of tokens to generate (default: 256)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for generation (default: 0.0)')
    
    return parser


def setup_minicpm4_test(args):
    """Setup MiniCPM4 test configuration"""
    # Get MiniCPM4 model paths
    model_paths = get_minicpm4_model_paths(
        path_prefix=args.path_prefix,
        use_quant=args.use_quant,
        use_eagle=args.use_eagle,
        use_eagle_quant=args.use_eagle_quant
    )
    
    # Get MiniCPM4 default configuration
    config = get_minicpm4_default_config(
        enable_yarn=args.enable_yarn,
        temperature=args.temperature,
        num_generate=args.num_generate,
        use_stream=args.use_stream
    )
    
    # Update with model paths
    config.update(model_paths)
    
    return config


def main():
    """Main entry point for MiniCPM4 test generation"""
    parser = create_minicpm4_test_parser()
    args = parser.parse_args()
    
    # Setup MiniCPM4 configuration
    config = setup_minicpm4_test(args)
    
    print("=" * 50)
    print("MiniCPM4 Test Configuration:")
    print("=" * 50)
    print(f"Base Model: {config.get('model_path', 'N/A')}")
    print(f"Draft Model: {config.get('draft_model_path', 'N/A')}")
    print(f"Quantization: Base={config['apply_quant']}, Eagle={config['apply_spec_quant']}")
    print(f"YARN Enabled: {config.get('enable_yarn', False)}")
    print(f"Generation: Stream={config['use_stream']}, Tokens={config['num_generate']}")
    print("=" * 50)
    
    # Setup model paths and create model
    model_path, draft_model_path, frspec_path = setup_model_paths(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = create_model(model_path, draft_model_path, config)
    
    # Prepare input
    input_ids = make_input(tokenizer, args)
    teminators = [] if not config['use_terminators'] else [tokenizer.eos_token_id]
    
    # Initialize model
    llm.init_storage()
    
    # Apply MiniCPM4 YARN configuration if enabled
    if config.get('enable_yarn', True):
        print("Applying MiniCPM4 YARN rope_scaling parameters")
        apply_minicpm4_yarn_config(llm)
    
    # Load frequency speculative vocabulary if enabled
    if config.get('apply_speculative', False) and frspec_path:
        print(f"Loading frequency vocabulary from {frspec_path}")
        setup_frspec_vocab(llm, frspec_path)
    
    # Load model weights
    llm.load_from_hf()
    
    # Run generation
    if config['use_stream']:
        run_stream_generation(llm, input_ids, config, teminators, tokenizer)
    else:
        run_non_stream_generation(llm, input_ids, config, teminators, tokenizer)
    
    llm.print_perf_summary()


if __name__ == "__main__":
    main() 