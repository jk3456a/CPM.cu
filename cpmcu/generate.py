#!/usr/bin/env python3
"""
CPM.cu Core Generation Module

Core generation functionality for CPM.cu models.
This module contains the main generation logic used by various frontends.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

from .utils import (
    setup_model_paths,
    create_model,
    setup_frspec_vocab
)
from .args import parse_test_args, display_config_summary


def make_input(tokenizer, args):
    """Create input for generation based on arguments"""
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
    elif args.prompt_text:
        prompt_content = args.prompt_text
    else:
        prompt_content = "Who are you"
    
    # Apply chat template if enabled (default: True)
    if getattr(args, 'use_chat_template', True):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_content}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            print("Applied chat template")
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}, using raw prompt")
            prompt = prompt_content
    else:
        prompt = prompt_content
        print("Using raw prompt (chat template disabled)")
    
    print(f"Input prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input_ids.to("cuda", dtype=torch.int32)


def run_stream_generation(llm, input_ids, config, terminators, tokenizer):
    """Run streaming generation"""
    print("Starting streaming generation...")
    
    results = llm.generate(
        input_ids=input_ids.view(-1),
        generation_length=config['num_generate'],
        teminators=terminators,
        use_stream=True
    )
    
    generated_text = ""
    for result in results:
        if isinstance(result, dict) and 'text' in result:
            text = result['text']
            print(text, end='', flush=True)
            generated_text += text
        elif isinstance(result, str):
            print(result, end='', flush=True)  
            generated_text += result
    
    print("\nStreaming generation completed!")
    return generated_text


def run_non_stream_generation(llm, input_ids, config, terminators, tokenizer):
    """Run non-streaming generation"""
    print("Starting non-streaming generation...")
    
    results = llm.generate(
        input_ids=input_ids.view(-1),
        generation_length=config['num_generate'],
        teminators=terminators,
        use_stream=False
    )
    
    # Handle different return formats based on model type
    if config.get('apply_speculative', False):
        # Speculative models return: (tokens, accept_lengths, decode_time, prefill_time)
        tokens, accept_lengths, decode_time, prefill_time = results
    else:
        # Base models return: (tokens, decode_time, prefill_time)
        tokens, decode_time, prefill_time = results
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    
    print(f"Generated text: {generated_text}")
    print(f"Decode time: {decode_time:.2f}s, Prefill time: {prefill_time:.2f}s")
    
    return generated_text


def run_generation(config):
    """Core generation function that can be called by various frontends"""
    # Create a dummy args object for compatibility with make_input
    class Args:
        def __init__(self, config):
            self.prompt_file = config.get('prompt_file')
            self.prompt_text = config.get('prompt_text')
            self.use_chat_template = config.get('use_chat_template', True)
    
    args = Args(config)
    
    # Validate required parameters
    if not config.get('model_path'):
        raise ValueError("model_path is required")
    
    # Setup model paths
    try:
        model_path, draft_model_path, frspec_path = setup_model_paths(config)
    except Exception as e:
        raise RuntimeError(f"Error setting up model paths: {e}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loaded tokenizer from: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading tokenizer: {e}")
    
    # Create model
    try:
        llm = create_model(model_path, draft_model_path, config)
        print(f"Created model: {type(llm).__name__}")
    except Exception as e:
        raise RuntimeError(f"Error creating model: {e}")
    
    # Prepare input
    try:
        input_ids = make_input(tokenizer, args)
        print(f"Input shape: {input_ids.shape}")
    except Exception as e:
        raise RuntimeError(f"Error preparing input: {e}")
    
    # Setup terminators
    terminators = []
    if config.get('use_terminators', True):
        terminators.append(tokenizer.eos_token_id)
    
    # Initialize model storage
    print("Initializing model storage...")
    llm.init_storage()
    
    # Load frequency speculative vocabulary if enabled
    if config.get('apply_speculative', False) and frspec_path:
        if setup_frspec_vocab(llm, frspec_path):
            print("Loaded frequency speculative vocabulary")
        else:
            print("Warning: Could not load frequency speculative vocabulary")
    
    # Load model weights
    print("Loading model weights...")
    llm.load_from_hf()
    print("Model loading completed!")
    
    # Run generation
    try:
        if config.get('use_stream', True):
            generated_text = run_stream_generation(llm, input_ids, config, terminators, tokenizer)
        else:
            generated_text = run_non_stream_generation(llm, input_ids, config, terminators, tokenizer)
        
        # Print performance summary if available
        if hasattr(llm, 'print_perf_summary'):
            llm.print_perf_summary()
            
        return generated_text
        
    except Exception as e:
        raise RuntimeError(f"Error during generation: {e}")


def main():
    """Main entry point for generation module"""
    # Parse arguments using unified parser
    args, config = parse_test_args()
    
    # Display configuration summary
    display_config_summary(config, "Generation Configuration")
    
    try:
        # Run generation
        generated_text = run_generation(config)
        print("\nGeneration completed successfully!")
        return 0
    except Exception as e:
        print(f"Generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 