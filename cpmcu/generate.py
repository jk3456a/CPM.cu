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
    ModelFactory,
    setup_frspec_vocab,
    apply_minicpm4_yarn_config
)
from .args import parse_test_args, display_config_summary


def print_generation_stats(stats, has_speculative=False):
    """Print generation statistics summary"""
    title = "Generation Summary"
    separator = "=" * 50
    
    print(f"\n{title}")
    print(separator)
    
    # Prefill information
    if stats.get('input_length') is not None:
        print(f"Prefill length: {stats['input_length']}")
        if stats.get('prefill_time') is not None and stats['prefill_time'] > 0:
            print(f"Prefill time: {stats['prefill_time']:.2f} s")
            print(f"Prefill tokens/s: {stats['input_length'] / stats['prefill_time']:.2f}")
    
    # Speculative decoding statistics
    if has_speculative and stats.get('accept_lengths'):
        mean_accept_length = sum(stats['accept_lengths']) / len(stats['accept_lengths'])
        print(f"Mean accept length: {mean_accept_length:.2f}")
    
    # Decode information
    if stats.get('decode_length'):
        print(f"Decode length: {stats['decode_length']}")
        
        if stats.get('decode_time') is not None and stats['decode_time'] > 0:
            print(f"Decode time: {stats['decode_time']:.2f} s")
            print(f"Decode tokens/s: {stats['decode_length'] / stats['decode_time']:.2f}")


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
    
    try:
        results = llm.generate(
            input_ids=input_ids.view(-1),
            generation_length=config['num_generate'],
            teminators=terminators,
            use_stream=True
        )
        
        generated_text = ""
        stats = {'input_length': len(input_ids.view(-1)), 'accept_lengths': []}
        has_speculative = config.get('draft_model_path') is not None
        
        # Process streaming results and collect statistics
        for result in results:
            if isinstance(result, dict):
                if 'text' in result:
                    text = result['text']
                    print(text, end='', flush=True)
                    generated_text += text
                
                # Update statistics from each result
                if 'prefill_time' in result and result['prefill_time'] > 0:
                    stats['prefill_time'] = result['prefill_time']
                if 'decode_time' in result and result['decode_time'] > 0:
                    stats['decode_time'] = result['decode_time']
                if 'accept_length' in result and result['accept_length'] > 0:
                    stats['accept_lengths'].append(result['accept_length'])
                    
            elif isinstance(result, str):
                print(result, end='', flush=True)
                generated_text += result
        
        print("\nStreaming generation completed!")
        
        # Set decode length and print statistics
        decode_length = len(tokenizer.encode(generated_text, add_special_tokens=False))
        stats['decode_length'] = decode_length
        print_generation_stats(stats, has_speculative)
        
        return generated_text
        
    except Exception as e:
        print(f"Error during streaming generation: {e}")
        raise RuntimeError(f"Error during streaming generation: {e}")


def run_non_stream_generation(llm, input_ids, config, terminators, tokenizer):
    """Run non-streaming generation"""
    print("Starting non-streaming generation...")
    
    try:
        results = llm.generate(
            input_ids=input_ids.view(-1),
            generation_length=config['num_generate'],
            teminators=terminators,
            use_stream=False
        )
        
        # Extract tokens and statistics from results
        input_length = len(input_ids.view(-1))
        has_speculative = config.get('draft_model_path') is not None
        
        if has_speculative:
            tokens, accept_lengths, decode_time, prefill_time = results
        else:
            tokens, decode_time, prefill_time = results
            accept_lengths = None
        
        generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Print generated text and statistics
        print(f"\n[Generated text]\n{generated_text}")
        
        # Create and populate statistics
        stats = {
            'input_length': input_length,
            'decode_length': len(tokens),
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'accept_lengths': accept_lengths if accept_lengths else []
        }
        
        print_generation_stats(stats, has_speculative)
        
        return generated_text
        
    except Exception as e:
        print(f"Error during non-streaming generation: {e}")
        raise RuntimeError(f"Error during non-streaming generation: {e}")


def run_generation(config):
    """Core generation function that can be called by various frontends"""
    # Display configuration summary at the start
    display_config_summary(config, "Generation Configuration")
    
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
        llm = ModelFactory.create_model(model_path, draft_model_path, config)
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
    if not config.get('ignore_eos', False):
        terminators.append(tokenizer.eos_token_id)
    
    # Initialize model storage
    print("Initializing model storage...")
    llm.init_storage()
    
    # Apply MiniCPM4 YARN configuration if enabled
    if config.get('minicpm4_yarn', False):
        try:
            apply_minicpm4_yarn_config(llm)
        except Exception as e:
            print(f"Warning: MiniCPM4 YARN configuration failed: {e}")
    
    # Load frequency speculative vocabulary if enabled (draft model exists)
    has_speculative = config.get('draft_model_path') is not None
    if has_speculative and (frspec_path is not None) and (config.get('frspec_vocab_size', 0) > 0):
        frspec_result = setup_frspec_vocab(llm, frspec_path, config.get('frspec_vocab_size', 0))
        if frspec_result is True:
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
            
        return generated_text
        
    except Exception as e:
        raise RuntimeError(f"Error during generation: {e}")


def main():
    """Main entry point for generation module"""
    # Parse arguments using unified parser
    args, config = parse_test_args()
    
    try:
        # Run generation (config summary will be displayed inside)
        generated_text = run_generation(config)
        return 0
    except Exception as e:
        print(f"Generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 