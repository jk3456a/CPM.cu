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

from .common.log_utils import logger, Console
from rich.panel import Panel
from rich.table import Table

from .common.utils import (
    setup_model_paths,
    create_model,
    setup_frspec_vocab,
    apply_minicpm4_yarn_config
)
from .common.args import parse_test_args
from .common.display import (
    print_config_summary, 
    TextStreamer,
    display_text
)
from rich.panel import Panel


def print_generation_stats(stats, has_speculative=False):
    """Print generation statistics summary using enhanced format."""
    from .common.display import print_performance_summary
    
    # Convert stats to the expected format
    formatted_stats = {}
    
    if stats.get('input_length') is not None:
        formatted_stats['prefill_length'] = stats['input_length']
    
    if stats.get('prefill_time') is not None:
        formatted_stats['prefill_time'] = stats['prefill_time']
    
    if stats.get('decode_length') is not None:
        formatted_stats['decode_length'] = stats['decode_length']
    
    if stats.get('decode_time') is not None:
        formatted_stats['decode_time'] = stats['decode_time']
    
    if has_speculative and stats.get('accept_lengths'):
        formatted_stats['accept_lengths'] = stats['accept_lengths']
    
    print_performance_summary(formatted_stats)


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
            logger.warning(f"Failed to apply chat template: {e}, using raw prompt")
            prompt = prompt_content
    else:
        prompt = prompt_content
        logger.info("Using raw prompt (chat template disabled)")
    
    # Show prompt with special characters escaped for better readability in logs
    escaped_prompt = repr(prompt[:100])[1:-1]  # Remove outer quotes from repr()
    logger.info(f"Input prompt: [dim]{escaped_prompt}{'...' if len(prompt) > 100 else ''}[/dim]")
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input_ids.to("cuda", dtype=torch.int32)


def run_stream_generation(llm, input_ids, config, terminators, tokenizer):
    """Run streaming generation with enhanced display"""
    logger.info("Starting streaming generation...")

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
        
        # Use enhanced streaming display
        with TextStreamer("Generated Response") as stream_display:
            # Process streaming results and collect statistics
            for result in results:
                if isinstance(result, dict):
                    if 'text' in result:
                        text = result['text']
                        stream_display.update(text)
                        generated_text += text
                    
                    # Update statistics from each result
                    if 'prefill_time' in result and result['prefill_time'] > 0:
                        stats['prefill_time'] = result['prefill_time']
                    if 'decode_time' in result and result['decode_time'] > 0:
                        stats['decode_time'] = result['decode_time']
                    if 'accept_length' in result and result['accept_length'] > 0:
                        stats['accept_lengths'].append(result['accept_length'])
                        
                elif isinstance(result, str):
                    stream_display.update(result)
                    generated_text += result
        
        logger.success("Streaming generation completed!")
        
        # Set decode length and print statistics
        decode_length = len(tokenizer.encode(generated_text, add_special_tokens=False))
        stats['decode_length'] = decode_length
        print_generation_stats(stats, has_speculative)
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error during streaming generation: {e}")
        raise RuntimeError(f"Error during streaming generation: {e}")


def run_non_stream_generation(llm, input_ids, config, terminators, tokenizer):
    """Run non-streaming generation with enhanced display"""
    logger.info("Starting non-streaming generation...")

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
        
        # Decode tokens and handle edge cases
        generated_text = tokenizer.decode(tokens, skip_special_tokens=True) or ""
        
        # Create and display panel
        display_text(generated_text, title="Generated Response")

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
        logger.error(f"Error during non-streaming generation: {e}")
        raise RuntimeError(f"Error during non-streaming generation: {e}")


def run_generation(args):
    """Core generation function that can be called by various frontends"""
    
    # Display complete configuration summary
    print_config_summary(args, "Configuration")
    
    # Validate required parameters
    if not getattr(args, 'model_path', None):
        raise ValueError("model_path is required")
    
    # Setup model paths - convert args to dict for compatibility with existing functions
    config = vars(args)
    
    try:
        model_path, draft_model_path, frspec_path = setup_model_paths(config)
    except Exception as e:
        raise RuntimeError(f"Error setting up model paths: {e}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logger.info(f"Loaded tokenizer from: [cyan]{model_path}[/cyan]")
    except Exception as e:
        raise RuntimeError(f"Error loading tokenizer: {e}")
    
    # Create model
    try:
        llm = create_model(model_path, draft_model_path, config)
        logger.info(f"Created model: [yellow]{type(llm).__name__}[/yellow]")
    except Exception as e:
        raise RuntimeError(f"Error creating model: {e}")
    
    # Prepare input
    try:
        input_ids = make_input(tokenizer, args)
        logger.info(f"Input tokens: {input_ids.shape[1]}")
    except Exception as e:
        raise RuntimeError(f"Error preparing input: {e}")
    
    # Setup terminators
    terminators = []
    if not getattr(args, 'ignore_eos', False):
        terminators.append(tokenizer.eos_token_id)
    
    # Initialize model storage
    logger.info("Initializing model storage...")
    llm.init_storage()
    
    # Apply MiniCPM4 YARN configuration if enabled
    if getattr(args, 'minicpm4_yarn', False):
        try:
            apply_minicpm4_yarn_config(llm)
        except Exception as e:
            logger.warning(f"MiniCPM4 YARN configuration failed: {e}")
    
    # Load frequency speculative vocabulary if enabled (draft model exists)
    has_speculative = getattr(args, 'draft_model_path', None) is not None
    if has_speculative and (frspec_path is not None) and (getattr(args, 'frspec_vocab_size', 0) > 0):
        frspec_result = setup_frspec_vocab(llm, frspec_path, getattr(args, 'frspec_vocab_size', 0))
        if frspec_result is True:
            logger.success("Loaded frequency speculative vocabulary")
        else:
            logger.warning("Could not load frequency speculative vocabulary")
    
    # Load model weights
    logger.info("Loading model weights...")
    llm.load_from_hf()
    logger.success("Model loading completed!")
    
    # Run generation - use config dict for compatibility with existing functions
    try:
        if getattr(args, 'use_stream', True):
            generated_text = run_stream_generation(llm, input_ids, config, terminators, tokenizer)
        else:
            generated_text = run_non_stream_generation(llm, input_ids, config, terminators, tokenizer)
            
        return generated_text
        
    except Exception as e:
        raise RuntimeError(f"Error during generation: {e}")


def main():
    """Entry point for the command-line interface"""
    try:
        args = parse_test_args()
        run_generation(args)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nGeneration interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main() 