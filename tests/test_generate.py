import torch
from cpmcu.llm import LLM
from cpmcu.llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from cpmcu.speculative import LLM_with_eagle
from cpmcu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
from transformers import AutoTokenizer
from cpmcu.utils import (
    get_default_config,
    check_or_download_model,
    get_model_paths,
    create_model,
    apply_minicpm4_yarn_config,
    setup_frspec_vocab
)
from cpmcu.args import parse_test_args, display_config_summary
import time
import numpy as np
import argparse
import sys
import os
import json
from huggingface_hub import snapshot_download   



def make_input(tokenizer, args, prompt_content=None):
    """Prepare input tokens from prompt content or file"""
    
    def make_haystack_prompt(digits, target_length_k):
        """Generate haystack prompt with pass key hidden in context"""
        # Simple calculation based on target length
        a = target_length_k * 16  # Scale factor for before text
        b = target_length_k * 33  # Scale factor for after text
        
        head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
        before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * a
        needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
        after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * b
        query = "Now, give me the exact number of the pass key. The pass key is "
        return head + before + needle + after + query
    
    haystack_specified = '--prompt-haystack' in sys.argv or '--prompt_haystack' in sys.argv
    if prompt_content is None:
        # Check command line arguments once
        file_specified = args.prompt_file is not None
        text_specified = args.prompt_text is not None
        
        if haystack_specified and not file_specified and not text_specified:
            # Use haystack prompt
            print(f"Using haystack prompt with {args.prompt_haystack}k tokens (explicitly requested)")
            prompt_content = make_haystack_prompt(681725493, args.prompt_haystack)
        else:
            # Handle file and text combination
            prompt_content = ""
            
            # Load from file if specified
            if file_specified:
                try:
                    with open(args.prompt_file, 'r', encoding='utf-8') as f:
                        prompt_content = f.read().strip()
                    print(f"Loaded prompt from file: {args.prompt_file}")
                except FileNotFoundError:
                    print(f"Warning: {args.prompt_file} not found, skipping file content")
                except Exception as e:
                    print(f"Error reading {args.prompt_file}: {e}, skipping file content")
            
            # Add prompt text (append if file content exists, otherwise use as main content)
            if text_specified:
                if prompt_content:
                    prompt_content += "\n" + args.prompt_text
                    print(f"Appended prompt text to file content")
                else:
                    prompt_content = args.prompt_text
                    print(f"Using prompt text: '{args.prompt_text}'")
            elif not prompt_content:
                # No prompt specified at all, use default
                prompt_content = "Who are you?"
                print(f"No prompt specified, using default: '{prompt_content}'")
    
    if config['test_minicpm4'] and haystack_specified: # TODO: haystack need w/o chat template, may be a bug
        prompt = prompt_content
    else:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_content}], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
    
    print(f"Input token count: {input_ids.shape[1]}")
    
    return input_ids

def print_generation_summary(mode, prefill_stats, decode_stats, config):
    """Print unified generation summary for both modes"""
    print("\n" + "=" * 50)
    print(f"{mode} Generation Summary:")
    print("=" * 50)
    
    # Prefill statistics
    print(f"Prefill length: {prefill_stats['length']}")
    print(f"Prefill time: {prefill_stats['time']:.2f} s")
    print(f"Prefill tokens/s: {prefill_stats['tokens_per_sec']:.2f}")

    # Eagle-specific statistics
    if config['apply_eagle'] and 'mean_accept_length' in decode_stats:
        print(f"Mean accept length: {decode_stats['mean_accept_length']:.2f}")
        # print(f"Decode token/s when acc = 1: {decode_stats['tokens_per_sec'] / decode_stats['mean_accept_length']:.2f}")
    
    # Decode statistics
    print(f"Decode length: {decode_stats['length']}")
    print(f"Decode time: {decode_stats['time']:.2f} s")
    print(f"Decode tokens/s: {decode_stats['tokens_per_sec']:.2f}")

def run_stream_generation(llm, input_ids, config, teminators, tokenizer):
    """Run streaming generation and display results"""
    print("\nGenerated text (streaming output):")
    print("-" * 50)
    
    # Statistics tracking
    prefill_length = input_ids.shape[1]
    prefill_time = 0.0
    total_decode_time = 0.0
    
    generated_text = ""
    total_decode_tokens = 0
    accept_lengths = []
    
    try:
        for result in llm.generate(input_ids, config['num_generate'], teminators=teminators, use_stream=True):
            token = result['token']
            text = result['text']
            is_finished = result['is_finished']
            
            # Track timing statistics
            if 'prefill_time' in result and result['prefill_time'] > 0:
                prefill_time = result['prefill_time']
            if 'decode_time' in result and result['decode_time'] > 0:
                total_decode_time = result['decode_time']
            
            generated_text += text
            total_decode_tokens += 1
            
            # Track accept lengths for eagle models
            if 'accept_length' in result and result['accept_length'] > 0:
                accept_lengths.append(result['accept_length'])
            
            print(text, end='', flush=True)
            
            if is_finished:
                break
                
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
    
    prefill_stats = {
        'length': prefill_length,
        'time': prefill_time,
        'tokens_per_sec': prefill_length / prefill_time if prefill_time > 0 else 0
    }
    
    decode_stats = {
        'length': total_decode_tokens,
        'time': total_decode_time,
        'tokens_per_sec': total_decode_tokens / total_decode_time if total_decode_time > 0 else 0
    }
    
    if config['apply_eagle'] and accept_lengths:
        decode_stats['mean_accept_length'] = np.mean(accept_lengths)
    
    print_generation_summary("Stream", prefill_stats, decode_stats, config)

def run_non_stream_generation(llm, input_ids, config, teminators, tokenizer):
    """Run non-stream generation and display results"""
    prefill_length = input_ids.shape[1]
    
    torch.cuda.synchronize()
    start_time = time.time()
    gen_result = llm.generate(input_ids, config['num_generate'], teminators=teminators, use_stream=False)
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Handle different return formats based on model type
    if config['apply_eagle']:
        # Eagle models return: (tokens, accept_lengths, decode_time, prefill_time)
        tokens, accept_lengths, decode_time, prefill_time = gen_result
        decode_length = len(tokens)
        mean_accept_length = np.mean(accept_lengths)
    else:
        # Base models return: (tokens, decode_time, prefill_time)
        tokens, decode_time, prefill_time = gen_result
        decode_length = len(tokens)
        mean_accept_length = None

    print("\n[Generated Result]")
    print(tokenizer.decode(tokens).strip())
    print("\n")

    prefill_stats = {
        'length': prefill_length,
        'time': prefill_time,
        'tokens_per_sec': prefill_length / prefill_time if prefill_time > 0 else 0
    }
    
    decode_stats = {
        'length': decode_length,
        'time': decode_time,
        'tokens_per_sec': decode_length / decode_time if decode_time > 0 else 0
    }
    
    if mean_accept_length is not None:
        decode_stats['mean_accept_length'] = mean_accept_length
    
    print_generation_summary("Non-Stream", prefill_stats, decode_stats, config)



def main(args, config):
    if not config['test_minicpm4']:
        print(f"test_minicpm4 is False, set apply_sparse to False")
        config['apply_sparse'] = False
    
    display_config_summary(config)
    
    # Get model paths and create model
    eagle_path, base_path, eagle_repo_id, base_repo_id = get_model_paths(args.path_prefix, config)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    llm = create_model(eagle_path, base_path, config)
    
    # Prepare input
    input_ids = make_input(tokenizer, args)
    teminators = [] if not config['use_terminators'] else [tokenizer.eos_token_id]
    
    # Initialize model
    llm.init_storage()
    
    # Apply MiniCPM4 YARN configuration if enabled
    if config['test_minicpm4'] and config['minicpm4_yarn']:
        apply_minicpm4_yarn_config(llm, config)
    
    if config['apply_eagle'] and config['frspec_vocab_size'] > 0:
        setup_frspec_vocab(llm, config, eagle_path, eagle_repo_id)
    llm.load_from_hf()
    
    # Run generation
    if config['use_stream']:
        run_stream_generation(llm, input_ids, config, teminators, tokenizer)
    else:
        run_non_stream_generation(llm, input_ids, config, teminators, tokenizer)
    
    llm.print_perf_summary()

if __name__ == "__main__":
    args, config = parse_test_args()
    main(args, config)
