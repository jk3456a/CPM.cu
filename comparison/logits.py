#!/usr/bin/env python3
"""
Logits capture and generation module
Handles logits capture during model generation
"""

import sys
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from cpmcu.common.logging import logger
from cpmcu.common.utils import (
    setup_model_paths,
    create_model,
    setup_frspec_vocab,
    apply_minicpm4_yarn_config
)
from cpmcu.common.display import Display
from cpmcu.common.args import create_cli_parser
from config import ComparisonConfig


class LogitsCapture:
    """Class to capture logits during generation"""
    
    def __init__(self, config_name: str):
        self.config_name = config_name
        self.captured_logits = []  # List of (step_id, logits_tensor, step_type)
        self.captured_tokens = []  # List of (step_id, token_id)
        self.step_counter = 0
        self.generation_info = []
        self.temperature = 0.0
        
    def save_data(self, filename: str):
        """Save captured data"""
        data = {
            'config_name': self.config_name,
            'captured_logits': self.captured_logits,
            'captured_tokens': self.captured_tokens,
            'total_steps': len(self.captured_logits)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"保存了 {len(self.captured_logits)} 个logits步骤到 {filename}")


def patch_model_for_logits_capture(model, capture: LogitsCapture):
    """Patch the model to capture logits during generation"""
    
    original_generate = model.generate
    
    def patched_generate(*args, **kwargs):
        print(f"[{capture.config_name}] 调用生成函数")
        
        # Store original methods
        original_prefill = model.prefill
        original_decode = model.decode
        
        decode_step_counter = [0]
        stored_logits = []  # Store all decode logits
        prefill_logits = [None]
        
        def patched_prefill(*prefill_args, **prefill_kwargs):
            print(f"[{capture.config_name}] 调用预填充")
            print(f"[{capture.config_name}] 预填充参数[0]长度: {len(prefill_args[0])}")
            logits = original_prefill(*prefill_args, **prefill_kwargs)
            prefill_logits[0] = logits.detach().cpu().numpy() if logits is not None else None
            return logits
        
        def patched_decode(*decode_args, **decode_kwargs):
            decode_step_counter[0] += 1
            print(f"[{capture.config_name}] 解码步骤 {decode_step_counter[0]}")
            logits = original_decode(*decode_args, **decode_kwargs)
            if logits is not None:
                stored_logits.append(logits.detach().cpu().numpy())
            return logits
        
        # Temporarily patch methods
        model.prefill = patched_prefill
        model.decode = patched_decode
        
        try:
            # Call original generate
            results = original_generate(*args, **kwargs)
            
            # Process results to extract accepted tokens and their logits
            if isinstance(results, tuple) and len(results) >= 2:
                tokens, accept_lengths = results[:2]
                
                # Capture prefill token and its logits
                if len(tokens) > 0 and prefill_logits[0] is not None:
                    capture.step_counter += 1
                    step_id = capture.step_counter
                    first_token = tokens[0]
                    
                    # For prefill, take the logits from the last position
                    if len(prefill_logits[0].shape) > 1:
                        # Multi-dimensional logits, take last position
                        token_logits = prefill_logits[0][-1] if prefill_logits[0].shape[0] > 1 else prefill_logits[0][0]
                    else:
                        token_logits = prefill_logits[0]
                    
                    capture.captured_logits.append((step_id, token_logits.copy(), "accepted_prefill"))
                    capture.captured_tokens.append((step_id, first_token))
                    print(f"[{capture.config_name}] 步骤 {step_id} (accepted_prefill): Token {first_token}, Logits预览: {token_logits[:5].tolist()}")
                
                # Process decode steps - only capture accepted tokens
                token_idx = 1  # Start from second token (first is from prefill)
                
                for step_idx, accept_length in enumerate(accept_lengths):
                    if step_idx >= len(stored_logits):
                        break
                    if token_idx >= len(tokens):
                        break
                    
                    step_logits = stored_logits[step_idx]
                    
                    # Only capture the accepted tokens from this decode step
                    for j in range(accept_length):
                        if token_idx + j >= len(tokens):
                            break
                        if j >= len(step_logits):
                            break
                        
                        capture.step_counter += 1
                        step_id = capture.step_counter
                        token = tokens[token_idx + j]
                        token_logits = step_logits[j]  # j-th token's logits in this step
                        
                        capture.captured_logits.append((step_id, token_logits.copy(), f"accepted_decode_{step_idx}_{j}"))
                        capture.captured_tokens.append((step_id, token))
                        print(f"[{capture.config_name}] 步骤 {step_id} (accepted_decode_{step_idx}_{j}): Token {token}, Logits预览: {token_logits[:5].tolist()}")
                    
                    token_idx += accept_length
            
            return results
            
        finally:
            # Restore original methods
            model.prefill = original_prefill
            model.decode = original_decode
    
    # Replace generate method
    model.generate = patched_generate
    
    def restore():
        model.generate = original_generate
    
    return restore


def run_generation_with_config(config: ComparisonConfig) -> Optional[LogitsCapture]:
    """Run generation with specific configuration and capture logits"""
    
    print(f"使用topk_per_iter={config.get_topk_per_iter()} (最大允许: {config.spec_tree_size - 1})")
    
    # Parse arguments using config
    parser = create_cli_parser()
    args = parser.parse_args(config.to_args_list())
    
    print(f"\n{'='*60}")
    print(f"运行 {config.config_name}")
    print(f"spec_num_iter: {config.spec_num_iter}, spec_tree_size: {config.spec_tree_size}")
    print(f"{'='*60}")
    
    # Configure display
    Display.configure(use_plain_mode=getattr(args, 'plain_output', False))
    
    # Setup model paths
    config_dict = vars(args)
    model_path, draft_model_path, frspec_path = setup_model_paths(config_dict)
    
    # Clear GPU memory before model creation
    torch.cuda.empty_cache()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create model with error handling
    try:
        llm = create_model(model_path, draft_model_path, config_dict)
    except Exception as e:
        print(f"创建模型时出错: {e}")
        # Try to free memory
        torch.cuda.empty_cache()
        return None
    
    # Load prompt
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt_content = f.read().strip()
    
    # Apply chat template
    if config.use_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}], 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        prompt = prompt_content
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda", dtype=torch.int32)
    print(f"输入tokens: {input_ids.shape[1]}")
    
    # Initialize model storage
    llm.init_storage()
    
    # Apply MiniCPM4 YARN configuration
    if config.minicpm4_yarn:
        apply_minicpm4_yarn_config(llm)
    
    # Load frequency speculative vocabulary
    has_speculative = draft_model_path is not None
    if has_speculative and (frspec_path is not None) and (getattr(args, 'frspec_vocab_size', 0) > 0):
        setup_frspec_vocab(llm, frspec_path, args.frspec_vocab_size)
    
    # Load model weights
    logger.info("加载模型权重...")
    llm.load_from_hf()
    logger.success("模型加载完成!")
    
    # Create logits capture
    capture = LogitsCapture(config.config_name)
    
    # Patch model for logits capture
    restore_func = patch_model_for_logits_capture(llm, capture)
    
    try:
        # Setup terminators
        terminators = [tokenizer.eos_token_id]
        
        # Run generation
        results = llm.generate(
            input_ids=input_ids.view(-1),
            generation_length=config.comparison_steps,
            teminators=terminators,
            use_stream=False
        )
        
        # Process results
        if isinstance(results, tuple):
            if len(results) >= 4:
                tokens, accept_lengths, decode_time, prefill_time = results[:4]
            elif len(results) >= 3:
                tokens, decode_time, prefill_time = results[:3]
                accept_lengths = []
            else:
                tokens = results[0]
                decode_time = prefill_time = 0.0
                accept_lengths = []
        else:
            tokens = results
            decode_time = prefill_time = 0.0
            accept_lengths = []
        
        # Decode generated text
        if isinstance(tokens, (list, tuple, torch.Tensor)):
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            generated_text = "无法解码tokens"
        
        print(f"\n{config.config_name} 结果:")
        print(f"生成的文本: {generated_text}")
        print(f"预填充时间: {prefill_time:.4f}s")
        print(f"解码时间: {decode_time:.4f}s")
        try:
            token_count = len(tokens)
            print(f"总tokens: {token_count}")
        except TypeError:
            print(f"总tokens: N/A")
        print(f"接受长度: {accept_lengths}")
        print(f"捕获步骤: {len(capture.captured_logits)}")
        
        # Save data
        capture.save_data(f"logits_capture_{config.config_name}.pkl")
        
        return capture
        
    except Exception as e:
        print(f"生成过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return capture
        
    finally:
        restore_func()
        # Free GPU memory after generation
        del llm
        torch.cuda.empty_cache()


def load_logits_data(filename: str) -> Optional[LogitsCapture]:
    """Load captured logits data from file"""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        capture = LogitsCapture(data['config_name'])
        capture.captured_logits = data['captured_logits']
        capture.captured_tokens = data['captured_tokens']
        
        print(f"从 {filename} 加载了 {len(capture.captured_logits)} 个logits步骤")
        return capture
        
    except Exception as e:
        print(f"加载logits数据时出错: {e}")
        return None 