#!/usr/bin/env python3
"""
CPM.cu Unified Argument Processing

Unified argument processing module to eliminate duplicate argument definitions
"""

import argparse
import sys
import torch
from typing import Dict, Any, Tuple
from .utils import get_default_config


def add_model_config_args(parser: argparse.ArgumentParser):
    """Add common model configuration arguments"""
    
    # Basic parameters
    parser.add_argument('--path-prefix', '--path_prefix', '-p', type=str, default='openbmb', 
                       help='Path prefix for model directories')

    # Model configuration boolean parameters
    parser.add_argument('--test-minicpm4', '--test_minicpm4', action='store_true', default=argparse.SUPPRESS,
                       help='Use MiniCPM4 model')
    parser.add_argument('--no-test-minicpm4', '--no_test_minicpm4', action='store_false', dest='test_minicpm4', default=argparse.SUPPRESS)
    
    parser.add_argument('--apply-eagle', '--apply_eagle', action='store_true', default=argparse.SUPPRESS,
                       help='Use Eagle speculative decoding')
    parser.add_argument('--no-apply-eagle', '--no_apply_eagle', action='store_false', dest='apply_eagle', default=argparse.SUPPRESS)
    
    parser.add_argument('--apply-quant', '--apply_quant', action='store_true', default=argparse.SUPPRESS,
                       help='Use quantized model')
    parser.add_argument('--no-apply-quant', '--no_apply_quant', action='store_false', dest='apply_quant', default=argparse.SUPPRESS)
    
    parser.add_argument('--apply-sparse', '--apply_sparse', action='store_true', default=argparse.SUPPRESS,
                       help='Use sparse attention')
    parser.add_argument('--no-apply-sparse', '--no_apply_sparse', action='store_false', dest='apply_sparse', default=argparse.SUPPRESS)
    
    parser.add_argument('--apply-eagle-quant', '--apply_eagle_quant', action='store_true', default=argparse.SUPPRESS,
                       help='Use quantized Eagle model')
    parser.add_argument('--no-apply-eagle-quant', '--no_apply_eagle_quant', action='store_false', dest='apply_eagle_quant', default=argparse.SUPPRESS)
    
    parser.add_argument('--apply-compress-lse', '--apply_compress_lse', action='store_true', default=argparse.SUPPRESS,
                       help='Apply LSE compression')
    parser.add_argument('--no-apply-compress-lse', '--no_apply_compress_lse', action='store_false', dest='apply_compress_lse', default=argparse.SUPPRESS)
    
    parser.add_argument('--cuda-graph', '--cuda_graph', action='store_true', default=argparse.SUPPRESS,
                       help='Use CUDA graph optimization')
    parser.add_argument('--no-cuda-graph', '--no_cuda_graph', action='store_false', dest='cuda_graph', default=argparse.SUPPRESS)
    
    parser.add_argument('--use-teminators', '--use_terminators', action='store_true', default=argparse.SUPPRESS,
                       help='Use terminators')
    parser.add_argument('--no-use-teminators', '--no_use_terminators', action='store_false', dest='use_terminators', default=argparse.SUPPRESS)
    
    parser.add_argument('--minicpm4-yarn', '--minicpm4_yarn', action='store_true', default=argparse.SUPPRESS,
                       help='Use MiniCPM4 YARN for long context')
    parser.add_argument('--no-minicpm4-yarn', '--no_minicpm4_yarn', action='store_false', dest='minicpm4_yarn', default=argparse.SUPPRESS)
    
    parser.add_argument('--use-enter', '--use_enter', action='store_true', default=argparse.SUPPRESS,
                       help='Use enter to generate')
    parser.add_argument('--no-use-enter', '--no_use_enter', action='store_false', dest='use_enter', default=argparse.SUPPRESS)
    
    parser.add_argument('--use-decode-enter', '--use_decode_enter', action='store_true', default=argparse.SUPPRESS,
                       help='Use enter before decode phase')
    parser.add_argument('--no-use-decode-enter', '--no_use_decode_enter', action='store_false', dest='use_decode_enter', default=argparse.SUPPRESS)

    # Numerical parameters
    parser.add_argument('--frspec-vocab-size', '--frspec_vocab_size', type=int, default=None,
                       help='Frequent speculation vocab size (default: 32768)')
    parser.add_argument('--eagle-window-size', '--eagle_window_size', type=int, default=None,
                       help='Eagle window size (default: 1024)')
    parser.add_argument('--eagle-num-iter', '--eagle_num_iter', type=int, default=None,
                       help='Eagle number of iterations (default: 2)')
    parser.add_argument('--eagle-topk-per-iter', '--eagle_topk_per_iter', type=int, default=None,
                       help='Eagle top-k per iteration (default: 10)')
    parser.add_argument('--eagle-tree-size', '--eagle_tree_size', type=int, default=None,
                       help='Eagle tree size (default: 12)')
    parser.add_argument('--sink-window-size', '--sink_window_size', type=int, default=None,
                       help='Sink window size (default: 1)')
    parser.add_argument('--block-window-size', '--block_window_size', type=int, default=None,
                       help='Block window size (default: 8)')
    parser.add_argument('--sparse-topk-k', '--sparse_topk_k', type=int, default=None,
                       help='Sparse attention top-k (default: 64)')
    parser.add_argument('--sparse-switch', '--sparse_switch', type=int, default=None,
                       help='Sparse switch threshold (default: 1)')
    parser.add_argument('--chunk-length', '--chunk_length', type=int, default=None,
                       help='Chunk length (default: 2048)')
    parser.add_argument('--memory-limit', '--memory_limit', type=float, default=None,
                       help='Memory limit (default: 0.9)')
    parser.add_argument('--temperature', '--temperature', type=float, default=None,
                       help='Temperature (default: 0.0)')
    parser.add_argument('--random-seed', '--random_seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--dtype', type=str, default=None, choices=['float16', 'bfloat16'],
                       help='Model dtype (default: float16)')


def create_server_parser() -> argparse.ArgumentParser:
    """Create server argument parser"""
    parser = argparse.ArgumentParser(description='CPM.cu Server')
    
    # Server-specific parameters
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    # Add model configuration parameters
    add_model_config_args(parser)
    
    return parser


def create_test_parser() -> argparse.ArgumentParser:
    """Create test script argument parser"""
    parser = argparse.ArgumentParser(description='CPM.cu Test Generation')
    
    # Test-specific parameters
    parser.add_argument('--prompt-file', '--prompt_file', type=str, default=None,
                       help='Path to prompt file')
    parser.add_argument('--prompt-text', '--prompt_text', type=str, default=None,
                       help='Direct prompt text')
    parser.add_argument('--prompt-haystack', '--prompt_haystack', type=int,
                       help='Generate haystack prompt with specified length in thousands')
    parser.add_argument('--use-stream', '--use_stream', action='store_true', default=argparse.SUPPRESS,
                       help='Use stream generation')
    parser.add_argument('--no-use-stream', '--no_use_stream', action='store_false', dest='use_stream', default=argparse.SUPPRESS)
    parser.add_argument('--num-generate', '--num_generate', type=int, default=None,
                       help='Number of tokens to generate (default: 256)')
    
    # Add model configuration parameters
    add_model_config_args(parser)
    
    return parser


def merge_args_with_config(args, default_config: Dict[str, Any], is_server: bool = False) -> Dict[str, Any]:
    """Merge arguments with default configuration"""
    config = default_config.copy()
    
    # Override config with any arguments that were explicitly specified
    # (using argparse.SUPPRESS means only specified args appear in the args namespace)
    for key in config.keys():
        if hasattr(args, key):
            arg_value = getattr(args, key)
            # For boolean args, they will be present if specified (due to SUPPRESS)
            # For numerical args, only override if not None
            if key == 'dtype':
                if arg_value is not None:
                    if is_server:
                        # Server keeps string format (fixes memory issue)
                        config[key] = arg_value
                    else:
                        # Test script converts to torch type
                        config[key] = torch.float16 if arg_value == 'float16' else torch.bfloat16
            elif isinstance(config[key], bool):
                # Boolean argument was explicitly specified
                config[key] = arg_value
            elif arg_value is not None:
                # Numerical argument with non-None value
                config[key] = arg_value
    
    # Handle additional arguments that are not in default config (like server-specific params)
    server_specific_args = ['host', 'port', 'path_prefix']
    test_specific_args = ['prompt_file', 'prompt_text', 'prompt_haystack']
    
    additional_args = server_specific_args if is_server else test_specific_args
    for key in additional_args:
        if hasattr(args, key):
            arg_value = getattr(args, key)
            if arg_value is not None:
                config[key] = arg_value
    
    return config


def parse_server_args() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parse server arguments"""
    parser = create_server_parser()
    args = parser.parse_args()
    config = merge_args_with_config(args, get_default_config(), is_server=True)
    return args, config


def parse_test_args() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parse test arguments"""
    parser = create_test_parser()
    args = parser.parse_args()
    config = merge_args_with_config(args, get_default_config(), is_server=False)
    return args, config


def display_config_summary(config: Dict[str, Any], title: str = "Configuration Parameters"):
    """Display configuration parameter summary
    
    Args:
        config: Configuration dictionary
        title: Display title, default is "Configuration Parameters"
    """
    print("=" * 50)
    print(f"{title}:")
    print("=" * 50)
    
    # Basic features
    print(f"Features: eagle={config['apply_eagle']}, quant={config['apply_quant']}, sparse={config['apply_sparse']}")
    
    # Generation parameters (display content based on whether these keys exist in config)
    generation_parts = [f"chunk_length={config['chunk_length']}", f"use_terminators={config['use_terminators']}"]
    if 'num_generate' in config:
        generation_parts.insert(0, f"num_generate={config['num_generate']}")
    if 'use_stream' in config:
        generation_parts.append(f"use_stream={config['use_stream']}")
    print(f"Generation: {', '.join(generation_parts)}")
    
    # Sampling parameters
    print(f"Sampling: temperature={config['temperature']}, random_seed={config['random_seed']}")
    
    # Demo parameters
    print(f"Demo: use_enter={config['use_enter']}, use_decode_enter={config['use_decode_enter']}")
    
    # Other parameters
    print(f"Others: dtype={config['dtype']}, minicpm4_yarn={config['minicpm4_yarn']}, cuda_graph={config['cuda_graph']}, memory_limit={config['memory_limit']}")
    
    # Conditionally display sparse attention parameters
    if config['apply_sparse']:
        print(f"Sparse Attention: sink_window={config['sink_window_size']}, block_window={config['block_window_size']}, sparse_topk_k={config['sparse_topk_k']}, sparse_switch={config['sparse_switch']}, compress_lse={config['apply_compress_lse']}")
    
    # Conditionally display Eagle parameters
    if config['apply_eagle']:
        print(f"Eagle: eagle_num_iter={config['eagle_num_iter']}, eagle_topk_per_iter={config['eagle_topk_per_iter']}, eagle_tree_size={config['eagle_tree_size']}, apply_eagle_quant={config['apply_eagle_quant']}, window_size={config['eagle_window_size']}, frspec_vocab_size={config['frspec_vocab_size']}")
    
    print("=" * 50)
    print() 