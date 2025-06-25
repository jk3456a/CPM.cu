#!/usr/bin/env python3
"""
CPM.cu Unified Argument Processing

Unified argument processing module for generic model support
"""

import argparse
import sys
import torch
from typing import Dict, Any, Tuple


def str2bool(v):
    """Convert string to boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_model_config_args(parser: argparse.ArgumentParser):
    """Add common model configuration arguments"""
    
    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model-path', '--model_path', '--model', type=str, required=True,
                           help='Path to the main model (local path or HuggingFace repo)')
    model_group.add_argument('--draft-model-path', '--draft_model_path', type=str, default=None,
                           help='Path to draft model for speculative decoding')
    model_group.add_argument('--frspec-path', '--frspec_path', type=str, default=None,
                           help='Path to frequency speculative vocabulary file (.pt file)')
    model_group.add_argument('--model-type', '--model_type', type=str, default='auto',
                           choices=['auto', 'llama', 'minicpm', 'minicpm4'],
                           help='Model type (default: auto-detect)')
    model_group.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                            help='Model dtype (default: float16)')
    model_group.add_argument('--minicpm4-yarn', '--minicpm4_yarn', default=False,
                            type=str2bool, nargs='?', const=True,
                            help='Enable MiniCPM4 YARN for long context support (default: False)')

    # System Configuration
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--cuda-graph', '--cuda_graph', default=True,
                            type=str2bool, nargs='?', const=True,
                            help='Use CUDA graph optimization (default: True)')
    system_group.add_argument('--memory-limit', '--memory_limit', type=float, default=0.9,
                            help='Memory limit (default: 0.9)')
    system_group.add_argument('--chunk-length', '--chunk_length', type=int, default=2048,
                            help='Chunked prefill size (default: 2048)')

    # Speculative Decoding
    spec_group = parser.add_argument_group('Speculative Decoding')
    spec_group.add_argument('--spec-window-size', '--spec_window_size', type=int, default=1024,
                           help='Speculative decoding slidingwindow size (default: 1024)')
    spec_group.add_argument('--spec-num-iter', '--spec_num_iter', type=int, default=2,
                           help='Speculative decoding number of iterations (default: 2)')
    spec_group.add_argument('--spec-topk-per-iter', '--spec_topk_per_iter', type=int, default=10,
                           help='Speculative decoding top-k per iteration (default: 10)')
    spec_group.add_argument('--spec-tree-size', '--spec_tree_size', type=int, default=12,
                           help='Speculative decoding tree size (default: 12)')
    spec_group.add_argument('--frspec-vocab-size', '--frspec_vocab_size', type=int, default=32768,
                           help='Frequent speculation vocab size (default: 32768)')

    # Sparse Attention
    sparse_group = parser.add_argument_group('Sparse Attention')
    sparse_group.add_argument('--sink-window-size', '--sink_window_size', type=int, default=1,
                            help='Sink window size (default: 1)')
    sparse_group.add_argument('--block-window-size', '--block_window_size', type=int, default=8,
                            help='Block window size (default: 8)')
    sparse_group.add_argument('--sparse-topk-k', '--sparse_topk_k', type=int, default=64,
                            help='Sparse attention top-k (default: 64)')
    sparse_group.add_argument('--sparse-switch', '--sparse_switch', type=int, default=0,
                            help='Sparse switch threshold (default: 0)')
    sparse_group.add_argument('--use-compress-lse', '--use_compress_lse', default=True,
                            type=str2bool, nargs='?', const=True,
                            help='Use LSE compression (default: True)')


def create_server_parser() -> argparse.ArgumentParser:
    """Create server argument parser"""
    parser = argparse.ArgumentParser(description='CPM.cu Server')
    
    # Server Configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    server_group.add_argument('--port', type=int, default=8000, help='Server port')
    
    add_model_config_args(parser)
    return parser


def create_test_parser() -> argparse.ArgumentParser:
    """Create test script argument parser"""
    parser = argparse.ArgumentParser(description='CPM.cu Test Generation')
    
    # Prompt Configuration
    prompt_group = parser.add_argument_group('Prompt Configuration')
    prompt_group.add_argument('--prompt-file', '--prompt_file', type=str, default=None,
                       help='Path to prompt file')
    prompt_group.add_argument('--prompt-text', '--prompt_text', type=str, default=None,
                       help='Direct prompt text')
    prompt_group.add_argument('--use-chat-template', '--use_chat_template', default=True,
                       type=str2bool, nargs='?', const=True,
                       help='Use chat template for prompt formatting (default: True)')
    
    # Generation Configuration
    generation_group = parser.add_argument_group('Generation Configuration')
    generation_group.add_argument('--use-stream', '--use_stream', default=True,
                       type=str2bool, nargs='?', const=True,
                       help='Use stream generation (default: True)')
    generation_group.add_argument('--num-generate', '--num_generate', type=int, default=1024,
                       help='Number of tokens to generate (default: 1024)')
    generation_group.add_argument('--temperature', '--temp', type=float, default=0.0,
                             help='Temperature (default: 0.0)')
    generation_group.add_argument('--random-seed', '--random_seed', type=int, default=None,
                            help='Random seed')
    generation_group.add_argument('--ignore-eos', '--ignore_eos', default=False,
                       type=str2bool, nargs='?', const=True,
                       help='Ignore EOS tokens during generation (default: False)')
    
    # Interactive Features
    interactive_group = parser.add_argument_group('Interactive Features')
    interactive_group.add_argument('--use-enter', '--use_enter', default=False,
                       type=str2bool, nargs='?', const=True,
                       help='[DEPRECATED] Use enter to generate (default: False)')
    interactive_group.add_argument('--use-decode-enter', '--use_decode_enter', default=False,
                       type=str2bool, nargs='?', const=True,
                       help='[DEPRECATED] Use enter before decode phase (default: False)')
    
    add_model_config_args(parser)
    return parser


def parse_server_args() -> argparse.Namespace:
    """Parse server arguments"""
    parser = create_server_parser()
    args = parser.parse_args()
    return args


def parse_test_args() -> argparse.Namespace:
    """Parse test arguments"""
    parser = create_test_parser()
    args = parser.parse_args()
    return args


def display_config_summary(args: argparse.Namespace, title: str = "Configuration Parameters"):
    """Display configuration parameter summary grouped by categories"""
    print("=" * 60)
    print(f"{title}:")
    print("=" * 60)
    
    # Model Configuration Group
    print("Model Configuration:")
    print(f"  • Model Path: {getattr(args, 'model_path', 'N/A')}")
    print(f"  • Model Type: {getattr(args, 'model_type', 'auto')}")
    print(f"  • Data Type: {getattr(args, 'dtype', 'float16')}")
    if hasattr(args, 'draft_model_path') and getattr(args, 'draft_model_path'):
        print(f"  • Draft Model: {args.draft_model_path}")
    if hasattr(args, 'frspec_path') and getattr(args, 'frspec_path'):
        print(f"  • FRSpec Path: {args.frspec_path}")
    if hasattr(args, 'minicpm4_yarn'):
        print(f"  • MiniCPM4 YARN: {getattr(args, 'minicpm4_yarn', False)}")
    print()
    
    # Server Configuration Group (only if server parameters exist)
    if hasattr(args, 'host') or hasattr(args, 'port'):
        print("Server Configuration:")
        print(f"  • Host: {getattr(args, 'host', 'N/A')}")
        print(f"  • Port: {getattr(args, 'port', 'N/A')}")
        print()
    
    # Prompt Configuration Group (only if prompt parameters exist)
    if hasattr(args, 'prompt_file') or hasattr(args, 'prompt_text') or hasattr(args, 'use_chat_template'):
        print("Prompt Configuration:")
        print(f"  • Prompt File: {bool(getattr(args, 'prompt_file', None))}")
        print(f"  • Prompt Text: {bool(getattr(args, 'prompt_text', None))}")
        print(f"  • Use Chat Template: {getattr(args, 'use_chat_template', 'N/A')}")
        print()
    
    # Generation Configuration Group (only if generation parameters exist)
    if hasattr(args, 'num_generate') or hasattr(args, 'use_stream') or hasattr(args, 'ignore_eos') or hasattr(args, 'temperature') or hasattr(args, 'random_seed'):
        print("Generation Configuration:")
        if hasattr(args, 'num_generate'):
            print(f"  • Number to Generate: {args.num_generate}")
        if hasattr(args, 'use_stream'):
            print(f"  • Use Stream: {args.use_stream}")
        if hasattr(args, 'ignore_eos'):
            print(f"  • Ignore EOS: {args.ignore_eos}")
        if hasattr(args, 'temperature'):
            print(f"  • Temperature: {args.temperature}")
        if hasattr(args, 'random_seed'):
            print(f"  • Random Seed: {args.random_seed}")
        print()
    
    # System Configuration Group
    print("System Configuration:")
    print(f"  • CUDA Graph: {getattr(args, 'cuda_graph', True)}")
    print(f"  • Memory Limit: {getattr(args, 'memory_limit', 0.9)}")
    print(f"  • Chunked Prefill Size: {getattr(args, 'chunk_length', 2048)}")
    print()
    
    # Speculative Decoding Group (only if draft model is provided)
    if hasattr(args, 'draft_model_path') and getattr(args, 'draft_model_path'):
        print("Speculative Decoding:")
        print(f"  • Window Size: {getattr(args, 'spec_window_size', 1024)}")
        print(f"  • Number of Iterations: {getattr(args, 'spec_num_iter', 2)}")
        print(f"  • Top-K per Iteration: {getattr(args, 'spec_topk_per_iter', 10)}")
        print(f"  • Tree Size: {getattr(args, 'spec_tree_size', 12)}")
        if hasattr(args, 'frspec_path') and getattr(args, 'frspec_path'):
            print(f"  • FRSpec Vocab Size: {getattr(args, 'frspec_vocab_size', 32768)}")
        print()
    
    # Sparse Attention Group (for MiniCPM4)
    if getattr(args, 'model_type', None) == 'minicpm4':
        print("Sparse Attention:")
        print(f"  • Sink Window Size: {getattr(args, 'sink_window_size', 1)}")
        print(f"  • Block Window Size: {getattr(args, 'block_window_size', 8)}")
        print(f"  • Sparse Top-K: {getattr(args, 'sparse_topk_k', 64)}")
        print(f"  • Sparse Switch: {getattr(args, 'sparse_switch', 0)}")
        print(f"  • Use Compress LSE: {getattr(args, 'use_compress_lse', True)}")
        print()
    
    print("=" * 60)
    print() 