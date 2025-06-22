#!/usr/bin/env python3
"""
CPM.cu Unified Argument Processing

Unified argument processing module for generic model support
"""

import argparse
import sys
import torch
from typing import Dict, Any, Tuple
from .utils import get_default_config


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
    """Add common model configuration arguments with logical grouping"""
    
    # Model Path Configuration Group
    model_group = parser.add_argument_group('Model Configuration', 'Model path and type configuration')
    model_group.add_argument('--model-path', '--model_path', '-m', type=str, required=True,
                           help='Path to the main model (local path or HuggingFace repo)')
    model_group.add_argument('--draft-model-path', '--draft_model_path', type=str, default=None,
                           help='Path to draft model for speculative decoding (local path or HuggingFace repo)')
    model_group.add_argument('--frspec-path', '--frspec_path', type=str, default=None,
                           help='Path to frequency speculative vocabulary file (.pt file)')
    model_group.add_argument('--model-type', '--model_type', type=str, default='auto',
                           choices=['auto', 'llama', 'minicpm', 'minicpm4'],
                           help='Model type (default: auto-detect)')
    
    # Model Runtime Configuration
    model_group.add_argument('--dtype', type=str, default=None, choices=['float16', 'bfloat16'],
                            help='Model dtype (default: float16)')
    model_group.add_argument('--chunk-length', '--chunk_length', type=int, default=None,
                                 help='Chunk length (default: 2048)')

    # System Features Group
    system_group = parser.add_argument_group('System Features', 'System-level configuration parameters')
    system_group.add_argument('--cuda-graph', '--cuda_graph', type=str2bool, nargs='?', const=True, default=None,
                            help='Use CUDA graph optimization (default: True). Values: true/false, yes/no, 1/0, or just --cuda-graph for True')
    system_group.add_argument('--memory-limit', '--memory_limit', type=float, default=None,
                            help='Memory limit (default: 0.9)')
    system_group.add_argument('--random-seed', '--random_seed', type=int, default=None,
                            help='Random seed')

    # Speculative Decoding Group
    spec_group = parser.add_argument_group('Speculative Decoding', 'Speculative decoding configuration')
    spec_group.add_argument('--apply-speculative', '--apply_speculative', type=str2bool, nargs='?', const=True, default=None,
                           help='Use speculative decoding (default: False). Values: true/false, yes/no, 1/0, or just --apply-speculative for True')
    spec_group.add_argument('--spec-window-size', '--spec_window_size', type=int, default=None,
                           help='Speculative decoding window size (default: 1024)')
    spec_group.add_argument('--spec-num-iter', '--spec_num_iter', type=int, default=None,
                           help='Speculative decoding number of iterations (default: 2)')
    spec_group.add_argument('--spec-topk-per-iter', '--spec_topk_per_iter', type=int, default=None,
                           help='Speculative decoding top-k per iteration (default: 10)')
    spec_group.add_argument('--spec-tree-size', '--spec_tree_size', type=int, default=None,
                           help='Speculative decoding tree size (default: 12)')
    spec_group.add_argument('--frspec-vocab-size', '--frspec_vocab_size', type=int, default=None,
                           help='Frequent speculation vocab size (default: 32768)')

    # Quantization Group
    quant_group = parser.add_argument_group('Quantization', 'Model quantization configuration')
    quant_group.add_argument('--apply-quant', '--apply_quant', type=str2bool, nargs='?', const=True, default=None,
                               help='Use quantized model (default: False). Values: true/false, yes/no, 1/0, or just --apply-quant for True')
    quant_group.add_argument('--apply-spec-quant', '--apply_spec_quant', type=str2bool, nargs='?', const=True, default=None,
                           help='Use quantized speculative model (default: False). Values: true/false, yes/no, 1/0, or just --apply-spec-quant for True')

    # Sparse Attention Group
    sparse_group = parser.add_argument_group('Sparse Attention', 'Sparse attention mechanism configuration')
    sparse_group.add_argument('--apply-sparse', '--apply_sparse', type=str2bool, nargs='?', const=True, default=None,
                            help='Use sparse attention (default: False). Values: true/false, yes/no, 1/0, or just --apply-sparse for True')
    sparse_group.add_argument('--apply-compress-lse', '--apply_compress_lse', type=str2bool, nargs='?', const=True, default=None,
                            help='Apply LSE compression (default: True). Values: true/false, yes/no, 1/0, or just --apply-compress-lse for True')
    sparse_group.add_argument('--sink-window-size', '--sink_window_size', type=int, default=None,
                            help='Sink window size (default: 1)')
    sparse_group.add_argument('--block-window-size', '--block_window_size', type=int, default=None,
                            help='Block window size (default: 8)')
    sparse_group.add_argument('--sparse-topk-k', '--sparse_topk_k', type=int, default=None,
                            help='Sparse attention top-k (default: 64)')
    sparse_group.add_argument('--sparse-switch', '--sparse_switch', type=int, default=None,
                            help='Sparse switch threshold (default: 1)')

    # Generation Parameters Group
    generation_group = parser.add_argument_group('Generation Parameters', 'Text generation configuration parameters')
    generation_group.add_argument('--temperature', '--temp', type=float, default=None,
                                 help='Temperature (default: 0.0)')


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
    
    # Chat template configuration
    parser.add_argument('--use-chat-template', '--use_chat_template', type=str2bool, nargs='?', const=True, default=None,
                       help='Use chat template for prompt formatting (default: True). Values: true/false, yes/no, 1/0, or just --use-chat-template for True')
    
    # Add use_stream parameter for test parser
    parser.add_argument('--use-stream', '--use_stream', type=str2bool, nargs='?', const=True, default=None,
                       help='Use stream generation (default: True). Values: true/false, yes/no, 1/0, or just --use-stream for True')
    
    parser.add_argument('--num-generate', '--num_generate', type=int, default=None,
                       help='Number of tokens to generate (default: 256)')
    
    # Test-specific generation parameters
    parser.add_argument('--use-terminators', '--use_terminators', type=str2bool, nargs='?', const=True, default=None,
                       help='Use terminators (default: True). Values: true/false, yes/no, 1/0, or just --use-terminators for True')
    
    # Interactive Features
    parser.add_argument('--use-enter', '--use_enter', type=str2bool, nargs='?', const=True, default=None,
                       help='Use enter to generate (default: False). Values: true/false, yes/no, 1/0, or just --use-enter for True')
    parser.add_argument('--use-decode-enter', '--use_decode_enter', type=str2bool, nargs='?', const=True, default=None,
                       help='Use enter before decode phase (default: False). Values: true/false, yes/no, 1/0, or just --use-decode-enter for True')
    
    # Add model configuration parameters
    add_model_config_args(parser)
    
    return parser


def merge_args_with_config(args, default_config: Dict[str, Any], is_server: bool = False) -> Dict[str, Any]:
    """Merge arguments with default configuration"""
    config = default_config.copy()
    
    # Override config with any arguments that were explicitly specified
    for key in config.keys():
        if hasattr(args, key):
            arg_value = getattr(args, key)
            # Handle dtype conversion
            if key == 'dtype':
                if arg_value is not None:
                    if is_server:
                        # Server keeps string format for JSON serialization
                        config[key] = arg_value
                    else:
                        # Test script converts to torch type
                        config[key] = torch.float16 if arg_value == 'float16' else torch.bfloat16
            elif isinstance(config[key], bool):
                # Boolean argument: only override if explicitly specified (not None)
                if arg_value is not None:
                    config[key] = arg_value
            elif arg_value is not None:
                # Numerical argument with non-None value
                config[key] = arg_value
    
    # Handle additional arguments that are not in default config (like server-specific params)
    server_specific_args = ['host', 'port', 'model_path', 'draft_model_path', 'frspec_path', 'model_type']
    test_specific_args = ['prompt_file', 'prompt_text', 'use_chat_template', 'model_path', 'draft_model_path', 'frspec_path', 'model_type']
    
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
    
    # Basic model info
    print(f"Model: {config.get('model_path', 'N/A')}")
    if config.get('draft_model_path'):
        print(f"Draft Model: {config['draft_model_path']}")
    if config.get('frspec_path'):
        print(f"FRSpec: {config['frspec_path']}")
    
    # Basic features
    print(f"Features: speculative={config.get('apply_speculative', False)}, quant={config.get('apply_quant', False)}, sparse={config.get('apply_sparse', False)}")
    
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
    print(f"Others: dtype={config['dtype']}, cuda_graph={config['cuda_graph']}, memory_limit={config['memory_limit']}")
    
    # Conditionally display sparse attention parameters
    if config.get('apply_sparse', False):
        print(f"Sparse Attention: sink_window={config['sink_window_size']}, block_window={config['block_window_size']}, sparse_topk_k={config['sparse_topk_k']}, sparse_switch={config['sparse_switch']}, compress_lse={config['apply_compress_lse']}")
    
    # Conditionally display speculative decoding parameters
    if config.get('apply_speculative', False):
        print(f"Speculative: num_iter={config.get('spec_num_iter', 2)}, topk_per_iter={config.get('spec_topk_per_iter', 10)}, tree_size={config.get('spec_tree_size', 12)}, spec_quant={config.get('apply_spec_quant', False)}, window_size={config.get('spec_window_size', 1024)}, frspec_vocab_size={config.get('frspec_vocab_size', 32768)}")
    
    print("=" * 50)
    print() 