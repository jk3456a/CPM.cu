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
    model_group.add_argument('--model-path', '--model_path', '-m', type=str, required=True,
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
    model_group.add_argument('--chunk-length', '--chunk_length', type=int, default=2048,
                            help='Chunk length (default: 2048)')
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

    # Speculative Decoding
    spec_group = parser.add_argument_group('Speculative Decoding')
    spec_group.add_argument('--spec-window-size', '--spec_window_size', type=int, default=1024,
                           help='Speculative decoding window size (default: 1024)')
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
    sparse_group.add_argument('--apply-compress-lse', '--apply_compress_lse', default=True,
                            type=str2bool, nargs='?', const=True,
                            help='Apply LSE compression (default: True)')


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
    generation_group.add_argument('--num-generate', '--num_generate', type=int, default=256,
                       help='Number of tokens to generate (default: 256)')
    generation_group.add_argument('--temperature', '--temp', type=float, default=0.0,
                             help='Temperature (default: 0.0)')
    generation_group.add_argument('--random-seed', '--random_seed', type=int, default=None,
                            help='Random seed')
    generation_group.add_argument('--use-terminators', '--use_terminators', default=True,
                       type=str2bool, nargs='?', const=True,
                       help='Use terminators (default: True)')
    
    # Interactive Features
    interactive_group = parser.add_argument_group('Interactive Features')
    interactive_group.add_argument('--use-enter', '--use_enter', default=False,
                       type=str2bool, nargs='?', const=True,
                       help='Use enter to generate (default: False)')
    interactive_group.add_argument('--use-decode-enter', '--use_decode_enter', default=False,
                       type=str2bool, nargs='?', const=True,
                       help='Use enter before decode phase (default: False)')
    
    add_model_config_args(parser)
    return parser


def args_to_config(args, is_server: bool = False) -> Dict[str, Any]:
    """Convert parsed arguments to configuration dictionary"""
    config = {}
    
    for key, value in vars(args).items():
        if key == 'dtype':
            # Handle dtype conversion
            if is_server:
                config[key] = value  # Server keeps string format
            else:
                config[key] = torch.float16 if value == 'float16' else torch.bfloat16
        else:
            config[key] = value
    
    return config


def parse_server_args() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parse server arguments"""
    parser = create_server_parser()
    args = parser.parse_args()
    config = args_to_config(args, is_server=True)
    return args, config


def parse_test_args() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parse test arguments"""
    parser = create_test_parser()
    args = parser.parse_args()
    config = args_to_config(args, is_server=False)
    return args, config


def display_config_summary(config: Dict[str, Any], title: str = "Configuration Parameters"):
    """Display configuration parameter summary"""
    print("=" * 50)
    print(f"{title}:")
    print("=" * 50)
    
    # Model information
    print(f"Model: {config.get('model_path', 'N/A')}")
    if config.get('draft_model_path'):
        print(f"Draft Model: {config['draft_model_path']}")
    if config.get('frspec_path'):
        print(f"FRSpec: {config['frspec_path']}")
    
    # Generation parameters
    generation_parts = [f"chunk_length={config['chunk_length']}"]
    if 'use_terminators' in config:
        generation_parts.append(f"use_terminators={config['use_terminators']}")
    if 'num_generate' in config:
        generation_parts.insert(0, f"num_generate={config['num_generate']}")
    if 'use_stream' in config:
        generation_parts.append(f"use_stream={config['use_stream']}")
    print(f"Generation: {', '.join(generation_parts)}")
    
    # Sampling parameters (only for test generation)
    if 'temperature' in config or 'random_seed' in config:
        sampling_parts = []
        if 'temperature' in config:
            sampling_parts.append(f"temperature={config['temperature']}")
        if 'random_seed' in config:
            sampling_parts.append(f"random_seed={config['random_seed']}")
        if sampling_parts:
            print(f"Sampling: {', '.join(sampling_parts)}")
    
    # Demo parameters
    demo_parts = []
    if 'use_enter' in config:
        demo_parts.append(f"use_enter={config['use_enter']}")
    if 'use_decode_enter' in config:
        demo_parts.append(f"use_decode_enter={config['use_decode_enter']}")
    if demo_parts:
        print(f"Demo: {', '.join(demo_parts)}")
    
    # System parameters
    print(f"Others: dtype={config['dtype']}, cuda_graph={config['cuda_graph']}, memory_limit={config['memory_limit']}")
    
    # Sparse attention parameters (if enabled)
    if config.get('model_type') == 'minicpm4':
        sparse_parts = [
            f"sink_window={config['sink_window_size']}",
            f"block_window={config['block_window_size']}",
            f"sparse_topk_k={config['sparse_topk_k']}",
            f"sparse_switch={config['sparse_switch']}",
            f"compress_lse={config['apply_compress_lse']}"
        ]
        print(f"Sparse Attention: {', '.join(sparse_parts)}")
    
    # Speculative decoding parameters
    spec_parts = [
        f"num_iter={config.get('spec_num_iter', 2)}",
        f"topk_per_iter={config.get('spec_topk_per_iter', 10)}",
        f"tree_size={config.get('spec_tree_size', 12)}",
        f"frspec_vocab_size={config.get('frspec_vocab_size', 32768)}"
    ]
    print(f"Speculative: {', '.join(spec_parts)}")
    
    print("=" * 50)
    print()


# Backward compatibility
class ConfigurationProcessor:
    @staticmethod
    def args_to_config(args, is_server: bool = False) -> Dict[str, Any]:
        """Deprecated: Use args_to_config function instead"""
        return args_to_config(args, is_server)


class ConfigurationDisplay:
    @staticmethod
    def display_config_summary(config: Dict[str, Any], title: str = "Configuration Parameters"):
        """Deprecated: Use display_config_summary function instead"""
        return display_config_summary(config, title) 