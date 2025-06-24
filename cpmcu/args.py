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
    model_group.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                            help='Model dtype (default: float16)')
    model_group.add_argument('--chunk-length', '--chunk_length', type=int, default=2048,
                            help='Chunk length (default: 2048)')

    # System Features Group
    system_group = parser.add_argument_group('System Features', 'System-level configuration parameters')
    system_group.add_argument('--cuda-graph', '--cuda_graph', default=True,
                            type=str2bool, nargs='?', const=True,
                            help='Use CUDA graph optimization (default: True). Values: true/false, yes/no, 1/0, or just --cuda-graph for True')
    system_group.add_argument('--memory-limit', '--memory_limit', type=float, default=0.9,
                            help='Memory limit (default: 0.9)')
    system_group.add_argument('--random-seed', '--random_seed', type=int, default=None,
                            help='Random seed')

    # Speculative Decoding Group
    spec_group = parser.add_argument_group('Speculative Decoding', 'Speculative decoding configuration')
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

    # Sparse Attention Group
    sparse_group = parser.add_argument_group('Sparse Attention', 'Sparse attention mechanism configuration')
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
                            help='Apply LSE compression (default: True). Values: true/false, yes/no, 1/0, or just --apply-compress-lse for True')

    # Generation Parameters Group
    generation_group = parser.add_argument_group('Generation Parameters', 'Text generation configuration parameters')
    generation_group.add_argument('--temperature', '--temp', type=float, default=0.0,
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
    parser.add_argument('--use-chat-template', '--use_chat_template', default=True,
                       type=str2bool, nargs='?', const=True,
                       help='Use chat template for prompt formatting (default: True). Values: true/false, yes/no, 1/0, or just --use-chat-template for True')
    
    # Add use_stream parameter for test parser
    parser.add_argument('--use-stream', '--use_stream', default=True,
                       type=str2bool, nargs='?', const=True,
                       help='Use stream generation (default: True). Values: true/false, yes/no, 1/0, or just --use-stream for True')
    
    parser.add_argument('--num-generate', '--num_generate', type=int, default=256,
                       help='Number of tokens to generate (default: 256)')
    
    # Test-specific generation parameters
    parser.add_argument('--use-terminators', '--use_terminators', default=True,
                       type=str2bool, nargs='?', const=True,
                       help='Use terminators (default: True). Values: true/false, yes/no, 1/0, or just --use-terminators for True')
    
    # Interactive Features
    parser.add_argument('--use-enter', '--use_enter', default=False,
                       type=str2bool, nargs='?', const=True,
                       help='Use enter to generate (default: False). Values: true/false, yes/no, 1/0, or just --use-enter for True')
    parser.add_argument('--use-decode-enter', '--use_decode_enter', default=False,
                       type=str2bool, nargs='?', const=True,
                       help='Use enter before decode phase (default: False). Values: true/false, yes/no, 1/0, or just --use-decode-enter for True')
    
    # Add model configuration parameters
    add_model_config_args(parser)
    
    return parser


class ConfigurationProcessor:
    """统一的配置处理类"""
    
    @staticmethod
    def args_to_config(args, is_server: bool = False) -> Dict[str, Any]:
        """Convert parsed arguments to configuration dictionary"""
        config = {}
        
        # Convert all argument attributes to config
        for key, value in vars(args).items():
            if key == 'dtype':
                # Handle dtype conversion
                if is_server:
                    # Server keeps string format for JSON compatibility
                    config[key] = value
                else:
                    # Test script converts to torch type
                    config[key] = torch.float16 if value == 'float16' else torch.bfloat16
            else:
                config[key] = value
        
        return config


def parse_server_args() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parse server arguments"""
    parser = create_server_parser()
    args = parser.parse_args()
    config = ConfigurationProcessor.args_to_config(args, is_server=True)
    return args, config


def parse_test_args() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """Parse test arguments"""
    parser = create_test_parser()
    args = parser.parse_args()
    config = ConfigurationProcessor.args_to_config(args, is_server=False)
    return args, config


class ConfigurationDisplay:
    """统一的配置显示类"""
    
    @staticmethod
    def display_config_summary(config: Dict[str, Any], title: str = "Configuration Parameters"):
        """Display configuration parameter summary with improved formatting"""
        print("=" * 50)
        print(f"{title}:")
        print("=" * 50)
        
        # Model information section
        ConfigurationDisplay._display_model_info(config)
        
        # Features section
        ConfigurationDisplay._display_features(config)
        
        # Generation parameters section
        ConfigurationDisplay._display_generation_params(config)
        
        # Sampling parameters section
        ConfigurationDisplay._display_sampling_params(config)
        
        # Demo parameters section
        ConfigurationDisplay._display_demo_params(config)
        
        # System parameters section
        ConfigurationDisplay._display_system_params(config)
        
        # Conditional sections
        if config.get('apply_sparse', False):
            ConfigurationDisplay._display_sparse_params(config)
        
        ConfigurationDisplay._display_spec_params(config)
        
        print("=" * 50)
        print()
    
    @staticmethod
    def _display_model_info(config):
        """显示模型信息"""
        print(f"Model: {config.get('model_path', 'N/A')}")
        if config.get('draft_model_path'):
            print(f"Draft Model: {config['draft_model_path']}")
        if config.get('frspec_path'):
            print(f"FRSpec: {config['frspec_path']}")
    
    @staticmethod
    def _display_features(config):
        """显示功能特性"""
        print(f"Features: speculative=auto, quant=auto, sparse={config.get('apply_sparse', False)}")
    
    @staticmethod
    def _display_generation_params(config):
        """显示生成参数"""
        generation_parts = [f"chunk_length={config['chunk_length']}"]
        if 'use_terminators' in config:
            generation_parts.append(f"use_terminators={config['use_terminators']}")
        if 'num_generate' in config:
            generation_parts.insert(0, f"num_generate={config['num_generate']}")
        if 'use_stream' in config:
            generation_parts.append(f"use_stream={config['use_stream']}")
        print(f"Generation: {', '.join(generation_parts)}")
    
    @staticmethod
    def _display_sampling_params(config):
        """显示采样参数"""
        print(f"Sampling: temperature={config['temperature']}, random_seed={config['random_seed']}")
    
    @staticmethod
    def _display_demo_params(config):
        """显示演示参数"""
        demo_parts = []
        if 'use_enter' in config:
            demo_parts.append(f"use_enter={config['use_enter']}")
        if 'use_decode_enter' in config:
            demo_parts.append(f"use_decode_enter={config['use_decode_enter']}")
        if demo_parts:
            print(f"Demo: {', '.join(demo_parts)}")
    
    @staticmethod
    def _display_system_params(config):
        """显示系统参数"""
        print(f"Others: dtype={config['dtype']}, cuda_graph={config['cuda_graph']}, memory_limit={config['memory_limit']}")
    
    @staticmethod
    def _display_sparse_params(config):
        """显示稀疏注意力参数"""
        sparse_parts = [
            f"sink_window={config['sink_window_size']}",
            f"block_window={config['block_window_size']}",
            f"sparse_topk_k={config['sparse_topk_k']}",
            f"sparse_switch={config['sparse_switch']}",
            f"compress_lse={config['apply_compress_lse']}"
        ]
        print(f"Sparse Attention: {', '.join(sparse_parts)}")
    
    @staticmethod
    def _display_spec_params(config):
        """显示投机解码参数"""
        spec_parts = [
            f"num_iter={config.get('spec_num_iter', 2)}",
            f"topk_per_iter={config.get('spec_topk_per_iter', 10)}",
            f"tree_size={config.get('spec_tree_size', 12)}",
            f"frspec_vocab_size={config.get('frspec_vocab_size', 32768)}"
        ]
        print(f"Speculative: {', '.join(spec_parts)}") 