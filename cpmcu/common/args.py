#!/usr/bin/env python3
"""
CPM.cu Unified Argument Processing

Unified argument processing module for generic model support
"""

import argparse
import sys
import torch
from typing import Dict, Any, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


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
    """Display configuration parameter summary using Rich."""
    console = Console()
    
    main_table = Table(box=None, show_header=False, pad_edge=False)

    # Helper to add a section table
    def add_section(title, data, color):
        table = Table(show_header=False, box=None, padding=(0, 2, 0, 2))
        table.add_column(style="bold")
        table.add_column()
        for key, value in data:
            if value is not None:
                styled_value = f"[green]True[/green]" if value is True else f"[red]False[/red]" if value is False else str(value)
                table.add_row(f"{key}:", styled_value)
        
        panel = Panel(table, title=f"[{color}]{title}[/{color}]", border_style=color, expand=False)
        main_table.add_row(panel)

    # --- Model Configuration ---
    model_data = [
        ("Model Path", getattr(args, 'model_path', 'N/A')),
        ("Model Type", getattr(args, 'model_type', 'auto')),
        ("Data Type", getattr(args, 'dtype', 'float16')),
        ("Draft Model", getattr(args, 'draft_model_path', None)),
        ("FRSpec Path", getattr(args, 'frspec_path', None)),
        ("MiniCPM4 YARN", getattr(args, 'minicpm4_yarn', False) if hasattr(args, 'minicpm4_yarn') else None),
    ]
    add_section("Model Configuration", [(k, v) for k, v in model_data if v is not None], "cyan")

    # --- Server Configuration ---
    if hasattr(args, 'host') or hasattr(args, 'port'):
        server_data = [
            ("Host", getattr(args, 'host', 'N/A')),
            ("Port", getattr(args, 'port', 'N/A')),
        ]
        add_section("Server Configuration", server_data, "magenta")

    # --- Prompt Configuration ---
    if hasattr(args, 'prompt_file') or hasattr(args, 'prompt_text') or hasattr(args, 'use_chat_template'):
        prompt_data = [
            ("Prompt File", bool(getattr(args, 'prompt_file', None))),
            ("Prompt Text", bool(getattr(args, 'prompt_text', None))),
            ("Use Chat Template", getattr(args, 'use_chat_template', 'N/A')),
        ]
        add_section("Prompt Configuration", prompt_data, "yellow")

    # --- Generation Configuration ---
    if hasattr(args, 'num_generate'):
        gen_data = [
            ("Number to Generate", getattr(args, 'num_generate', None)),
            ("Use Stream", getattr(args, 'use_stream', None)),
            ("Ignore EOS", getattr(args, 'ignore_eos', None)),
            ("Temperature", getattr(args, 'temperature', None)),
            ("Random Seed", getattr(args, 'random_seed', None)),
        ]
        add_section("Generation Configuration", [(k, v) for k, v in gen_data if v is not None], "blue")

    # --- System Configuration ---
    system_data = [
        ("CUDA Graph", getattr(args, 'cuda_graph', True)),
        ("Memory Limit", getattr(args, 'memory_limit', 0.9)),
        ("Chunked Prefill Size", getattr(args, 'chunk_length', 2048)),
    ]
    add_section("System Configuration", system_data, "white")

    # --- Speculative Decoding ---
    if hasattr(args, 'draft_model_path') and getattr(args, 'draft_model_path'):
        spec_data = [
            ("Window Size", getattr(args, 'spec_window_size', 1024)),
            ("Number of Iterations", getattr(args, 'spec_num_iter', 2)),
            ("Top-K per Iteration", getattr(args, 'spec_topk_per_iter', 10)),
            ("Tree Size", getattr(args, 'spec_tree_size', 12)),
            ("FRSpec Vocab Size", getattr(args, 'frspec_vocab_size', 32768) if hasattr(args, 'frspec_path') and args.frspec_path else None),
        ]
        add_section("Speculative Decoding", [(k,v) for k,v in spec_data if v is not None], "purple")

    # --- Sparse Attention ---
    if getattr(args, 'model_type', None) == 'minicpm4':
        sparse_data = [
            ("Sink Window Size", getattr(args, 'sink_window_size', 1)),
            ("Block Window Size", getattr(args, 'block_window_size', 8)),
            ("Sparse Top-K", getattr(args, 'sparse_topk_k', 64)),
            ("Sparse Switch", getattr(args, 'sparse_switch', 0)),
            ("Use Compress LSE", getattr(args, 'use_compress_lse', True)),
        ]
        add_section("Sparse Attention", sparse_data, "red")

    console.print(Panel(main_table, title=f"[bold yellow]{title}[/bold yellow]", border_style="bold", expand=False)) 