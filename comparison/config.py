#!/usr/bin/env python3
"""
Configuration module for logits comparison
Handles all configuration-related functionality
"""

import argparse
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ComparisonConfig:
    """Configuration for logits comparison"""
    spec_num_iter: int
    spec_tree_size: int
    comparison_steps: int
    config_name: str
    model_path: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    draft_model_path: str = "jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B"
    prompt_file: str = "prompt_small.txt"
    memory_limit: float = 0.75
    chunk_length: int = 1024
    cuda_graph: bool = False
    minicpm4_yarn: bool = True
    use_chat_template: bool = False

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate_config()
    
    def validate_config(self):
        """Validate configuration parameters"""
        if self.spec_tree_size < 2:
            raise ValueError(f"spec_tree_size ({self.spec_tree_size}) must be at least 2")
        
        max_topk_per_iter = self.spec_tree_size - 1
        if max_topk_per_iter <= 0:
            raise ValueError(f"tree_size ({self.spec_tree_size}) too small, must be at least 2")
    
    def get_topk_per_iter(self) -> int:
        """Calculate appropriate topk_per_iter to satisfy constraint: topk_per_iter <= tree_size - 1"""
        max_topk_per_iter = self.spec_tree_size - 1
        topk_per_iter = min(8, max_topk_per_iter)  # Default is 8, but ensure it's valid
        return topk_per_iter
    
    def to_args_list(self) -> list:
        """Convert configuration to argument list for argument parser"""
        topk_per_iter = self.get_topk_per_iter()
        
        args_list = [
            "--model-path", self.model_path,
            "--draft-model-path", self.draft_model_path,
            "--prompt-file", self.prompt_file,
            "--spec-num-iter", str(self.spec_num_iter),
            "--spec-topk-per-iter", str(topk_per_iter),
            "--spec-tree-size", str(self.spec_tree_size),
            "--num-generate", str(self.comparison_steps),
            "--memory-limit", str(self.memory_limit),
            "--chunk-length", str(self.chunk_length),
            "--cuda-graph", str(self.cuda_graph).lower()
        ]
        
        if self.minicpm4_yarn:
            args_list.append("--minicpm4-yarn")
        
        return args_list


def create_comparison_config_parser() -> argparse.ArgumentParser:
    """Create argument parser for comparison configuration"""
    parser = argparse.ArgumentParser(description='配置逻辑比较参数')
    
    parser.add_argument('--comparison-steps', '--comparison_steps', type=int, default=20,
                       help='配置间比较的token数量 (默认: 20)')
    parser.add_argument('--spec-num-iter', type=int, default=5,
                       help='投机迭代次数 (默认: 5)')
    parser.add_argument('--spec-tree-size', type=int, default=32,
                       help='投机树大小 (默认: 32)')
    parser.add_argument('--config-name', type=str, default=None,
                       help='配置名称，用于标识不同的配置')
    parser.add_argument('--model-path', type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct",
                       help='主模型路径')
    parser.add_argument('--draft-model-path', type=str, default="jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B",
                       help='草稿模型路径')
    parser.add_argument('--prompt-file', type=str, default="prompt_small.txt",
                       help='提示文件路径')
    parser.add_argument('--memory-limit', type=float, default=0.75,
                       help='内存限制 (默认: 0.75)')
    parser.add_argument('--chunk-length', type=int, default=1024,
                       help='块长度 (默认: 1024)')
    parser.add_argument('--cuda-graph', action='store_true',
                       help='启用CUDA图')
    parser.add_argument('--minicpm4-yarn', action='store_true', default=True,
                       help='启用MiniCPM4 YARN配置')
    parser.add_argument('--use-chat-template', action='store_true',
                       help='使用聊天模板')
    
    return parser


def create_config_from_args(args: argparse.Namespace) -> ComparisonConfig:
    """从命令行参数创建配置对象"""
    config_name = args.config_name
    if config_name is None:
        config_name = f"iter{args.spec_num_iter}_tree{args.spec_tree_size}"
    
    return ComparisonConfig(
        spec_num_iter=args.spec_num_iter,
        spec_tree_size=args.spec_tree_size,
        comparison_steps=args.comparison_steps,
        config_name=config_name,
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        prompt_file=args.prompt_file,
        memory_limit=args.memory_limit,
        chunk_length=args.chunk_length,
        cuda_graph=args.cuda_graph,
        minicpm4_yarn=args.minicpm4_yarn,
        use_chat_template=args.use_chat_template
    )


def create_two_config_comparison_parser() -> argparse.ArgumentParser:
    """创建用于两个配置比较的参数解析器"""
    parser = argparse.ArgumentParser(description='比较两种不同投机解码配置之间的逻辑')
    
    parser.add_argument('--comparison-steps', '--comparison_steps', type=int, default=20,
                       help='配置间比较的token数量 (默认: 20)')
    
    # Configuration 1
    parser.add_argument('--config1-iter', type=int, default=5,
                       help='配置1的spec_num_iter (默认: 5)')
    parser.add_argument('--config1-tree-size', type=int, default=32,
                       help='配置1的spec_tree_size (默认: 32)')
    
    # Configuration 2
    parser.add_argument('--config2-iter', type=int, default=2,
                       help='配置2的spec_num_iter (默认: 2)')
    parser.add_argument('--config2-tree-size', type=int, default=12,
                       help='配置2的spec_tree_size (默认: 12)')
    
    return parser


def create_configs_from_comparison_args(args: argparse.Namespace) -> tuple[ComparisonConfig, ComparisonConfig]:
    """从比较参数创建两个配置对象"""
    config1 = ComparisonConfig(
        spec_num_iter=args.config1_iter,
        spec_tree_size=args.config1_tree_size,
        comparison_steps=args.comparison_steps,
        config_name=f"config1_iter{args.config1_iter}_tree{args.config1_tree_size}"
    )
    
    config2 = ComparisonConfig(
        spec_num_iter=args.config2_iter,
        spec_tree_size=args.config2_tree_size,
        comparison_steps=args.comparison_steps,
        config_name=f"config2_iter{args.config2_iter}_tree{args.config2_tree_size}"
    )
    
    return config1, config2 