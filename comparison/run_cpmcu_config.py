#!/usr/bin/env python3
"""
Single Config Logits Capture Script
Runs a single configuration and captures logits

支持预设配置和自定义配置：
- config1: spec_num_iter=5, spec_tree_size=32
- config2: spec_num_iter=2, spec_tree_size=12
"""

import torch
import argparse
from config import ComparisonConfig
from logits import run_generation_with_config
from analysis import analyze_single_capture


def create_enhanced_parser():
    """创建增强的参数解析器，支持预设配置"""
    parser = argparse.ArgumentParser(description='运行单配置logits捕获')
    
    # 预设配置选项
    parser.add_argument('--preset', choices=['config1', 'config2'], 
                       help='使用预设配置: config1(iter5_tree32) 或 config2(iter2_tree12)')
    
    # 自定义配置选项
    parser.add_argument('--spec-num-iter', type=int, default=5,
                       help='投机迭代次数 (默认: 5)')
    parser.add_argument('--spec-tree-size', type=int, default=32,
                       help='投机树大小 (默认: 32)')
    parser.add_argument('--comparison-steps', type=int, default=20,
                       help='生成的token数量 (默认: 20)')
    parser.add_argument('--config-name', type=str, default=None,
                       help='配置名称 (可选，自动生成如果未提供)')
    
    # 模型路径
    parser.add_argument('--model-path', type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct",
                       help='主模型路径')
    parser.add_argument('--draft-model-path', type=str, default="jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B",
                       help='草稿模型路径')
    parser.add_argument('--prompt-file', type=str, default="prompt_small.txt",
                       help='提示文件路径')
    
    # 其他选项
    parser.add_argument('--memory-limit', type=float, default=0.85,
                       help='内存限制 (默认: 0.7)')
    parser.add_argument('--chunk-length', type=int, default=1024,
                       help='块长度 (默认: 1024)')
    parser.add_argument('--cuda-graph', action='store_true',
                       help='启用CUDA图')
    parser.add_argument('--minicpm4-yarn', action='store_true', default=True,
                       help='启用MiniCPM4 YARN配置')
    parser.add_argument('--use-chat-template', action='store_true',
                       help='使用聊天模板')
    
    return parser


def apply_preset_config(args):
    """应用预设配置"""
    if args.preset == 'config1':
        args.spec_num_iter = 5
        args.spec_tree_size = 32
        if args.config_name is None:
            args.config_name = 'config1_iter5_tree32'
        print(f"📋 使用预设Config1: spec_num_iter=5, spec_tree_size=32")
    elif args.preset == 'config2':
        args.spec_num_iter = 2
        args.spec_tree_size = 12
        if args.config_name is None:
            args.config_name = 'config2_iter2_tree12'
        print(f"📋 使用预设Config2: spec_num_iter=2, spec_tree_size=12")
    else:
        # 自定义配置
        if args.config_name is None:
            args.config_name = f"custom_iter{args.spec_num_iter}_tree{args.spec_tree_size}"
        print(f"📋 使用自定义配置")


def main():
    """主函数：运行单个配置的logits捕获"""
    
    # 解析命令行参数
    parser = create_enhanced_parser()
    args = parser.parse_args()
    
    print(f"🚀 开始单配置Logits捕获...")
    
    # 应用预设配置
    apply_preset_config(args)
    
    # 创建配置对象
    config = ComparisonConfig(
        spec_num_iter=args.spec_num_iter,
        spec_tree_size=args.spec_tree_size,
        comparison_steps=args.comparison_steps,
        config_name=args.config_name,
        model_path=args.model_path,
        draft_model_path=args.draft_model_path,
        prompt_file=args.prompt_file,
        memory_limit=args.memory_limit,
        chunk_length=args.chunk_length,
        cuda_graph=args.cuda_graph,
        minicpm4_yarn=args.minicpm4_yarn,
        use_chat_template=args.use_chat_template
    )
    
    print(f"\n📊 配置信息:")
    print(f"  配置名称: {config.config_name}")
    print(f"  spec_num_iter: {config.spec_num_iter}")
    print(f"  spec_tree_size: {config.spec_tree_size}")
    print(f"  topk_per_iter: {config.get_topk_per_iter()}")
    print(f"  比较步骤数: {config.comparison_steps}")
    
    # 运行生成并捕获logits
    print(f"\n" + "="*60)
    print(f"运行配置: {config.config_name}")
    print("="*60)
    
    capture = run_generation_with_config(config)
    
    if capture:
        print(f"\n✅ 配置成功完成!")
        print(f"   捕获步骤: {len(capture.captured_logits)}")
        
        # 生成的文件名
        output_file = f"logits_capture_{config.config_name}.pkl"
        print(f"   输出文件: {output_file}")
        
        # 简单分析（不打印详细信息）
        analysis = analyze_single_capture(capture)
        
        print(f"\n📄 数据已保存，可用于后续分析")
        
        # 返回输出文件名（如果作为模块调用）
        return output_file, config.config_name
        
    else:
        print("❌ 配置运行失败")
        return None, None
    
    # 清理内存
    torch.cuda.empty_cache()
    print(f"\n🧹 内存已清理")


def run_config_by_name(preset_name, comparison_steps=20):
    """通过预设名称运行配置的便捷函数"""
    import sys
    
    # 模拟命令行参数
    original_argv = sys.argv.copy()
    sys.argv = ['run_single_config.py', '--preset', preset_name, '--comparison-steps', str(comparison_steps)]
    
    try:
        result = main()
        return result
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main() 