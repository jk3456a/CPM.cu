#!/usr/bin/env python3
"""
Analyze Saved Logits Data Script
专门用于分析两个配置（config1 vs config2）的logits比较

支持多种使用方式：
1. 自动查找config1和config2的数据文件
2. 手动指定两个数据文件
3. 复现logits_comparison_direct.py的比较结果
"""

import argparse
import os
import glob
from logits import load_logits_data
from analysis import compare_logits_data


def find_config_files(directory="."):
    """自动查找config1和config2的数据文件"""
    config1_pattern = os.path.join(directory, "logits_capture_config1_*.pkl")
    config2_pattern = os.path.join(directory, "logits_capture_config2_*.pkl")
    
    config1_files = glob.glob(config1_pattern)
    config2_files = glob.glob(config2_pattern)
    
    return config1_files, config2_files


def create_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(description='分析两个配置的logits数据比较')
    
    # 模式选择
    parser.add_argument('--mode', choices=['auto', 'manual'], default='auto',
                       help='模式: auto(自动查找) 或 manual(手动指定)')
    
    # 自动模式选项
    parser.add_argument('--directory', type=str, default='.',
                       help='搜索数据文件的目录 (默认: 当前目录)')
    
    # 手动模式选项
    parser.add_argument('--config1-file', type=str,
                       help='Config1的数据文件路径')
    parser.add_argument('--config2-file', type=str,
                       help='Config2的数据文件路径')
    
    # 比较选项
    parser.add_argument('--comparison-steps', type=int, default=20,
                       help='比较步骤数 (默认: 20)')
    
    return parser


def main():
    """主函数：分析两个配置的logits比较"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    print(f"🚀 开始Config1 vs Config2 Logits比较分析...")
    
    config1_file = None
    config2_file = None
    
    if args.mode == 'auto':
        # 自动查找配置文件
        print(f"📁 在目录 {args.directory} 中自动搜索配置文件...")
        
        config1_files, config2_files = find_config_files(args.directory)
        
        if not config1_files:
            print("❌ 未找到config1的数据文件 (logits_capture_config1_*.pkl)")
            return
        if not config2_files:
            print("❌ 未找到config2的数据文件 (logits_capture_config2_*.pkl)")
            return
        
        # 使用最新的文件
        config1_file = max(config1_files, key=os.path.getmtime)
        config2_file = max(config2_files, key=os.path.getmtime)
        
        print(f"✅ 自动找到配置文件:")
        print(f"   Config1: {config1_file}")
        print(f"   Config2: {config2_file}")
        
    else:
        # 手动指定文件
        if not args.config1_file or not args.config2_file:
            print("❌ 手动模式需要指定 --config1-file 和 --config2-file")
            return
        
        config1_file = args.config1_file
        config2_file = args.config2_file
        
        print(f"📋 使用手动指定的配置文件:")
        print(f"   Config1: {config1_file}")
        print(f"   Config2: {config2_file}")
    
    # 检查文件存在性
    if not os.path.exists(config1_file):
        print(f"❌ Config1文件不存在: {config1_file}")
        return
    if not os.path.exists(config2_file):
        print(f"❌ Config2文件不存在: {config2_file}")
        return
    
    # 加载数据
    print(f"\n📥 加载数据文件...")
    
    print(f"   加载Config1: {config1_file}")
    capture1 = load_logits_data(config1_file)
    if not capture1:
        print(f"❌ Config1数据加载失败")
        return
    
    print(f"   加载Config2: {config2_file}")
    capture2 = load_logits_data(config2_file)
    if not capture2:
        print(f"❌ Config2数据加载失败")
        return
    
    print(f"✅ 数据加载完成")
    print(f"   Config1 ({capture1.config_name}): {len(capture1.captured_logits)} 步骤")
    print(f"   Config2 ({capture2.config_name}): {len(capture2.captured_logits)} 步骤")
    
    # 进行比较分析
    print(f"\n🔍 开始详细比较分析...")
    print(f"   比较步骤数: {args.comparison_steps}")
    
    compare_logits_data(capture1, capture2, args.comparison_steps)
    
    print(f"\n✅ 比较分析完成!")
    print(f"   详细结果已保存到: detailed_logits_comparison_results.json")
    print(f"\n💡 这个结果与 logits_comparison_direct.py 的输出应该是一致的")


def compare_configs(config1_file=None, config2_file=None, comparison_steps=20, directory="."):
    """便捷函数：比较两个配置文件"""
    import sys
    
    # 构建命令行参数
    argv = ['analyze_saved_data.py', '--comparison-steps', str(comparison_steps)]
    
    if config1_file and config2_file:
        # 手动模式
        argv.extend(['--mode', 'manual', '--config1-file', config1_file, '--config2-file', config2_file])
    else:
        # 自动模式
        argv.extend(['--mode', 'auto', '--directory', directory])
    
    # 模拟命令行调用
    original_argv = sys.argv.copy()
    sys.argv = argv
    
    try:
        main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main() 