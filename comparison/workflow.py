#!/usr/bin/env python3
"""
Logits Comparison Workflow Script
实现 runconfig1 -> runconfig2 -> analyze 的完整工作流程

这个脚本复用已有的模块，按照用户要求的流程执行：
1. 运行Config1 (spec_num_iter=5, spec_tree_size=32)
2. 运行Config2 (spec_num_iter=2, spec_tree_size=12)  
3. 分析比较两个配置的结果

目标是复现 logits_comparison_direct.py 的比较结果。
"""

import os
import sys
import argparse
import torch
from run_single_config import run_config_by_name
from analyze_saved_data import compare_configs


def create_workflow_parser():
    """创建工作流参数解析器"""
    parser = argparse.ArgumentParser(description='Logits比较完整工作流程')
    
    parser.add_argument('--comparison-steps', type=int, default=20,
                       help='比较步骤数 (默认: 20)')
    parser.add_argument('--skip-config1', action='store_true',
                       help='跳过Config1的运行（如果数据文件已存在）')
    parser.add_argument('--skip-config2', action='store_true',
                       help='跳过Config2的运行（如果数据文件已存在）')
    parser.add_argument('--only-analyze', action='store_true',
                       help='只执行分析步骤（假设数据文件已存在）')
    
    return parser


def check_existing_files():
    """检查已存在的数据文件"""
    config1_file = "logits_capture_config1_iter5_tree32.pkl"
    config2_file = "logits_capture_config2_iter2_tree12.pkl"
    
    config1_exists = os.path.exists(config1_file)
    config2_exists = os.path.exists(config2_file)
    
    return config1_exists, config2_exists, config1_file, config2_file


def main():
    """主工作流程函数"""
    
    parser = create_workflow_parser()
    args = parser.parse_args()
    
    print("🚀 开始Logits比较完整工作流程")
    print("=" * 60)
    print("流程: runconfig1 -> runconfig2 -> analyze")
    print("目标: 复现 logits_comparison_direct.py 的比较结果")
    print("=" * 60)
    
    # 检查已存在的文件
    config1_exists, config2_exists, config1_file, config2_file = check_existing_files()
    
    if config1_exists:
        print(f"📁 发现已存在的Config1数据: {config1_file}")
    if config2_exists:
        print(f"📁 发现已存在的Config2数据: {config2_file}")
    
    # 步骤1: 运行Config1
    if not args.only_analyze and (not args.skip_config1 or not config1_exists):
        print(f"\n🔧 步骤1: 运行Config1")
        print("-" * 40)
        print("配置: spec_num_iter=5, spec_tree_size=32")
        
        try:
            result = run_config_by_name('config1', args.comparison_steps)
            if result and result[0]:
                print(f"✅ Config1运行成功: {result[0]}")
                config1_file = result[0]
            else:
                print("❌ Config1运行失败")
                return
        except Exception as e:
            print(f"❌ Config1运行出错: {e}")
            return
        
        # 清理内存
        torch.cuda.empty_cache()
        print("🧹 GPU内存已清理")
        
    else:
        print(f"\n⏭️  跳过Config1运行")
        if not config1_exists:
            print(f"❌ Config1数据文件不存在: {config1_file}")
            return
    
    # 步骤2: 运行Config2
    if not args.only_analyze and (not args.skip_config2 or not config2_exists):
        print(f"\n🔧 步骤2: 运行Config2")
        print("-" * 40)
        print("配置: spec_num_iter=2, spec_tree_size=12")
        
        try:
            result = run_config_by_name('config2', args.comparison_steps)
            if result and result[0]:
                print(f"✅ Config2运行成功: {result[0]}")
                config2_file = result[0]
            else:
                print("❌ Config2运行失败")
                return
        except Exception as e:
            print(f"❌ Config2运行出错: {e}")
            return
        
        # 清理内存
        torch.cuda.empty_cache()
        print("🧹 GPU内存已清理")
        
    else:
        print(f"\n⏭️  跳过Config2运行")
        if not config2_exists:
            print(f"❌ Config2数据文件不存在: {config2_file}")
            return
    
    # 步骤3: 分析比较
    print(f"\n🔍 步骤3: 分析比较")
    print("-" * 40)
    print(f"比较文件:")
    print(f"  Config1: {config1_file}")
    print(f"  Config2: {config2_file}")
    print(f"  比较步骤: {args.comparison_steps}")
    
    try:
        compare_configs(config1_file, config2_file, args.comparison_steps)
        print(f"✅ 比较分析完成!")
        
    except Exception as e:
        print(f"❌ 比较分析出错: {e}")
        return
    
    # 完成
    print(f"\n🎉 完整工作流程执行完成!")
    print("=" * 60)
    print("📄 生成的文件:")
    print(f"  - {config1_file}")
    print(f"  - {config2_file}")
    print(f"  - detailed_logits_comparison_results.json")
    print("\n💡 这个结果应该与 logits_comparison_direct.py 的输出一致")


def quick_run(comparison_steps=20):
    """便捷函数：快速运行完整工作流程"""
    import sys
    
    # 模拟命令行参数
    original_argv = sys.argv.copy()
    sys.argv = ['workflow.py', '--comparison-steps', str(comparison_steps)]
    
    try:
        main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main() 