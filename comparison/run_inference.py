#!/usr/bin/env python3
"""
Unified Inference Frontend - 统一推理前端
支持CPM.cu和SGLang两种推理框架的统一接口
"""

import argparse
import sys
import os
import time
from typing import Optional

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from config import ComparisonConfig, create_comparison_config_parser, create_config_from_args


def create_unified_parser() -> argparse.ArgumentParser:
    """创建统一的参数解析器"""
    parser = create_comparison_config_parser()
    parser.description = "统一推理前端 - 支持CPM.cu和SGLang框架"
    
    # 添加框架选择参数
    parser.add_argument('--framework', '-f', 
                       choices=['cpmcu', 'sglang', 'both'],
                       default='cpmcu',
                       help='选择推理框架: cpmcu, sglang, 或 both (默认: cpmcu)')
    
    # 添加通用参数
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--save-results', action='store_true',
                       help='保存推理结果到文件')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='输出文件目录 (默认: ./outputs)')
    parser.add_argument('--compare-frameworks', action='store_true',
                       help='比较两个框架的输出结果 (需要--framework both)')
    
    return parser


def run_cpmcu_inference(config: ComparisonConfig, verbose: bool = False, 
                       save_results: bool = False, output_dir: str = './outputs') -> Optional[dict]:
    """运行CPM.cu推理"""
    if verbose:
        print("🚀 启动CPM.cu推理...")
    
    try:
        from logits import run_generation_with_config
        
        start_time = time.time()
        result = run_generation_with_config(config)
        total_time = time.time() - start_time
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"cpmcu_{config.config_name}.pkl")
            
            import pickle
            save_data = {
                "framework": "cpmcu",
                "config": config,
                "result": result,
                "total_time": total_time,
                "timestamp": time.time()
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            if verbose:
                print(f"💾 CPM.cu结果已保存到: {save_path}")
        
        return {
            "framework": "cpmcu",
            "result": result,
            "total_time": total_time,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ CPM.cu推理失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def run_sglang_inference(config: ComparisonConfig, verbose: bool = False,
                        save_results: bool = False, output_dir: str = './outputs') -> Optional[dict]:
    """运行SGLang推理"""
    if verbose:
        print("🚀 启动SGLang推理...")
    
    try:
        from sglang_inference import run_sglang_inference
        
        start_time = time.time()
        result = run_sglang_inference(config, verbose=verbose)
        total_time = time.time() - start_time
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"sglang_{config.config_name}.pkl")
            
            import pickle
            save_data = {
                "framework": "sglang",
                "config": config,
                "result": result,
                "total_time": total_time,
                "timestamp": time.time()
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            if verbose:
                print(f"💾 SGLang结果已保存到: {save_path}")
        
        return {
            "framework": "sglang",
            "result": result,
            "total_time": total_time,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ SGLang推理失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None


def compare_framework_results(cpmcu_result: dict, sglang_result: dict, verbose: bool = False):
    """比较两个框架的结果"""
    if not cpmcu_result or not sglang_result:
        print("⚠️  无法比较：一个或多个框架推理失败")
        return
    
    print("\n" + "="*60)
    print("📊 框架比较结果")
    print("="*60)
    
    # 基本性能比较
    cpmcu_time = cpmcu_result.get('total_time', 0)
    sglang_time = sglang_result.get('total_time', 0)
    
    print(f"⏱️  推理耗时:")
    print(f"  CPM.cu:  {cpmcu_time:.2f}s")
    print(f"  SGLang:  {sglang_time:.2f}s")
    if cpmcu_time > 0 and sglang_time > 0:
        ratio = cpmcu_time / sglang_time
        faster_framework = "SGLang" if ratio > 1 else "CPM.cu"
        print(f"  🏆 {faster_framework} 更快 ({abs(ratio-1)*100:.1f}%)")
    
    # Token数量比较
    cpmcu_tokens = getattr(cpmcu_result.get('result'), 'total_tokens', 0)
    sglang_tokens = getattr(sglang_result.get('result'), 'total_tokens', 0)
    
    print(f"\n🎯 生成Token数:")
    print(f"  CPM.cu:  {cpmcu_tokens}")
    print(f"  SGLang:  {sglang_tokens}")
    
    # 吞吐量比较
    cpmcu_throughput = getattr(cpmcu_result.get('result'), 'throughput', 0)
    sglang_throughput = getattr(sglang_result.get('result'), 'throughput', 0)
    
    print(f"\n🚀 吞吐量 (tok/s):")
    print(f"  CPM.cu:  {cpmcu_throughput:.1f}")
    print(f"  SGLang:  {sglang_throughput:.1f}")
    
    # Logits数据比较
    cpmcu_logits = getattr(cpmcu_result.get('result'), 'logits_data', [])
    sglang_logits = getattr(sglang_result.get('result'), 'logits_data', [])
    
    print(f"\n📊 Logits数据:")
    print(f"  CPM.cu:  {len(cpmcu_logits)} 步骤")
    print(f"  SGLang:  {len(sglang_logits)} 步骤")
    
    if verbose and cpmcu_logits and sglang_logits:
        print(f"\n🔍 详细数据对比 (前3步):")
        for i in range(min(3, len(cpmcu_logits), len(sglang_logits))):
            cpmcu_step = cpmcu_logits[i]
            sglang_step = sglang_logits[i]
            print(f"  步骤 {i}:")
            print(f"    CPM.cu:  Token {cpmcu_step.get('token_id', 'N/A')}, Logprob {cpmcu_step.get('logprob', 'N/A')}")
            print(f"    SGLang:  Token {sglang_step.get('token_id', 'N/A')}, Logprob {sglang_step.get('logprob', 'N/A')}")
    
    print("="*60)


def main():
    """主函数"""
    parser = create_unified_parser()
    args = parser.parse_args()
    
    # 创建配置
    config = create_config_from_args(args)
    
    print("="*60)
    print("🎯 统一推理前端")
    print("="*60)
    print(f"配置名称: {config.config_name}")
    print(f"选择框架: {args.framework}")
    print(f"投机参数: iter={config.spec_num_iter}, tree={config.spec_tree_size}, topk={config.get_topk_per_iter()}")
    print(f"比较步数: {config.comparison_steps}")
    print("="*60)
    
    results = {}
    
    if args.framework in ['cpmcu', 'both']:
        print("\n🔧 运行CPM.cu推理...")
        cpmcu_result = run_cpmcu_inference(
            config, 
            verbose=args.verbose,
            save_results=args.save_results,
            output_dir=args.output_dir
        )
        if cpmcu_result:
            results['cpmcu'] = cpmcu_result
            print(f"✅ CPM.cu推理完成: {cpmcu_result['total_time']:.2f}s")
        else:
            print("❌ CPM.cu推理失败")
    
    if args.framework in ['sglang', 'both']:
        print("\n🔧 运行SGLang推理...")
        sglang_result = run_sglang_inference(
            config,
            verbose=args.verbose,
            save_results=args.save_results,
            output_dir=args.output_dir
        )
        if sglang_result:
            results['sglang'] = sglang_result
            print(f"✅ SGLang推理完成: {sglang_result['total_time']:.2f}s")
        else:
            print("❌ SGLang推理失败")
    
    # 比较结果
    if args.framework == 'both' and 'cpmcu' in results and 'sglang' in results:
        if args.compare_frameworks:
            compare_framework_results(
                results['cpmcu'], 
                results['sglang'], 
                verbose=args.verbose
            )
    
    # 总结
    print(f"\n🎉 推理任务完成!")
    if results:
        print(f"📈 成功运行的框架: {', '.join(results.keys())}")
        if args.save_results:
            print(f"💾 结果已保存到: {args.output_dir}")
    else:
        print("❌ 所有框架推理都失败了")
        sys.exit(1)


if __name__ == "__main__":
    main() 