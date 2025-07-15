#!/usr/bin/env python3
"""
Analysis module for logits comparison
Handles comparison and analysis of captured logits data
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from logits import LogitsCapture


def compare_logits_data(capture1: LogitsCapture, capture2: LogitsCapture, comparison_steps: int = 20):
    """Compare captured logits data"""
    
    print(f"\n{'='*80}")
    print(f"详细LOGITS比较: {capture1.config_name} vs {capture2.config_name}")
    print(f"{'='*80}")
    
    logits1 = capture1.captured_logits
    logits2 = capture2.captured_logits
    tokens1 = capture1.captured_tokens
    tokens2 = capture2.captured_tokens
    
    print(f"{capture1.config_name}: {len(logits1)} 步骤")
    print(f"{capture2.config_name}: {len(logits2)} 步骤")
    
    if len(logits1) == 0 or len(logits2) == 0:
        print("⚠️  一个或两个捕获都为空。无法比较。")
        return
    
    min_steps = min(len(logits1), len(logits2))
    comparison_results = []
    
    # Use the smaller of min_steps and comparison_steps for comparison
    steps_to_show = min(min_steps, comparison_steps)
    print(f"\n详细的逐步比较 (前 {steps_to_show} 步):")
    
    for i in range(steps_to_show):
        step_id1, logits_data1, step_type1 = logits1[i]
        step_id2, logits_data2, step_type2 = logits2[i]
        
        token_id1, token1 = tokens1[i]
        token_id2, token2 = tokens2[i]
        
        # Calculate metrics
        mse = np.mean((logits_data1 - logits_data2) ** 2)
        cos_sim = np.dot(logits_data1, logits_data2) / (np.linalg.norm(logits_data1) * np.linalg.norm(logits_data2))
        
        # Top-5 tokens for each config
        top5_indices1 = np.argsort(logits_data1)[-5:][::-1]
        top5_indices2 = np.argsort(logits_data2)[-5:][::-1]
        top5_values1 = logits_data1[top5_indices1]
        top5_values2 = logits_data2[top5_indices2]
        
        comparison_result = {
            'step_index': i,
            'step_id1': step_id1,
            'step_id2': step_id2,
            'step_type1': step_type1,
            'step_type2': step_type2,
            'token1': token1,
            'token2': token2,
            'mse': mse,
            'cosine_similarity': cos_sim,
            'top5_tokens1': top5_indices1.tolist(),
            'top5_values1': top5_values1.tolist(),
            'top5_tokens2': top5_indices2.tolist(),
            'top5_values2': top5_values2.tolist()
        }
        
        comparison_results.append(comparison_result)
        
        print(f"\n" + "="*60)
        print(f"步骤 {i+1}:")
        print(f"  {capture1.config_name}: 步骤ID {step_id1}, 类型 {step_type1}, Token {token1}")
        print(f"  {capture2.config_name}: 步骤ID {step_id2}, 类型 {step_type2}, Token {token2}")
        print(f"  MSE: {mse:.8f}")
        print(f"  余弦相似度: {cos_sim:.8f}")
        
        print(f"  Top-5 logits {capture1.config_name}:")
        for j, (idx, val) in enumerate(zip(top5_indices1, top5_values1)):
            print(f"    {j+1}. Token {idx}: {val:.6f}")
        
        print(f"  Top-5 logits {capture2.config_name}:")
        for j, (idx, val) in enumerate(zip(top5_indices2, top5_values2)):
            print(f"    {j+1}. Token {idx}: {val:.6f}")
        
        if token1 != token2:
            print(f"  ⚠️  TOKEN不匹配: {token1} vs {token2}")
        
        if mse > 1e-6:
            print(f"  ⚠️  大MSE差异: {mse:.8f}")
    
    # Summary statistics
    mse_values = [r['mse'] for r in comparison_results]
    cos_sim_values = [r['cosine_similarity'] for r in comparison_results]
    
    print(f"\n{'='*80}")
    print(f"摘要统计")
    print(f"{'='*80}")
    print(f"比较的总步骤: {min_steps}")
    print(f"显示的步骤: {len(comparison_results)}")
    print(f"平均MSE: {np.mean(mse_values):.8f}")
    print(f"最大MSE: {np.max(mse_values):.8f}")
    print(f"最小MSE: {np.min(mse_values):.8f}")
    print(f"平均余弦相似度: {np.mean(cos_sim_values):.8f}")
    print(f"最小余弦相似度: {np.min(cos_sim_values):.8f}")
    
    token_mismatches = sum(1 for r in comparison_results if r['token1'] != r['token2'])
    print(f"Token序列不匹配: {token_mismatches}/{len(comparison_results)}")
    
    # Save comparison results
    comparison_summary = {
        'config1': capture1.config_name,
        'config2': capture2.config_name,
        'steps_compared': min_steps,
        'comparison_results': [
            {
                'step_index': r['step_index'],
                'step_id1': r['step_id1'],
                'step_id2': r['step_id2'],
                'step_type1': r['step_type1'],
                'step_type2': r['step_type2'],
                'token1': r['token1'],
                'token2': r['token2'],
                'mse': float(r['mse']),
                'cosine_similarity': float(r['cosine_similarity']) if not np.isnan(r['cosine_similarity']) else None,
                'top5_tokens1': r['top5_tokens1'],
                'top5_values1': [float(v) for v in r['top5_values1']],
                'top5_tokens2': r['top5_tokens2'],
                'top5_values2': [float(v) for v in r['top5_values2']]
            } for r in comparison_results
        ],
        'summary_stats': {
            'avg_mse': float(np.mean(mse_values)),
            'max_mse': float(np.max(mse_values)),
            'min_mse': float(np.min(mse_values)),
            'avg_cosine_similarity': float(np.nanmean(cos_sim_values)) if not np.all(np.isnan(cos_sim_values)) else None,
            'min_cosine_similarity': float(np.nanmin(cos_sim_values)) if not np.all(np.isnan(cos_sim_values)) else None,
            'token_mismatches': token_mismatches
        }
    }
    
    with open('detailed_logits_comparison_results.json', 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    print(f"\n详细比较结果保存到 detailed_logits_comparison_results.json")


def analyze_single_capture(capture: LogitsCapture) -> Dict[str, Any]:
    """Analyze a single LogitsCapture instance"""
    
    if not capture.captured_logits:
        print(f"⚠️  {capture.config_name} 没有捕获的logits数据")
        return {}
    
    print(f"\n{'='*60}")
    print(f"分析 {capture.config_name}")
    print(f"{'='*60}")
    
    logits_data = [logits for _, logits, _ in capture.captured_logits]
    tokens = [token for _, token in capture.captured_tokens]
    
    # Calculate statistics
    logits_array = np.array(logits_data)
    
    analysis = {
        'config_name': capture.config_name,
        'total_steps': len(capture.captured_logits),
        'logits_shape': logits_array.shape,
        'logits_stats': {
            'mean': float(np.mean(logits_array)),
            'std': float(np.std(logits_array)),
            'min': float(np.min(logits_array)),
            'max': float(np.max(logits_array))
        },
        'token_stats': {
            'unique_tokens': len(set(tokens)),
            'total_tokens': len(tokens),
            'token_range': {
                'min': int(min(tokens)) if tokens else 0,
                'max': int(max(tokens)) if tokens else 0
            }
        }
    }
    
    print(f"总步骤: {analysis['total_steps']}")
    print(f"Logits形状: {analysis['logits_shape']}")
    print(f"Logits统计: 均值={analysis['logits_stats']['mean']:.4f}, 标准差={analysis['logits_stats']['std']:.4f}")
    print(f"Token统计: 总数={analysis['token_stats']['total_tokens']}, 唯一={analysis['token_stats']['unique_tokens']}")
    
    return analysis


def compare_multiple_captures(captures: List[LogitsCapture], comparison_steps: int = 20):
    """Compare multiple LogitsCapture instances"""
    
    if len(captures) < 2:
        print("⚠️  至少需要两个captures进行比较")
        return
    
    print(f"\n{'='*80}")
    print(f"多配置LOGITS比较")
    print(f"{'='*80}")
    
    # Analyze each capture individually
    analyses = []
    for capture in captures:
        analysis = analyze_single_capture(capture)
        analyses.append(analysis)
    
    # Pairwise comparisons
    print(f"\n{'='*60}")
    print(f"成对比较")
    print(f"{'='*60}")
    
    for i in range(len(captures)):
        for j in range(i + 1, len(captures)):
            print(f"\n比较 {captures[i].config_name} 与 {captures[j].config_name}:")
            compare_logits_data(captures[i], captures[j], comparison_steps)
    
    return analyses


def generate_comparison_report(captures: List[LogitsCapture], output_file: str = "logits_comparison_report.json"):
    """Generate a comprehensive comparison report"""
    
    if not captures:
        print("⚠️  没有captures可生成报告")
        return
    
    report = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'total_configs': len(captures),
        'config_names': [capture.config_name for capture in captures],
        'individual_analyses': [],
        'pairwise_comparisons': []
    }
    
    # Individual analyses
    for capture in captures:
        analysis = analyze_single_capture(capture)
        report['individual_analyses'].append(analysis)
    
    # Pairwise comparisons summary
    for i in range(len(captures)):
        for j in range(i + 1, len(captures)):
            if captures[i].captured_logits and captures[j].captured_logits:
                min_steps = min(len(captures[i].captured_logits), len(captures[j].captured_logits))
                
                # Quick MSE calculation for first few steps
                quick_comparison_steps = min(10, min_steps)
                mse_values = []
                
                for k in range(quick_comparison_steps):
                    logits1 = captures[i].captured_logits[k][1]
                    logits2 = captures[j].captured_logits[k][1]
                    mse = np.mean((logits1 - logits2) ** 2)
                    mse_values.append(float(mse))
                
                comparison_summary = {
                    'config1': captures[i].config_name,
                    'config2': captures[j].config_name,
                    'min_steps': min_steps,
                    'quick_comparison_steps': quick_comparison_steps,
                    'avg_mse': float(np.mean(mse_values)) if mse_values else None,
                    'max_mse': float(np.max(mse_values)) if mse_values else None
                }
                
                report['pairwise_comparisons'].append(comparison_summary)
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n比较报告保存到 {output_file}")
    return report 