#!/usr/bin/env python3
"""
CPM.cu Benchmark Module

Dataset loading and evaluation functionality for CPM.cu benchmark evaluation.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from .logging import logger


def load_questions(filename: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file"""
    questions = []
    with open(filename, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                obj = json.loads(line)
                questions.append(obj)
    return questions


def load_dataset(dataset_type: str, dataset_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Load dataset based on type and path"""
    
    # Determine dataset file path
    if dataset_path:
        dataset_file = dataset_path
    else:
        # Default paths after benchmark restructure
        default_paths = {
            "mtbench": "benchmark/specbench/datasets/mtbench.jsonl",
            "specbench": "benchmark/specbench/datasets/specbench.jsonl", 
            "gsm8k": "benchmark/specbench/datasets/gsm8k.jsonl",
            "qa": "benchmark/specbench/datasets/qa.jsonl",
            "wmt14": "benchmark/specbench/datasets/wmt14.jsonl",
            "rag": "benchmark/specbench/datasets/rag.jsonl",
            "summarization": "benchmark/specbench/datasets/summarization.jsonl",
            "ruler": "benchmark/ruler/ruler.jsonl",
        }
        
        if dataset_type not in default_paths:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                           f"Supported types: {list(default_paths.keys())}")
        
        dataset_file = default_paths[dataset_type]
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    # Load raw questions
    raw_questions = load_questions(dataset_file)
    
    # Parse questions based on dataset type
    questions = []
    
    if dataset_type in ["mtbench", "specbench"]:
        # MTBench/SpecBench format: preserve all turns for multi-turn conversations
        for data in raw_questions:
            turns = data.get('turns', [])
            if len(turns) >= 1:
                questions.append({
                    'id': data.get('question_id', len(questions)),
                    'question': turns[0],  # First turn (for backward compatibility)
                    'category': data.get('category', 'general'),
                    'turns': turns  # Keep all turns for multi-turn support
                })
    
    elif dataset_type == "gsm8k":
        # GSM8K format: single turn with reference answer
        for data in raw_questions:
            turns = data.get('turns', [])
            if len(turns) >= 1:
                questions.append({
                    'id': data.get('question_id', len(questions)),
                    'question': turns[0],
                    'category': data.get('category', 'math_reasoning'),
                    'reference': data.get('reference', []),
                    'turns': turns  # Keep turns even for single-turn
                })
    
    else:
        # Generic format: try to extract question from turns or other fields
        for data in raw_questions:
            turns = data.get('turns', [])
            question_text = ""
            
            if turns and len(turns) > 0:
                question_text = turns[0]
            else:
                # Fallback to other possible field names
                question_text = (data.get('question') or 
                               data.get('prompt') or 
                               data.get('text') or 
                               str(data))
            
            if question_text:
                questions.append({
                    'id': data.get('question_id') or data.get('id', len(questions)),
                    'question': question_text,
                    'category': data.get('category', 'general'),
                    'turns': turns if turns else [question_text]  # Ensure turns always exists
                })
    
    question_count = len(questions)
    logger.info(f"Loaded {question_count} questions from {dataset_file} ({dataset_type})")
    return questions, question_count


def save_results(results: List[Dict[str, Any]], output_dir: str, 
                dataset_type: str, model_name: str) -> str:
    """Save evaluation results to JSON file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model_name.replace('/', '_').replace('\\', '_')
    filename = f"{dataset_type}_{model_safe}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Calculate summary statistics
    successful = len([r for r in results if not r.get('error', False)])
    total_time = sum(r.get('timing', {}).get('total_time', 0) for r in results if not r.get('error', False))
    total_tokens = sum(r.get('tokens', {}).get('output_length', 0) for r in results if not r.get('error', False))
    
    # Calculate mean accept length for speculative decoding
    all_accept_lengths = []
    for r in results:
        if 'accept_lengths' in r and r['accept_lengths']:
            all_accept_lengths.extend(r['accept_lengths'])
    
    summary_stats = {
        'total_time': round(total_time, 2),
        'avg_time_per_question': round(total_time / successful, 2) if successful > 0 else 0,
        'total_output_tokens': total_tokens,
        'avg_tokens_per_question': round(total_tokens / successful, 2) if successful > 0 else 0,
        'throughput_tokens_per_sec': round(total_tokens / total_time, 2) if total_time > 0 else 0
    }
    
    # Add mean accept length if available
    if all_accept_lengths:
        mean_accept_length = sum(all_accept_lengths) / len(all_accept_lengths)
        summary_stats['mean_accept_length'] = round(mean_accept_length, 2)
        logger.info(f"Mean accepted tokens: {mean_accept_length:.2f}")
    
    output_data = {
        'dataset_type': dataset_type,
        'model_name': model_name,
        'timestamp': timestamp,
        'total_questions': len(results),
        'successful_questions': successful,
        'success_rate': successful / len(results) if results else 0,
        'summary_stats': summary_stats,
        'results': results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Results saved to: {filepath}")
    
    # Build summary message
    summary_parts = [
        f"{successful}/{len(results)} questions successful",
        f"{output_data['summary_stats']['throughput_tokens_per_sec']:.2f} tokens/s"
    ]
    
    # Add mean accept length to summary if available
    if 'mean_accept_length' in summary_stats:
        summary_parts.append(f"mean accept length: {summary_stats['mean_accept_length']:.2f}")
    
    logger.info(f"Summary: {', '.join(summary_parts)}")
    
    return filepath


def get_available_datasets() -> List[str]:
    """Get list of available dataset types"""
    return ["mtbench", "specbench", "gsm8k", "qa", "wmt14", "rag", "summarization", "ruler"]


def validate_dataset_exists(dataset_type: str, dataset_path: Optional[str] = None) -> bool:
    """Check if dataset file exists"""
    try:
        load_dataset(dataset_type, dataset_path)
        return True
    except (FileNotFoundError, ValueError):
        return False 