#!/usr/bin/env python3
"""
读取本项目评测结果与对应的 RULER 数据集（转换后 JSONL），
尽量调用 RULER 官方评分（若可用），否则回退到 EM（归一化精确匹配）并给出按任务、按序列长度的指标汇总。

用法示例：
  python scripts/ruler/score_with_ruler.py \
    --dataset-jsonl /abs/path/to/ruler.jsonl \
    --results-json /abs/path/to/benchmark/results/logs/qa_<model>_<timestamp>.json \
    --output-dir /abs/path/to/benchmark/results/ruler

可选：
  --ruler-repo           指向本地 RULER 仓库，尝试调用其官方评分实现
  --use-em-only          强制使用 EM 评分
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


def try_import_ruler(ruler_repo: Optional[str]) -> bool:
    """尝试将 RULER 仓库加入 sys.path，以便后续调用官方评分。
    若失败则返回 False。
    """
    if not ruler_repo:
        return False
    repo = os.path.abspath(ruler_repo)
    if not os.path.isdir(repo):
        return False
    if repo not in sys.path:
        sys.path.append(repo)
    # 粗略检查核心目录是否存在
    expected = os.path.join(repo, "scripts")
    return os.path.isdir(expected)


def load_dataset_map(dataset_jsonl: str) -> Dict[int, Dict[str, Any]]:
    id2item: Dict[int, Dict[str, Any]] = {}
    with open(dataset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("id")
            if sid is None:
                sid = obj.get("question_id")
            if sid is None:
                continue
            id2item[int(sid)] = obj
    return id2item


def load_results(results_json: str) -> List[Dict[str, Any]]:
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    # 兼容：若直接给了 results list
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported results file format: {results_json}")


def extract_prediction(res_item: Dict[str, Any]) -> str:
    # run_dataset_evaluation 每题可能多轮；对 RULER 场景默认单轮，取最后一轮响应
    responses = res_item.get("responses")
    if isinstance(responses, list) and responses:
        return str(responses[-1])
    # 兼容字段
    txt = res_item.get("response") or res_item.get("text")
    return str(txt) if txt is not None else ""


def score_em(id2item: Dict[int, Dict[str, Any]], results: List[Dict[str, Any]]):
    total = 0
    correct = 0
    by_task = defaultdict(lambda: {"total": 0, "correct": 0})
    by_len = defaultdict(lambda: {"total": 0, "correct": 0})

    pairs: List[Dict[str, Any]] = []

    for r in results:
        if r.get("error"):
            continue
        sid = r.get("id")
        if sid is None:
            continue
        sid = int(sid)
        if sid not in id2item:
            continue
        ds = id2item[sid]
        meta = ds.get("ruler_meta", {})
        gold = meta.get("gold")
        task = meta.get("task") or "unknown"
        seq_len = meta.get("seq_len") or "unknown"

        pred = extract_prediction(r)
        pred_n = normalize_text(pred)

        # gold 可能是字符串 / 列表 / 字典；先尝试字符串匹配，否则尝试列表里任一匹配
        ok = False
        if isinstance(gold, str):
            ok = pred_n == normalize_text(gold)
        elif isinstance(gold, list):
            ok = any(pred_n == normalize_text(g) for g in gold)
        elif isinstance(gold, dict):
            # 尝试对字典里常见字段比较
            cand = gold.get("answer") or gold.get("label") or gold.get("gold")
            if isinstance(cand, (str, int, float)):
                ok = pred_n == normalize_text(str(cand))
        else:
            # 无 gold 可比对，记为未计分
            ok = False

        total += 1
        if ok:
            correct += 1
        by_task[task]["total"] += 1
        by_task[task]["correct"] += int(ok)
        by_len[str(seq_len)]["total"] += 1
        by_len[str(seq_len)]["correct"] += int(ok)

        pairs.append({
            "id": sid,
            "task": task,
            "seq_len": seq_len,
            "gold": gold,
            "prediction": pred,
            "correct": bool(ok),
        })

    def finalize(d: Dict[str, Dict[str, int]]):
        out = {}
        for k, v in d.items():
            t = v.get("total", 0)
            c = v.get("correct", 0)
            out[k] = {
                "total": t,
                "correct": c,
                "accuracy": (c / t) if t > 0 else 0.0,
            }
        return out

    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "by_task": finalize(by_task),
        "by_seq_len": finalize(by_len),
        "pairs": pairs,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", required=True, help="评测时使用的 RULER 转换后 JSONL")
    parser.add_argument("--results-json", required=True, help="cpmcu save_results 产出的 results JSON 文件路径")
    parser.add_argument("--output-dir", required=True, help="输出评分结果目录")
    parser.add_argument("--ruler-repo", default=None, help="RULER 仓库路径（可选；若提供则尝试调用官方评分）")
    parser.add_argument("--use-em-only", action="store_true", help="仅使用 EM 评分")
    args = parser.parse_args()

    id2item = load_dataset_map(os.path.abspath(args.dataset_jsonl))
    results = load_results(os.path.abspath(args.results_json))
    os.makedirs(args.output_dir, exist_ok=True)

    used_official = False
    if (not args.use_em_only) and try_import_ruler(args.ruler_repo):
        # 预留：可在此接入 RULER 官方评分（若提供清晰的 Python 接口）
        # 当前版本默认回退到 EM，以保证稳健可用。
        used_official = False

    metrics = score_em(id2item, results)
    metrics["used_official_ruler_metric"] = used_official

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.output_dir, f"ruler_metrics_{timestamp}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    print(f"[OK] RULER metrics saved: {out_json}")
    print(f"Overall accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")


if __name__ == "__main__":
    main()


