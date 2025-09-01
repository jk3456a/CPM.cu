#!/usr/bin/env python3
"""
生成 RULER 样本并转换为本项目可直接评测的 JSONL 格式（最小改动接入）。

用法示例：
  python scripts/ruler/generate_and_convert.py \
    --ruler-repo /abs/path/to/third_party/RULER \
    --root-dir /abs/path/to/ruler_workdir \
    --seq-lens 4096 8192 16384 \
    --tasks niah variable_tracking qa \
    --output-jsonl /abs/path/to/ruler.jsonl

说明：
- 若 --skip-generate 打开，则跳过调用 RULER 的生成流程，只做转换；
- 转换产出的 JSONL 每行至少包含：id、question、category、turns，并保留 ruler 元信息：task、seq_len、gold。
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Dict, Any


def run_cmd(cmd: List[str]):
    print("[CMD]", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res.stdout


def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def generate_with_ruler(ruler_repo: str, root_dir: str, seq_lens: List[int], tasks: List[str], seed: int):
    """调用 RULER 官方脚本生成样本到 root_dir 下，保持原始结构。"""
    scripts_dir = os.path.join(ruler_repo, "scripts")
    if not os.path.isdir(scripts_dir):
        raise FileNotFoundError(f"RULER scripts not found: {scripts_dir}")

    env = os.environ.copy()
    env["ROOT_DIR"] = root_dir
    ensure_dirs(root_dir)

    # 这里仅示意性地调用数据准备入口；具体命令需与 RULER 的脚本保持一致
    # 常见流程：bash run.sh <MODEL> synthetic 但我们只需要样本，可直接调用 data 生成脚本
    # 若你的 RULER 分支提供独立数据生成入口，请在此替换为相应脚本。
    config_tasks = os.path.join(scripts_dir, "config_tasks.sh")
    if not os.path.exists(config_tasks):
        print("[WARN] config_tasks.sh not found, skip generation and assume data already present.")
        return

    # 为不同 seq_len 与 task 生成数据（最小可用接口：通过环境变量控制）
    for L in seq_lens:
        for task in tasks:
            cmd = [
                "bash", "-lc",
                f"cd {scripts_dir} && SEQ_LEN={L} TASK_NAME={task} SEED={seed} ROOT_DIR={root_dir} bash data/synthetic/generate_one.sh"
            ]
            try:
                run_cmd(cmd)
            except Exception as e:
                print(f"[WARN] generation failed for task={task}, L={L}: {e}. Continue...")


def discover_samples(root_dir: str) -> List[Dict[str, Any]]:
    """在 ROOT_DIR 下发现 RULER 生成的样本与标注文件。
    该函数需要按你的 RULER 数据落盘结构调整。目前实现：查找 *.jsonl / *.json 的样本集。
    """
    candidates = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.endswith(".jsonl") or fn.endswith(".json"):
                candidates.append(os.path.join(base, fn))
    return [{"path": p} for p in sorted(candidates)]


def to_project_jsonl(sample_paths: List[str], output_jsonl: str):
    """将 RULER 样本统一转换为本项目通用 JSONL。
    约定：
      - 每条样本包含文本 prompt（字段名可能是 prompt/input/text 之一），gold 答案（answer/label 等），以及可用的 task/seq_len 元信息。
      - 转换后结构：
         {
           "question_id": <int>,
           "category": "ruler",
           "turns": ["<prompt>"],
           "question": "<prompt>",
           "ruler_meta": {"task": ..., "seq_len": ..., "gold": ...}
         }
    """
    cnt = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for path in sample_paths:
            try:
                if path.endswith(".jsonl"):
                    with open(path, "r", encoding="utf-8") as fin:
                        for line in fin:
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            prompt = obj.get("prompt") or obj.get("input") or obj.get("text") or obj.get("question")
                            if not isinstance(prompt, str):
                                continue
                            gold = obj.get("gold") or obj.get("answer") or obj.get("label")
                            task = obj.get("task") or obj.get("task_name") or "synthetic"
                            seq_len = obj.get("seq_len") or obj.get("sequence_length")
                            rec = {
                                "question_id": cnt,
                                "id": cnt,
                                "category": "ruler",
                                "turns": [prompt],
                                "question": prompt,
                                "ruler_meta": {
                                    "source": os.path.basename(path),
                                    "task": task,
                                    "seq_len": seq_len,
                                    "gold": gold,
                                },
                            }
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            cnt += 1
                else:
                    with open(path, "r", encoding="utf-8") as fin:
                        data = json.load(fin)
                    # data 可能是数组或对象；尽力解析
                    if isinstance(data, list):
                        iterable = data
                    elif isinstance(data, dict):
                        iterable = data.get("data") or data.get("samples") or []
                    else:
                        iterable = []
                    for obj in iterable:
                        prompt = obj.get("prompt") or obj.get("input") or obj.get("text") or obj.get("question")
                        if not isinstance(prompt, str):
                            continue
                        gold = obj.get("gold") or obj.get("answer") or obj.get("label")
                        task = obj.get("task") or obj.get("task_name") or "synthetic"
                        seq_len = obj.get("seq_len") or obj.get("sequence_length")
                        rec = {
                            "question_id": cnt,
                            "id": cnt,
                            "category": "ruler",
                            "turns": [prompt],
                            "question": prompt,
                            "ruler_meta": {
                                "source": os.path.basename(path),
                                "task": task,
                                "seq_len": seq_len,
                                "gold": gold,
                            },
                        }
                        with open(output_jsonl, "a", encoding="utf-8") as fout2:
                            fout2.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        cnt += 1
            except Exception as e:
                print(f"[WARN] skip file {path}: {e}")
    print(f"[OK] wrote {cnt} samples to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-repo", required=True, help="RULER 仓库路径（可为子模块或外部路径）")
    parser.add_argument("--root-dir", required=True, help="RULER 生成与工作目录 ROOT_DIR")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[4096], help="需要生成的序列长度列表")
    parser.add_argument("--tasks", nargs="+", default=["niah", "variable_tracking", "qa"], help="任务名列表")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-jsonl", required=True, help="转换输出的 JSONL 文件路径")
    parser.add_argument("--skip-generate", action="store_true", help="仅转换，不执行 RULER 数据生成")
    args = parser.parse_args()

    ruler_repo = os.path.abspath(args.ruler_repo)
    root_dir = os.path.abspath(args.root_dir)
    output_jsonl = os.path.abspath(args.output_jsonl)

    if not args.skip_generate:
        generate_with_ruler(ruler_repo, root_dir, args.seq_lens, args.tasks, args.seed)

    files = discover_samples(root_dir)
    sample_paths = [x["path"] for x in files]
    if not sample_paths:
        print(f"[WARN] no samples discovered under {root_dir}. Did you generate data?")
    ensure_dirs(os.path.dirname(output_jsonl))
    to_project_jsonl(sample_paths, output_jsonl)


if __name__ == "__main__":
    main()


