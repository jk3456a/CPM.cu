## RULER 最小改动集成

本目录提供将 [RULER](https://github.com/NVIDIA/RULER/tree/main) 以“零侵入/最小改动”的方式接入到本项目评测流程的适配脚本：

- `generate_and_convert.py`: 调用（或跳过）RULER 的数据生成，并将其样本转换为本项目可直接评测的 JSONL。
- `score_with_ruler.py`: 读取本项目评测结果并进行指标计算；优先预留官方评分接口，当前默认回退到 EM（精确匹配）以确保可用性。

### 依赖与准备
- 建议将 RULER 仓库作为子模块放置至 `third_party/RULER/`，或使用本地任意路径并在命令中传入 `--ruler-repo`。
- 生成的数据与中间结果建议统一放在 `--root-dir` 指定的目录。

### 1) 生成与转换
```bash
python scripts/ruler/generate_and_convert.py \
  --ruler-repo /abs/path/to/third_party/RULER \
  --root-dir /abs/path/to/ruler_workdir \
  --seq-lens 4096 8192 16384 \
  --tasks niah variable_tracking qa \
  --output-jsonl /abs/path/to/ruler.jsonl
```

说明：
- 若已手动生成样本，可加 `--skip-generate` 仅做转换。
- 转换后的 `ruler.jsonl` 每行包含 `id/question/turns/category` 以及 `ruler_meta`（保留 `task/seq_len/gold`）。

### 2) 使用现有 CLI 进行评测
```bash
CUDA_VISIBLE_DEVICES=0 python -m cpmcu.cli \
  --model-path /abs/path/to/model \
  --model-type minicpm4 --minicpm4-yarn True \
  --dataset qa --dataset-path /abs/path/to/ruler.jsonl \
  --use-chat-template False --num-generate 16 \
  --chunk-length 4096 --memory-limit 0.95 --cuda-graph True \
  --sink-window-size 4 --block-window-size 16 --sparse-topk-k 32 --sparse-switch 2048 \
  --output-dir /abs/path/to/benchmark/results/logs --plain-output True
```

评测输出文件位于 `--output-dir`，文件名形如：`qa_<model>_<timestamp>.json`。

### 3) 评分（保留 RULER 指标口径）
```bash
python scripts/ruler/score_with_ruler.py \
  --dataset-jsonl /abs/path/to/ruler.jsonl \
  --results-json /abs/path/to/benchmark/results/logs/qa_<model>_<timestamp>.json \
  --output-dir /abs/path/to/benchmark/results/ruler
```

说明：
- 该脚本将按 `id` 对齐样本与预测，基于 `ruler_meta.gold` 进行打分。当前默认使用 EM（归一化精确匹配）；后续可在 `--ruler-repo` 存在时优先调用官方评分模块以获得更丰富的细项指标和曲线。
- 产出文件命名形如：`ruler_metrics_<timestamp>.json`，包含整体准确率、按任务与序列长度分组的准确率、以及对齐的样本与预测对。

### 多卡并行（可选）
- 将 `ruler.jsonl` 切分为多份，绑定不同 GPU 并发评测，最后汇总多个结果文件后再调用 `score_with_ruler.py` 统一评分。

### 合规
- RULER 采用 Apache-2.0 许可证，可修改并集成，但需在分发时保留其 `LICENSE/NOTICE` 并注明修改来源。


