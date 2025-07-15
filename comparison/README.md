# 逻辑比较框架

这是一个模块化的逻辑比较框架，支持不同配置的投机解码推理和比较分析。

## 核心文件

### 配置模块
- **`config.py`**: 配置管理，包含 ComparisonConfig 类和参数解析器
- **`logits.py`**: CPM.cu 框架的逻辑捕获和推理功能  
- **`sglang_inference.py`**: SGLang 框架的EAGLE投机解码推理功能 *(新增)*

### 分析模块
- **`analysis.py`**: 数据比较和分析功能
- **`run_single_config.py`**: 运行单个配置 (CPM.cu框架)
- **`run_sglang_config.py`**: 运行单个配置 (SGLang框架) *(新增)*
- **`analyze_saved_data.py`**: 分析已保存的数据
- **`workflow.py`**: 完整的工作流程管理

## 主要功能

### 1. CPM.cu 投机解码推理 (原有)
使用传统的CPM.cu框架进行投机解码推理：

```bash
# 运行单个配置
python run_single_config.py --spec-num-iter 5 --spec-tree-size 32 --comparison-steps 20

# 使用预设配置
python run_single_config.py --config-name config1
```

### 2. SGLang EAGLE投机解码推理 (新增)
使用SGLang框架进行高速离线EAGLE投机解码推理：

```bash
# 运行SGLang推理
python run_sglang_config.py --spec-num-iter 5 --spec-tree-size 32 --comparison-steps 20 --verbose

# 保存结果
python run_sglang_config.py --config-name config1 --save-results --verbose

# 自定义参数
python run_sglang_config.py \
    --spec-num-iter 3 \
    --spec-tree-size 16 \
    --comparison-steps 15 \
    --model-path "unsloth/Meta-Llama-3.1-8B-Instruct" \
    --draft-model-path "jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B" \
    --save-results --verbose
```

#### SGLang推理特性
- **EAGLE投机解码**: 使用EAGLE算法进行高速投机解码
- **参数映射**: 自动将config参数映射到SGLang引擎参数
  - `spec_num_iter` → `speculative_num_steps`
  - `spec_tree_size` → `speculative_num_draft_tokens`
  - `get_topk_per_iter()` → `speculative_eagle_topk`
- **性能优化**: 参考官方配置的内存和CUDA优化
- **兼容性**: 与现有config系统完全兼容

### 3. 分析和比较
```bash
# 分析保存的数据
python analyze_saved_data.py --data1 logits_capture_config1.pkl --data2 sglang_capture_config1.pkl

# 完整工作流程
python workflow.py --config1-iter 5 --config1-tree-size 32 --config2-iter 2 --config2-tree-size 12
```

## 配置参数

### 核心投机参数
- `--spec-num-iter`: 投机迭代次数 (默认: 5)
- `--spec-tree-size`: 投机树大小 (默认: 32)  
- `--comparison-steps`: 比较的token数量 (默认: 20)

### 模型配置
- `--model-path`: 主模型路径 (默认: "unsloth/Meta-Llama-3.1-8B-Instruct")
- `--draft-model-path`: 草稿模型路径 (默认: "jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B")
- `--prompt-file`: 提示文件路径 (默认: "prompt_small.txt")

### 系统配置
- `--memory-limit`: 内存限制 (默认: 0.7)
- `--chunk-length`: 块长度 (默认: 1024)
- `--cuda-graph`: 启用CUDA图

## SGLang vs CPM.cu 对比

| 特性 | CPM.cu | SGLang EAGLE |
|------|--------|--------------|
| 推理框架 | 传统CPM.cu | SGLang + EAGLE |
| 投机算法 | 基础投机解码 | EAGLE优化算法 |
| 性能优化 | 基本优化 | 高级缓存和图优化 |
| 内存管理 | 基础管理 | 智能内存分片 |
| 并发支持 | 有限 | 高并发支持 |
| 配置兼容性 | ✅ | ✅ |

## 预设配置

框架提供两个预设配置：

- **Config1**: `spec_num_iter=5, spec_tree_size=32` (高性能配置)
- **Config2**: `spec_num_iter=2, spec_tree_size=12` (低延迟配置)

## 输出文件

### CPM.cu框架
- `logits_capture_<config_name>.pkl`: CPM.cu推理结果

### SGLang框架  
- `sglang_capture_<config_name>.pkl`: SGLang推理结果

### 分析结果
- `detailed_logits_comparison_results.json`: 详细比较分析结果

## 环境要求

### 基础要求
- Python 3.8+
- CUDA 支持的GPU
- 足够的GPU内存 (建议16GB+)

### CPM.cu依赖
- CPM.cu框架及相关依赖

### SGLang依赖 *(新增)*
- SGLang: `pip install sglang`
- PyTorch with CUDA support
- 兼容的模型文件

## 错误处理

### 常见问题
1. **内存不足**: 降低 `--memory-limit` 或 `--spec-tree-size`
2. **模型加载失败**: 检查模型路径和网络连接
3. **推理超时**: 减少 `--comparison-steps` 或优化硬件配置

### SGLang特定问题
1. **SGLang未安装**: `pip install sglang`
2. **EAGLE模型不兼容**: 确认草稿模型路径正确
3. **CUDA图错误**: 尝试设置 `--cuda-graph false`

## 性能建议

### SGLang优化建议
1. **内存优化**: 根据GPU内存调整 `--memory-limit`
2. **并发调优**: 大批量推理时调整并发参数
3. **缓存优化**: 启用radix cache以提高效率
4. **CUDA图**: 在支持的硬件上启用CUDA图加速

## 扩展说明

该框架设计为可扩展的，可以轻松添加其他推理框架：

1. 创建新的推理模块 (如 `framework_inference.py`)
2. 实现相同的接口 (配置输入，结果输出)
3. 添加对应的运行脚本
4. 更新README文档

现在的框架已经支持CPM.cu和SGLang两个主要的投机解码框架，为投机解码研究提供了全面的工具集。 