# CPM.cu

<strong>中文 | [English Version](./README.md)</strong>

CPM.cu 是一个针对端侧大模型推理设计的轻量、高效的 CUDA 推理框架，核心支持 **稀疏架构**、**投机采样** 和 **低位宽量化** 等前沿技术创新。

<div id="news"></div>

## 🔥 项目进展

- [2025.06.06] 为 [MiniCPM4](https://github.com/openbmb/minicpm) 优化。
    - 支持 InfLLM-v2 注意力内核
    - 支持 MTP 层的滑动窗口，优化长上下文处理
    - 支持 MTP 层的量化
- [2025.05.29] 支持 [SpecMQuant](https://github.com/AI9Stars/SpecMQuant) 的量化。
    - 支持 LLM 的 Marlin GPTQ 内核
    - 支持量化 LLM 的投机采样
- [2025.03.01] 在 [FR-Spec](https://github.com/thunlp/FR-Spec) 发布首个版本。
    - 速度最快的投机采样实现
    - 支持 FR-Spec, 基于词频优化的投机采样
    - 支持 Flash-Attention 中的树状投机采样验证
    - 支持静态内存管理和内存复用
    - 支持计算融合内核
    - 支持分块预填充
    - 支持 CUDA Graph

<div id="demo"></div>

## 效果演示

https://github.com/user-attachments/assets/ab36fd7a-485b-4707-b72f-b80b5c43d024

<div id="getstart"></div>

## 快速开始

- [框架安装](#install)
- [模型权重](#modelweights)
- [运行示例](#example)
- [OpenAI API 服务](#openai-api)

<div id="install"></div>

## 框架安装

### 从源码安装

本库的构建依赖于 PyTorch 和 Ninja，请在安装本库前确保已正确安装这两个依赖。

```bash
git clone https://github.com/OpenBMB/CPM.cu.git --recursive
cd CPM.cu
pip install .
```

如遇到安装问题，请根据错误提示进行解决，或通过 GitHub Issues 提交问题反馈。你可以使用 `python setup.py --help-config` 查看更多安装配置信息。

<div id="modelweights"></div>

## 准备模型

请按照 [MiniCPM4 的 README](https://github.com/openbmb/minicpm) 的说明下载模型权重。

<div id="example"></div>

## 运行示例

我们提供了一个简单的示例来展示如何使用 CPM.cu。

```bash
cd examples
python3 minicpm4/test_generate.py --prompt-file <输入文件路径>
```

如果您不指定模型路径，脚本将从 OpenBMB 的 Hugging Face 仓库加载模型。
如果你想使用本地路径，我们推荐不修改所有模型文件名并放在同一目录下，这样可以通过-p指定该目录运行模型。否则建议修改代码中的路径。
您可以使用 --help 了解更多关于脚本的功能。

我们还有一个脚本，`examples/long_prompt_gen.py`，用于生成长代码总结。
这个脚本会自动从本仓库中收集代码，并提示模型"Summarize the code."。

```bash
cd examples
python3 long_prompt_gen.py # 生成 prompt.txt (更多细节请见 --help)
python3 minicpm4/test_generate.py --prompt-file ../prompt.txt
```

输出应为如下格式：

```bash
Generated text (streaming output):
--------------------------------------------------
Prefilling: 100.0% (106850/106850 tokens) @ 6565.3 tokens/s - Complete!

<Generated Output HERE>
==================================================
Stream Generation Summary:
==================================================
Prefill length: 106850
Prefill time: 16.36 s
Prefill tokens/s: 6530.77
Mean accept length: 2.50
Decode length: 118
Decode time: 0.76 s
Decode tokens/s: 154.59
```

其中：

- `Prefill` (输入) 和 `Decode` (输出) 速度通过（长度、时间和 token/s）输出。
- `Mean accept length` (平均接受长度) 是使用投机采样时接受 token 的平均长度。

<div id="openai-api"></div>

## OpenAI API 服务（实验性）

启动 OpenAI 兼容的 API 服务器（参数与 `examples/minicpm4/test_generate.py` 相同）：

```bash
cd examples
python minicpm4/start_server.py [options]
```

测试 API，支持流式和非流式模式：

```bash
cd examples
python test_openai_api.py [--no-stream]
```

目前仅支持 `/v1/chat/completions` 接口，`model` 字段无效。

## 代码结构

```bash
CPM.cu/
├── src/
│   ├── flash_attn/ # attention: 稀疏, 投机树状验证等
│   ├── model/
│   │   ├── minicpm4/ # minicpm4 模型
│   │   ├── w4a16_gptq_marlin/ # Marlin GPTQ 计算内核
│   │   └── ... # 通用层
│   ├── entry.cu # pybind: 绑定 CUDA 和 Python
│   └── ...
├── cpmcu/ # Python 接口
└── ...
```
## 更多

### 词频文件生成
我们提供了FR-Spec的词频生成脚本，位于"scripts/fr_spec/gen_fr_index.py"，运行方式如下：
```bash
python scripts/fr_spec/gen_fr_index.py --model_path <your modelpath>
```
你可以修改代码使用自己的数据集。如果你的任务是特定垂直领域，根据领域构造词频对速度提升有显著收益。

### GPTQ转Marlin格式
我们提供了GPTQ量化模型转Marlin格式的转换脚本，位于"scripts/model_convert/gptq2marlin.py"，运行方式如下：
```bash
python scripts/model_convert/gptq2marlin.py \
    --src <gptq_model_path> \
    --dst <marlin_model_path>
```
该脚本支持MiniCPM、Llama和EAGLE格式，会自动检测模型类型并进行相应转换。

## 致谢

我们的 `src/flash_attn` 文件夹基于 [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/v2.6.3/csrc/flash_attn) 并进行了修改。

我们从以下仓库中获取了实现灵感：

- [EAGLE](https://github.com/SafeAILab/EAGLE)
- [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)

## 引用

如果您觉得我们的工作有价值，请引用我们的论文。

```
@article{zhao2025fr,
  title={FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling},
  author={Zhao, Weilin and Pan, Tengyu and Han, Xu and Zhang, Yudi and Sun, Ao and Huang, Yuxiang and Zhang, Kaihuo and Zhao, Weilun and Li, Yuxuan and Wang, Jianyong and others},
  journal={arXiv preprint arXiv:2502.14856},
  year={2025}
}

@article{zhang2025specmqaunt,
  title={Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design},
  author={Zhang, Yudi and Zhao, Weilin and Han, Xu and Zhao, Tiejun and Xu, Wang and Cao, Hailong and Zhu, Conghui},
  journal={arXiv preprint arXiv:2505.22179},
  year={2025}
}

@article{minicpm4,
  title={MiniCPM4: Ultra-Efficient LLMs on End Devices},
  author={MiniCPM},
  year={2025}
}