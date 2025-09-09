# CPM.cu Docker开发指南

本文档面向**开发者**，说明如何构建和定制CPM.cu Docker镜像。

如果你是**用户**，只想使用预构建的镜像，请查看 [docker_use.md](docker_use.md)。

## 概述

这个Dockerfile为CPM.cu项目提供了完整的CUDA 12.6环境，支持Python 3.10和灵活的编译配置。

**设计目标**: 构建开箱即用的通用镜像，支持主流GPU架构，用户无需重新编译。

## 快速构建

### 使用构建脚本（推荐）

```bash
# 默认构建（支持所有主流GPU）
./build.sh

# 自定义CUDA架构（减小镜像大小）
./build.sh "80,90"          # 仅数据中心GPU
./build.sh "86,89"          # 仅消费级GPU  
./build.sh "87"             # 仅Jetson设备
```

### 手动构建

```bash
# 默认构建（支持所有主流GPU）
docker build -t cpmcu:cuda12.6-release .

# 自定义CUDA架构
docker build --build-arg CUDA_ARCH="80,86" -t cpmcu:cuda12.6-custom .
```

## 构建配置

### 默认配置（推荐）
- **CUDA架构**: `80,86,87,89,90` (支持主流GPU，120需要cuda12.8+)
- **数据类型**: `fp16,bf16` (完整支持)
- **编译模式**: Release (生产优化)
- **并行编译**: 4线程 (内存友好)
- **镜像大小**: ~4-5GB

### 支持的GPU架构

| 架构 | GPU型号 | 用途 |
|------|---------|------|
| 80 | A100, A800 | 数据中心 |
| 86 | RTX 3090, RTX 3080, RTX 3070 | 消费级 |
| 87 | Jetson Orin | 边缘计算 |
| 89 | RTX 4090, RTX 4080 | 最新消费级 |
| 90 | H100, H800 | 最新数据中心 |
| 120（需cuda12.8+） | RTX 5090, RTX 5070Ti | Blackwell架构 |

## 构建参数

### 环境变量

构建时支持的环境变量：

- `CUDA_ARCH`: CUDA架构列表（如"80,86,89"）
- `CPMCU_DEBUG`: 调试模式（0=关闭，1=开启）
- `CPMCU_PERF`: 性能监控（0=关闭，1=开启）
- `CPMCU_DTYPE`: 数据类型支持（默认"fp16,bf16"）
- `MAX_JOBS`: 编译并行度
- `NVCC_THREADS`: NVCC编译线程数

### 自定义构建示例

```bash
# 仅支持数据中心GPU（减小镜像）
./build.sh "80,90"

# 仅支持消费级GPU
./build.sh "86,89"

# 支持特定GPU
./build.sh "89"              # 仅RTX 4090/4080

# 包含最新Blackwell架构
./build.sh "80,86,89,120"    # 主流 + RTX 50系列（需cuda12.8+）
```

## 镜像优化

### 大小优化策略

1. **精确架构选择**: 只编译目标GPU架构
2. **多阶段构建**: 分离构建和运行环境
3. **依赖清理**: 移除构建工具和缓存

### 性能优化策略

1. **并行编译**: 合理设置MAX_JOBS
2. **内存管理**: 避免编译时OOM
3. **缓存利用**: 利用Docker层缓存

## 故障排除

### 编译失败

**内存不足**:
```bash
# 减少并行编译
docker build --build-arg MAX_JOBS=2 -t cpmcu:cuda12.6 .
```

**CUDA架构不支持**:
```bash
# 检查GPU计算能力
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

**依赖问题**:
```bash
# 检查CUDA版本兼容性
docker run --rm nvidia/cuda:12.6.2-devel-ubuntu22.04 nvcc --version
```

### 构建优化

**加速构建**:
```bash
# 使用BuildKit
DOCKER_BUILDKIT=1 docker build -t cpmcu:cuda12.6 .

# 并行构建
docker build --build-arg MAX_JOBS=8 -t cpmcu:cuda12.6 .
```

**减小镜像**:
```bash
# 最小架构集
./build.sh "89"  # 仅RTX 4090/4080

# 多阶段构建优化
docker build --target=runtime -t cpmcu:cuda12.6-slim .
```

## 开发工作流

### 本地开发

1. **修改Dockerfile**
2. **测试构建**: `./build.sh`
3. **验证功能**: 运行测试容器
4. **优化配置**: 调整编译参数

### CI/CD集成

```yaml
# GitHub Actions示例
- name: Build Docker Image
  run: |
    cd docker/cuda12.6
    ./build.sh "80,86,89"  # 支持主流GPU
    
- name: Test Image
  run: |
    docker run --gpus all cpmcu:cuda12.6-release python -c "import cpmcu; print('OK')"
```

### 发布流程

1. **构建多架构镜像**
2. **运行完整测试**
3. **标记版本**: `docker tag cpmcu:cuda12.6-release cpmcu:v1.0.0`
4. **推送镜像**: `docker push cpmcu:v1.0.0`

## 高级定制


### 自定义依赖

```dockerfile
# 在安装CPM.cu前添加
RUN pip install your-custom-package
```

### 多阶段构建

```dockerfile
# 构建阶段
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS builder
# ... 构建逻辑 ...

# 运行阶段
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04 AS runtime
COPY --from=builder /workspace/CPM.cu /workspace/CPM.cu
```
