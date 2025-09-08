# CPM.cu Docker Developer Guide

This document is for **developers** who want to build and customize CPM.cu Docker images.

If you are a **user** who just wants to use pre-built images, please see [docker_use.md](docker_use.md).

## Overview

This Dockerfile provides a complete CUDA 12.6 environment for the CPM.cu project, supporting Python 3.10 and flexible compilation configurations.

**Design Goal**: Build ready-to-use universal images that support mainstream GPU architectures, eliminating the need for users to recompile.

## Quick Build

### Using Build Script (Recommended)

```bash
# Default build (supports all mainstream GPUs)
./build.sh

# Custom CUDA architecture (reduce image size)
./build.sh "80,90"          # Data center GPUs only
./build.sh "86,89"          # Consumer GPUs only  
./build.sh "87"             # Jetson devices only
```

### Manual Build

```bash
# Default build (supports all mainstream GPUs)
docker build -t cpmcu:cuda12.6-release .

# Custom CUDA architecture
docker build --build-arg CUDA_ARCH="80,86" -t cpmcu:cuda12.6-custom .
```

## Build Configuration

### Default Configuration (Recommended)
- **CUDA Architecture**: `80,86,87,89,90` (supports mainstream GPUs, 120 requires CUDA 12.8+)
- **Data Types**: `fp16,bf16` (full support)
- **Build Mode**: Release (production optimized)
- **Parallel Compilation**: 4 threads (memory friendly)
- **Image Size**: ~4-5GB

### Supported GPU Architectures

| Architecture | GPU Models | Use Case |
|--------------|------------|----------|
| 80 | A100, A800 | Data Center |
| 86 | RTX 3090, RTX 3080, RTX 3070 | Consumer |
| 87 | Jetson Orin | Edge Computing |
| 89 | RTX 4090, RTX 4080 | Latest Consumer |
| 90 | H100, H800 | Latest Data Center |
| 120 | RTX 5090, RTX 5070Ti | Blackwell Architecture |

## Build Parameters

### Environment Variables

Supported environment variables during build:

- `CUDA_ARCH`: CUDA architecture list (e.g., "80,86,89")
- `CPMCU_DEBUG`: Debug mode (0=off, 1=on)
- `CPMCU_PERF`: Performance monitoring (0=off, 1=on)
- `CPMCU_DTYPE`: Data type support (default "fp16,bf16")
- `MAX_JOBS`: Compilation parallelism
- `NVCC_THREADS`: NVCC compilation thread count

### Custom Build Examples

```bash
# Data center GPUs only (reduce image size)
./build.sh "80,90"

# Consumer GPUs only
./build.sh "86,89"

# Specific GPU support
./build.sh "89"              # RTX 4090/4080 only

# Include latest Blackwell architecture
./build.sh "80,86,89,120"    # Mainstream + RTX 50 series
```

## Image Optimization

### Size Optimization Strategies

1. **Precise Architecture Selection**: Compile only target GPU architectures
2. **Multi-stage Builds**: Separate build and runtime environments
3. **Dependency Cleanup**: Remove build tools and caches

### Performance Optimization Strategies

1. **Parallel Compilation**: Set MAX_JOBS appropriately
2. **Memory Management**: Avoid OOM during compilation
3. **Cache Utilization**: Leverage Docker layer caching

## Troubleshooting

### Build Failures

**Out of Memory**:
```bash
# Reduce parallel compilation
docker build --build-arg MAX_JOBS=2 -t cpmcu:cuda12.6 .
```

**Unsupported CUDA Architecture**:
```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

**Dependency Issues**:
```bash
# Check CUDA version compatibility
docker run --rm nvidia/cuda:12.6.2-devel-ubuntu22.04 nvcc --version
```

### Build Optimization

**Accelerate Build**:
```bash
# Use BuildKit
DOCKER_BUILDKIT=1 docker build -t cpmcu:cuda12.6 .

# Parallel build
docker build --build-arg MAX_JOBS=8 -t cpmcu:cuda12.6 .
```

**Reduce Image Size**:
```bash
# Minimal architecture set
./build.sh "89"  # RTX 4090/4080 only

# Multi-stage build optimization
docker build --target=runtime -t cpmcu:cuda12.6-slim .
```

## Development Workflow

### Local Development

1. **Modify Dockerfile**
2. **Test Build**: `./build.sh`
3. **Verify Functionality**: Run test container
4. **Optimize Configuration**: Adjust compilation parameters

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Build Docker Image
  run: |
    cd docker/cuda12.6
    ./build.sh "80,86,89"  # Support mainstream GPUs
    
- name: Test Image
  run: |
    docker run --gpus all cpmcu:cuda12.6-release python -c "import cpmcu; print('OK')"
```

### Release Process

1. **Build Multi-architecture Images**
2. **Run Complete Tests**
3. **Tag Version**: `docker tag cpmcu:cuda12.6-release cpmcu:v1.0.0`
4. **Push Image**: `docker push cpmcu:v1.0.0`

## Advanced Customization

### Adding New Architectures

```dockerfile
# Add new architecture in Dockerfile
ARG CUDA_ARCH="80,86,87,89,90,120,130"  # Add 130
```

### Custom Dependencies

```dockerfile
# Add before installing CPM.cu
RUN pip install your-custom-package
```

### Multi-stage Build

```dockerfile
# Build stage
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS builder
# ... build logic ...

# Runtime stage
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04 AS runtime
COPY --from=builder /workspace/CPM.cu /workspace/CPM.cu
```
