# CPM.cu Arguments Processing Architecture

## Overview

The CPM.cu project implements a unified argument processing architecture that supports multiple execution scenarios including test generation, server deployment, and model-specific configurations. The architecture is designed with modularity, extensibility, and backward compatibility in mind.

## Core Architecture Components

### 1. Unified Argument Processing (`cpmcu/args.py`)

The central module that provides standardized argument parsing and configuration management across all components.

#### Key Design Principles:
- **Unified Interface**: Single source of truth for all configuration parameters
- **Logical Grouping**: Arguments are organized into functional groups for better maintainability
- **Type Safety**: Built-in type conversion and validation
- **Default Configuration**: Intelligent defaults with override capabilities
- **Multi-Context Support**: Handles both server and test execution contexts

#### Core Functions:

```python
def parse_server_args() -> Tuple[argparse.Namespace, Dict[str, Any]]
def parse_test_args() -> Tuple[argparse.Namespace, Dict[str, Any]]
def merge_args_with_config(args, default_config: Dict[str, Any], is_server: bool = False) -> Dict[str, Any]
def display_config_summary(config: Dict[str, Any], title: str = "Configuration Parameters")
```

### 2. Argument Groups Structure

Arguments are logically organized into the following groups:

#### Model Configuration Group
- Model paths and type configuration
- Primary model, draft model, and frequency speculation paths
- Model type detection and dtype configuration

#### System Features Group  
- CUDA graph optimization
- Memory management
- Random seed configuration

#### Speculative Decoding Group
- Window size and iteration parameters
- Tree-based speculation configuration
- Frequency speculation vocabulary settings

#### Sparse Attention Group
- Sparse attention mechanism configuration
- LSE compression settings
- Window sizes and top-k parameters

#### Generation Parameters Group
- Temperature and sampling configuration
- Token generation limits
- Chat template handling

## Parameter Reference

### Core Interfaces
- **`parse_server_args()`**: Server deployment (adds `--host`, `--port`)
- **`parse_test_args()`**: Test generation (adds `--prompt-file/text`, `--num-generate`, etc.)

### Parameter Groups

| Group | Key Parameters | Description |
|-------|----------------|-------------|
| **Model** | `--model-path` (required), `--draft-model-path`, `--frspec-path`, `--model-type`, `--dtype` | Model paths and configuration |
| **System** | `--cuda-graph`, `--memory-limit`, `--random-seed`, `--chunk-length` | System optimization settings |
| **Speculative** | `--spec-window-size`, `--spec-num-iter`, `--spec-topk-per-iter`, `--spec-tree-size`, `--frspec-vocab-size` | Speculative decoding parameters |
| **Sparse** | `--apply-sparse`, `--apply-compress-lse`, `--sink-window-size`, `--block-window-size`, `--sparse-topk-k`, `--sparse-switch` | Sparse attention configuration |
| **Generation** | `--temperature`, `--use-stream`, `--use-terminators`, `--use-chat-template`, `--num-generate` | Text generation settings |
| **Server** | `--host`, `--port` | Server deployment settings |
| **Test** | `--prompt-file`, `--prompt-text`, `--use-enter` (deprecated), `--use-decode-enter` (deprecated) | Test-specific parameters |
| **MiniCPM4** | `--path-prefix`, `--apply-quant`, `--apply-eagle`, `--apply-eagle-quant`, `--minicpm4-yarn` | MiniCPM4-exclusive parameters |

## Configuration Processing

The system merges command-line arguments with defaults, performs type conversion and validation. Boolean arguments support flexible formats: `true/false`, `yes/no`, `1/0`, or flag-only usage (e.g., `--cuda-graph`).

## Script Architecture Comparison

### Generic vs Model-Specific Scripts

The CPM.cu project provides two distinct sets of execution scripts, each serving different purposes and use cases:

#### Generic Scripts (`examples/`)
- **Purpose**: Universal wrappers for core CPM.cu functionality
- **Target**: Any supported model type (llama, minicpm, minicpm4, etc.)
- **Architecture**: Simple wrappers around `cpmcu/generate.py` and `cpmcu/server.py`
- **Arguments**: Supports all standard CPM.cu arguments through unified interface

**Generic Test Generation** (`examples/test_generate.py`):
```python
# Simple wrapper that directly calls cpmcu.generate.main()
from cpmcu.generate import main
if __name__ == "__main__": main()
```

**Generic Server** (`examples/start_server.py`):
```python  
# Simple wrapper that directly calls cpmcu.server.main()
from cpmcu.server import main
if __name__ == "__main__": main()
```

#### Model-Specific Scripts (`examples/minicpm4/`)
- **Purpose**: Optimized implementations for MiniCPM4 model reproduction
- **Target**: Specifically designed for MiniCPM4 models
- **Architecture**: Extended functionality with model-specific optimizations
- **Arguments**: Supports both standard CPM.cu arguments AND MiniCPM4-exclusive parameters

**MiniCPM4 Test Generation** (`examples/minicpm4/test_generate.py & examples/minicpm4/start_server.py`):
- Extends base test interface with MiniCPM4-specific defaults
- YARN support configuration 
- Sparse attention optimization
- Integrated haystack prompt generation
- Additional MiniCPM4-exclusive arguments:
  ```python
  --path-prefix: Model repository prefix (default: openbmb)
  --apply-quant: Enable quantization for base model (auto-detected from model path)
  --apply-eagle: Enable Eagle speculative decoding (auto-detected when draft model provided)
  --apply-eagle-quant: Enable quantization for Eagle draft model (auto-detected from draft model path)
  --minicpm4-yarn: Enable MiniCPM4 YARN for long context support (default: True)
  --use-enter: Interactive mode - wait for user input before starting prefill phase (default: False) [DEPRECATED]
  --use-decode-enter: Interactive mode - wait for user input before starting decode phase (default: False) [DEPRECATED]
  ```

#### MiniCPM4 Parameter Design Philosophy

**Testing and Comparison Parameters**:
The parameters `--path-prefix`, `--apply-quant`, `--apply-eagle`, and `--apply-eagle-quant` are primarily designed to facilitate testing and performance comparison across different model configurations. For MiniCPM4 models, each mode corresponds to fixed model names, allowing users to easily switch between configurations using boolean combinations:

- **Base Model**: `{path-prefix}/MiniCPM4-8B`
- **Quantized Model**: `{path-prefix}/MiniCPM4-8B-marlin-cpmcu`
- **Draft Model (Eagle)**: `{path-prefix}/MiniCPM4-8B-Eagle-FRSpec`
- **Quantized Draft Model**: `{path-prefix}/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu`

**Interactive Parameters (Deprecated)**:
The `--use-enter` and `--use-decode-enter` parameters are temporary interactive features designed for demonstration purposes. These will be removed in future versions as they are primarily used for debugging and educational demonstrations.

#### MiniCPM4 Configuration Matrix

| Configuration | Base Model | Draft Model | Description |
|---------------|------------|-------------|-------------|
| **Base** | MiniCPM4-8B | - | Standard model without quantization or speculative decoding |
| **Quantized** | MiniCPM4-8B-marlin-cpmcu | - | Quantized base model for improved performance |
| **Eagle** | MiniCPM4-8B | MiniCPM4-8B-Eagle-FRSpec | Speculative decoding with Eagle draft model |
| **Eagle + Quant** | MiniCPM4-8B-marlin-cpmcu | MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu | Fully optimized: quantized base + quantized Eagle draft |

The system automatically selects the appropriate model paths based on the boolean parameter combinations, eliminating the need for manual path specification in most testing scenarios.

### Key Differences Summary

| Aspect | Generic Scripts | MiniCPM4 Scripts |
|--------|----------------|------------------|
| **Model Support** | Universal (any model) | MiniCPM4 optimized |
| **Arguments** | Standard CPM.cu args only | Standard + MiniCPM4-specific |
| **Defaults** | Generic defaults | MiniCPM4-optimized defaults |
| **Implementation** | Simple wrapper functions | Extended with custom logic |
| **Use Case** | General model testing/serving | MiniCPM4 reproduction & optimization |
| **Configuration** | Manual model path specification | Automatic path selection via boolean combinations |
| **Testing Focus** | Flexible parameter tuning | Structured performance comparison |

### When to Use Which

**Use Generic Scripts When:**
- Working with various model types (llama, minicpm, etc.)
- Need maximum flexibility in configuration
- Developing or testing new model integrations
- Require custom parameter combinations

**Use MiniCPM4 Scripts When:**
- Specifically working with MiniCPM4 models
- Want optimized defaults and performance
- Need YARN configuration support
- Reproducing MiniCPM4 research results
  - Minimal configuration overhead desired

## Usage Examples

### Core Usage Examples

**Generic Scripts** (any model):
```bash
python -m cpmcu.generate --model-path path/to/model --prompt-text "Your prompt"
python -m cpmcu.server --model-path path/to/model --host 0.0.0.0 --port 8000
```

**MiniCPM4 Scripts** (optimized defaults):
```bash
python examples/minicpm4/test_generate.py --prompt-text "Your prompt"  # Auto-configured
python examples/minicpm4/start_server.py --host 0.0.0.0 --port 8000     # Server mode
```

### MiniCPM4 Configuration Testing
```bash
# Test different configurations
python examples/minicpm4/test_generate.py --prompt-text "Your prompt"                    # Base model
python examples/minicpm4/test_generate.py --apply-quant true --prompt-text "Your prompt"  # Quantized
python examples/minicpm4/test_generate.py --apply-eagle true --prompt-text "Your prompt"  # Eagle spec
python examples/minicpm4/test_generate.py --apply-quant --apply-eagle --apply-eagle-quant --prompt-text "Your prompt"  # Full optimization
```

**Note**: In MiniCPM4 scripts, `--apply-quant`, `--apply-eagle`, and `--apply-eagle-quant` are automatically detected from model paths and typically don't need manual specification.

### Interactive Parameters (Deprecated)

**⚠️ Note**: `--use-enter` and `--use-decode-enter` are deprecated interactive debugging parameters that will be removed in future versions.

## Model-Specific Extensions

## Integration Points

- **`cpmcu/generate.py`**: Core generation via `run_generation(config)`
- **`cpmcu/server.py`**: Server deployment via `launch_server(config)`  
- **`cpmcu/utils.py`**: Configuration defaults and model utilities

## Extensibility & Best Practices

**Adding Parameters**: Add to `add_model_config_args()` → update `get_default_config()` → handle in `merge_args_with_config()` → update display.

**Model Extensions**: Create model-specific config modules with custom parsers and validation logic.

**Key Principles**: Logical grouping, sensible defaults, flexible boolean formats, comprehensive validation.

## Summary

This architecture provides a robust, extensible foundation for argument processing across the CPM.cu ecosystem while maintaining simplicity and consistency in usage patterns.

### MiniCPM4 Design Philosophy

The MiniCPM4-specific parameters exemplify a key design principle in CPM.cu: **simplifying complex configurations through intelligent automation**. Rather than requiring users to manually specify multiple model paths for different configurations, the system provides boolean switches that automatically select the appropriate pre-defined model combinations. This approach:

- **Streamlines Testing**: Enables quick performance comparisons across different optimization levels
- **Reduces Errors**: Eliminates manual path specification errors through automated selection
- **Maintains Flexibility**: Still allows custom path prefixes for different model repositories
- **Supports Research**: Facilitates systematic evaluation of quantization and speculative decoding impacts

The deprecation of interactive parameters (`--use-enter`, `--use-decode-enter`) reflects the project's evolution from debugging tools to production-ready interfaces, demonstrating the commitment to maintaining clean, focused APIs as the system matures. 