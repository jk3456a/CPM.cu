#!/usr/bin/env python3
"""
CPM.cu Utility Functions

Common utility functions for generic model support.
"""

import os
import sys
import json
import torch
from huggingface_hub import snapshot_download


class ConfigurationManager:
    """统一的配置管理类"""
    
    @staticmethod
    def convert_dtype(dtype_value, for_server=False):
        """统一的dtype转换逻辑"""
        if dtype_value is None:
            return None
            
        if isinstance(dtype_value, torch.dtype):
            if for_server:
                # Server mode: convert to string for JSON serialization
                return 'float16' if dtype_value == torch.float16 else 'bfloat16'
            else:
                return dtype_value
        elif isinstance(dtype_value, str):
            if for_server:
                return dtype_value
            else:
                # Convert string to torch dtype
                return torch.float16 if dtype_value == 'float16' else torch.bfloat16
        else:
            # Handle other formats
            dtype_str = str(dtype_value).split('.')[-1]
            if for_server:
                return dtype_str
            else:
                return torch.float16 if 'float16' in dtype_str else torch.bfloat16
    
    @staticmethod
    def load_from_file(config_path: str, for_server: bool = False) -> dict:
        """Load configuration from JSON file with unified dtype handling"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Apply dtype conversion
        if 'dtype' in config:
            config['dtype'] = ConfigurationManager.convert_dtype(config['dtype'], for_server)
        
        return config


def detect_quantization_from_path(model_path):
    """Auto-detect quantization from model path"""
    path_lower = model_path.lower()
    # Check for common quantization keywords in model path
    quant_keywords = ['marlin', 'gptq', 'quant', 'awq', 'int4', 'int8', 'w4a16', 'qat']
    return any(keyword in path_lower for keyword in quant_keywords)


def setup_model_paths(config):
    """Setup and validate model paths with automatic feature detection"""
    model_path = check_or_download_model(config['model_path'])
    
    # Auto-detect model type if not specified
    if config.get('model_type', 'auto') == 'auto':
        config['model_type'] = detect_model_type(model_path)
        print(f"Auto-detected model type: {config['model_type']}")
    
    # Auto-detect quantization from main model path
    detected_quant = detect_quantization_from_path(model_path)
    config['apply_quant'] = detected_quant
    if detected_quant:
        print(f"Auto-detected quantization from model path: {model_path}")
    else:
        print(f"No quantization detected from model path: {model_path}")
    
    draft_model_path = None
    if config.get('draft_model_path'):
        draft_model_path = check_or_download_model(config['draft_model_path'])
        
        # Enable speculative decoding when draft model is present
        config['apply_speculative'] = True
        print(f"Draft model specified, enabling speculative decoding")
        
        # Auto-detect draft model quantization
        detected_spec_quant = detect_quantization_from_path(draft_model_path)
        config['apply_spec_quant'] = detected_spec_quant
        if detected_spec_quant:
            print(f"Auto-detected quantization for draft model: {draft_model_path}")
        else:
            print(f"No quantization detected for draft model: {draft_model_path}")
    else:
        # No draft model, disable speculative decoding
        config['apply_speculative'] = False
        config['apply_spec_quant'] = False
    
    frspec_path = None
    if config.get('frspec_path'):
        # Handle frspec_path the same way as model paths - support both local paths and HF URLs
        frspec_path = check_or_download_model(config['frspec_path'])
        if not os.path.exists(frspec_path):
            print(f"Warning: FRSpec file not found: {config['frspec_path']}")
            frspec_path = None
    
    return model_path, draft_model_path, frspec_path


def check_or_download_model(path):
    """Check if model exists locally, otherwise download from HuggingFace"""
    if os.path.exists(path):
        return path
    else:
        cache_dir = snapshot_download(path)
        return cache_dir


def detect_model_type(model_path):
    """Auto-detect model type based on config.json"""
    try:
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check architecture type
            arch = config.get('architectures', [''])[0].lower()
            model_type = config.get('model_type', '').lower()
            
            if 'minicpm' in arch or 'minicpm' in model_type:
                # Further detect MiniCPM version
                if config.get('scale_emb', None) is not None:
                    return 'minicpm4'
                else:
                    return 'minicpm'
            elif 'llama' in arch or 'llama' in model_type:
                return 'llama'
            else:
                return 'unknown'
    except Exception as e:
        print(f"Warning: Could not detect model type from {model_path}: {e}")
        return 'unknown'


class ModelFactory:
    """简化的模型工厂类"""
    
    @staticmethod
    def create_model(model_path, draft_model_path, config):
        """Create model instance based on configuration"""
        from .llm import LLM
        from .llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
        from .speculative import LLM_with_eagle
        from .speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
        
        # Handle dtype conversion using unified manager
        dtype_value = ConfigurationManager.convert_dtype(config['dtype'], for_server=False)
        
        # Build common arguments
        common_kwargs = ModelFactory._build_common_kwargs(config, dtype_value)
        
        # Auto-detect model features
        base_model_quantized = config.get('apply_quant', detect_quantization_from_path(model_path))
        has_draft_model = draft_model_path is not None
        draft_model_quantized = config.get('apply_spec_quant', 
                                         detect_quantization_from_path(draft_model_path) if has_draft_model else False)
        
        ModelFactory._print_model_info(base_model_quantized, has_draft_model, draft_model_quantized, config)
        
        # Create model based on configuration
        if base_model_quantized:
            if has_draft_model:
                print(f"Creating quantized model with Eagle speculative decoding")
                spec_kwargs = ModelFactory._build_spec_kwargs(config, draft_model_quantized)
                return W4A16GPTQMarlinLLM_with_eagle(draft_model_path, model_path, **common_kwargs, **spec_kwargs)
            else:
                print(f"Creating quantized model")
                return W4A16GPTQMarlinLLM(model_path, **common_kwargs)
        else:
            if has_draft_model:
                print(f"Creating model with Eagle speculative decoding")
                spec_kwargs = ModelFactory._build_spec_kwargs(config, draft_model_quantized)
                return LLM_with_eagle(draft_model_path, model_path, **common_kwargs, **spec_kwargs)
            else:
                print(f"Creating standard model")
                return LLM(model_path, **common_kwargs)
    
    @staticmethod
    def _build_common_kwargs(config, dtype_value):
        """构建通用参数"""
        return {
            'dtype': dtype_value,
            'chunk_length': config['chunk_length'],
            'cuda_graph': config['cuda_graph'],
            'apply_sparse': config.get('model_type') == 'minicpm4',
            'sink_window_size': config['sink_window_size'],
            'block_window_size': config['block_window_size'],
            'sparse_topk_k': config['sparse_topk_k'],
            'sparse_switch': config['sparse_switch'],
            'apply_compress_lse': config['apply_compress_lse'],
            'memory_limit': config['memory_limit'],
            'use_enter': config.get('use_enter', False),
            'use_decode_enter': config.get('use_decode_enter', False),
            'temperature': config['temperature'],
            'random_seed': config['random_seed'],
        }
    
    @staticmethod
    def _build_spec_kwargs(config, draft_model_quantized):
        """构建投机解码参数"""
        return {
            'num_iter': config.get('spec_num_iter', 2),
            'topk_per_iter': config.get('spec_topk_per_iter', 10),
            'tree_size': config.get('spec_tree_size', 12),
            'eagle_window_size': config.get('spec_window_size', 1024),
            'frspec_vocab_size': config.get('frspec_vocab_size', 32768),
            'apply_eagle_quant': draft_model_quantized,
            # Model-specific settings
            'use_rope': config.get('model_type') in ['minicpm', 'minicpm4'],
            'use_input_norm': config.get('model_type') in ['minicpm', 'minicpm4'],
            'use_attn_norm': config.get('model_type') in ['minicpm', 'minicpm4']
        }
    
    @staticmethod
    def _print_model_info(base_quantized, has_draft, draft_quantized, config):
        """打印模型信息"""
        print(f"Model creation:")
        print(f"  Base model quantized: {base_quantized}")
        print(f"  Speculative decoding: {has_draft}")
        print(f"  Draft model quantized: {draft_quantized}")
        print(f"  Model type: {config.get('model_type')}")


def setup_frspec_vocab(llm, frspec_path, frspec_vocab_size=32768):
    """Setup frequency speculative vocabulary for speculative models"""
    if not frspec_path:
        return False
        
    # If frspec_path is a directory (model directory), look for freq_{vocab_size}.pt file
    if os.path.isdir(frspec_path):
        freq_file = os.path.join(frspec_path, f"freq_{frspec_vocab_size}.pt")
        if os.path.exists(freq_file):
            frspec_path = freq_file
        else:
            print(f"Warning: FRSpec file freq_{frspec_vocab_size}.pt not found in directory: {frspec_path}")
            return False
    
    # If frspec_path is a specific file, use it directly
    if os.path.exists(frspec_path):
        print(f"Loading frequency vocabulary from: {frspec_path}")
        with open(frspec_path, 'rb') as f:
            token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
        llm._load("token_id_remap", token_id_remap, cls="eagle")
        return True
    else:
        print(f"Warning: FRSpec file not found: {frspec_path}")
        return False 