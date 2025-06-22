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

def load_config_from_file(config_path: str, keep_dtype_as_string: bool = False) -> dict:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Convert dtype string to torch dtype unless keeping as string for server mode
    if 'dtype' in config and not keep_dtype_as_string:
        if config['dtype'] == 'float16':
            config['dtype'] = torch.float16
        elif config['dtype'] == 'bfloat16':
            config['dtype'] = torch.bfloat16
    
    return config

def get_default_config():
    """Get default configuration for generic models"""
    return {
        "use_stream": True,
        "apply_speculative": False,
        "apply_quant": False,
        "apply_sparse": False,
        "apply_spec_quant": False,
        "frspec_vocab_size": 32768,
        "spec_window_size": 1024,
        "spec_num_iter": 2,
        "spec_topk_per_iter": 10,
        "spec_tree_size": 12,
        "apply_compress_lse": True,
        "sink_window_size": 1,
        "block_window_size": 8,
        "sparse_topk_k": 64,
        "sparse_switch": 1,
        "num_generate": 256,
        "chunk_length": 2048,
        "memory_limit": 0.9,
        "cuda_graph": True,
        "dtype": torch.float16,
        "use_terminators": True,
        "temperature": 0.0,
        "random_seed": None,
        "use_enter": False,
        "use_decode_enter": False,
        "use_chat_template": True,
        "model_type": "auto"
    }

def create_temp_config(config: dict, for_server: bool = False) -> dict:
    """Create a temporary config with proper type conversions for serialization"""
    temp_config = config.copy()
    
    # Handle dtype conversion based on context
    if 'dtype' in temp_config:
        if for_server:
            # For server mode, convert torch.dtype to string to avoid JSON serialization issues
            if hasattr(temp_config['dtype'], '__name__'):
                # Convert torch.dtype to string
                temp_config['dtype'] = str(temp_config['dtype']).split('.')[-1]
            elif isinstance(temp_config['dtype'], torch.dtype):
                # Handle torch.dtype objects
                if temp_config['dtype'] == torch.float16:
                    temp_config['dtype'] = 'float16'
                elif temp_config['dtype'] == torch.bfloat16:
                    temp_config['dtype'] = 'bfloat16'
                else:
                    temp_config['dtype'] = str(temp_config['dtype']).split('.')[-1]
        else:
            # For non-server mode, ensure it's a proper torch dtype
            if isinstance(temp_config['dtype'], str):
                if temp_config['dtype'] == 'float16':
                    temp_config['dtype'] = torch.float16
                elif temp_config['dtype'] == 'bfloat16':
                    temp_config['dtype'] = torch.bfloat16
    
    return temp_config

def setup_model_paths(config):
    """Setup and validate model paths"""
    model_path = check_or_download_model(config['model_path'])
    
    # Auto-detect model type if not specified
    if config.get('model_type', 'auto') == 'auto':
        config['model_type'] = detect_model_type(model_path)
        print(f"Auto-detected model type: {config['model_type']}")
    
    draft_model_path = None
    if config.get('draft_model_path'):
        draft_model_path = check_or_download_model(config['draft_model_path'])
        config['apply_speculative'] = True
        print(f"Draft model specified, enabling speculative decoding")
    
    frspec_path = None
    if config.get('frspec_path'):
        if os.path.exists(config['frspec_path']):
            frspec_path = config['frspec_path']
        else:
            print(f"Warning: FRSpec file not found: {config['frspec_path']}")
    
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

def create_model(model_path, draft_model_path, config):
    """Create model instance based on configuration"""
    from .llm import LLM
    from .llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
    from .speculative import LLM_with_eagle
    from .speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
    
    # Handle dtype conversion - convert string to torch dtype if needed
    dtype_value = config['dtype']
    if isinstance(dtype_value, str):
        if dtype_value == 'float16':
            dtype_value = torch.float16
        elif dtype_value == 'bfloat16':
            dtype_value = torch.bfloat16
    
    # Base model arguments
    common_kwargs = {
        'dtype': dtype_value,
        'chunk_length': config['chunk_length'],
        'cuda_graph': config['cuda_graph'],
        'apply_sparse': config.get('apply_sparse', False),
        'sink_window_size': config['sink_window_size'],
        'block_window_size': config['block_window_size'],
        'sparse_topk_k': config['sparse_topk_k'],
        'sparse_switch': config['sparse_switch'],
        'apply_compress_lse': config['apply_compress_lse'],
        'memory_limit': config['memory_limit'],
        'use_enter': config['use_enter'],
        'use_decode_enter': config['use_decode_enter'],
        'temperature': config['temperature'],
        'random_seed': config['random_seed'],
    }
    
    # Speculative decoding arguments
    spec_kwargs = {
        'num_iter': config.get('spec_num_iter', 2),
        'topk_per_iter': config.get('spec_topk_per_iter', 10),
        'tree_size': config.get('spec_tree_size', 12),
        'eagle_window_size': config.get('spec_window_size', 1024),
        'frspec_vocab_size': config.get('frspec_vocab_size', 32768),
        'apply_eagle_quant': config.get('apply_spec_quant', False),
        # Model-specific settings
        'use_rope': config.get('model_type') in ['minicpm', 'minicpm4'],
        'use_input_norm': config.get('model_type') in ['minicpm', 'minicpm4'],
        'use_attn_norm': config.get('model_type') in ['minicpm', 'minicpm4']
    }
    
    # Create model based on configuration
    if config.get('apply_quant', False):
        if config.get('apply_speculative', False) and draft_model_path:
            return W4A16GPTQMarlinLLM_with_eagle(draft_model_path, model_path, **common_kwargs, **spec_kwargs)
        else:
            return W4A16GPTQMarlinLLM(model_path, **common_kwargs)
    else:
        if config.get('apply_speculative', False) and draft_model_path:
            return LLM_with_eagle(draft_model_path, model_path, **common_kwargs, **spec_kwargs)
        else:
            return LLM(model_path, **common_kwargs)

def setup_frspec_vocab(llm, frspec_path):
    """Setup frequency speculative vocabulary for speculative models"""
    if frspec_path and os.path.exists(frspec_path):
        print(f"Loading frequency vocabulary from: {frspec_path}")
        with open(frspec_path, 'rb') as f:
            token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
        llm._load("token_id_remap", token_id_remap, cls="eagle")
        return True
    return False 