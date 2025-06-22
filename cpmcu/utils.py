#!/usr/bin/env python3
"""
CPM.cu Utility Functions

Common utility functions shared across CPM.cu modules.
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
    """Get default configuration"""
    # Return hardcoded default configuration directly
    return {
        "test_minicpm4": True,
        "use_stream": True,
        "apply_eagle": True,
        "apply_quant": True,
        "apply_sparse": True,
        "apply_eagle_quant": True,
        "minicpm4_yarn": True,
        "frspec_vocab_size": 32768,
        "eagle_window_size": 1024,
        "eagle_num_iter": 2,
        "eagle_topk_per_iter": 10,
        "eagle_tree_size": 12,
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
        "use_decode_enter": False
    }

def create_temp_config(config: dict, for_server: bool = False) -> dict:
    """Create a temporary config with proper type conversions for serialization"""
    temp_config = config.copy()
    
    # Handle dtype conversion based on context
    if 'dtype' in temp_config:
        if for_server:
            # For server mode, keep as string to avoid JSON serialization issues
            if hasattr(temp_config['dtype'], '__name__'):
                # Convert torch.dtype to string
                temp_config['dtype'] = str(temp_config['dtype']).split('.')[-1]
        else:
            # For non-server mode, ensure it's a proper torch dtype
            if isinstance(temp_config['dtype'], str):
                if temp_config['dtype'] == 'float16':
                    temp_config['dtype'] = torch.float16
                elif temp_config['dtype'] == 'bfloat16':
                    temp_config['dtype'] = torch.bfloat16
    
    return temp_config

def setup_model_paths(config: dict):
    """Setup model paths from configuration"""
    # Extract model paths from config
    model_path = config.get('model_path')
    draft_model_path = config.get('draft_model_path')
    frspec_path = config.get('frspec_path')
    
    # If not directly specified, derive from path_prefix pattern
    if not model_path:
        model_path, draft_model_path, _, _ = get_model_paths(config.get('path_prefix', 'openbmb'), config)
    
    return model_path, draft_model_path, frspec_path

def check_or_download_model(path):
    """Check if model exists locally, otherwise download from HuggingFace"""
    if os.path.exists(path):
        return path
    else:
        cache_dir = snapshot_download(path)
        return cache_dir

def get_model_paths(path_prefix, config):
    """Get model paths based on configuration"""
    if config['test_minicpm4']:
        if config['apply_eagle_quant']:
            eagle_repo_id = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu"
        else:
            eagle_repo_id = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec"
    else:
        eagle_repo_id = f"{path_prefix}/EAGLE-LLaMA3-Instruct-8B"
    
    if not config['apply_quant']:
        if config['test_minicpm4']:
            base_repo_id = f"{path_prefix}/MiniCPM4-8B"
        else:
            base_repo_id = f"{path_prefix}/Meta-Llama-3-8B-Instruct"
    else:
        base_repo_id = f"{path_prefix}/MiniCPM4-8B-marlin-cpmcu"

    eagle_path = check_or_download_model(eagle_repo_id)
    base_path = check_or_download_model(base_repo_id)
    
    return eagle_path, base_path, eagle_repo_id, base_repo_id

def get_minicpm4_yarn_factors():
    """Get MiniCPM4 YARN factors configuration"""
    return [
        0.9977997200264581, 1.014658295992452, 1.0349680404997148, 1.059429246056193,
        1.0888815016813513, 1.1243301355211495, 1.166977103606075, 1.2182568066927284,
        1.2798772354275727, 1.3538666751582975, 1.4426259039919596, 1.5489853358570191,
        1.6762658237220625, 1.8283407612492941, 2.0096956085876183, 2.225478927469756,
        2.481536379650452, 2.784415934557119, 3.1413289096347365, 3.560047844772632,
        4.048719380066383, 4.752651957515948, 5.590913044973868, 6.584005926629993,
        7.7532214876576155, 9.119754865903639, 10.704443927019176, 12.524994176518703,
        14.59739595363613, 16.93214476166354, 19.53823297353041, 22.417131025031697,
        25.568260840911098, 28.991144156566317, 32.68408069090375, 36.65174474170465,
        40.90396065611201, 45.4664008671033, 50.37147343433591, 55.6804490772103,
        61.470816952306556, 67.8622707390618, 75.00516023410414, 83.11898235973767,
        92.50044360202462, 103.57086856690864, 116.9492274587385, 118.16074567836519,
        119.18497548708795, 120.04810876261652, 120.77352815196981, 121.38182790207875,
        121.89094985353891, 122.31638758099915, 122.6714244963338, 122.9673822552567,
        123.21386397019609, 123.41898278254268, 123.58957065488238, 123.73136519024158,
        123.84917421274221, 123.94701903496814, 124.02825801299717, 124.09569231686116
    ]

def apply_minicpm4_yarn_config(llm, config):
    """Apply MiniCPM4 YARN configuration to model config"""
    yarn_factors = get_minicpm4_yarn_factors()
    
    # Create or modify rope_scaling configuration
    if not hasattr(llm.config, 'rope_scaling') or llm.config.rope_scaling is None:
        llm.config.rope_scaling = {}
        
    llm.config.rope_scaling['rope_type'] = 'longrope'
    llm.config.rope_scaling['long_factor'] = yarn_factors
    llm.config.rope_scaling['short_factor'] = yarn_factors
    print("Forcing MiniCPM4 YARN rope_scaling parameters")

def create_model(eagle_path, base_path, config):
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
    
    common_kwargs = {
        'dtype': dtype_value,
        'chunk_length': config['chunk_length'],
        'cuda_graph': config['cuda_graph'],
        'apply_sparse': config['apply_sparse'],
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
    
    eagle_kwargs = {
        'num_iter': config['eagle_num_iter'],
        'topk_per_iter': config['eagle_topk_per_iter'],
        'tree_size': config['eagle_tree_size'],
        'eagle_window_size': config['eagle_window_size'],
        'frspec_vocab_size': config['frspec_vocab_size'],
        'apply_eagle_quant': config['apply_eagle_quant'],
        'use_rope': config['test_minicpm4'],
        'use_input_norm': config['test_minicpm4'],
        'use_attn_norm': config['test_minicpm4']
    }
    
    if config['apply_quant']:
        if config['apply_eagle']:
            return W4A16GPTQMarlinLLM_with_eagle(eagle_path, base_path, **common_kwargs, **eagle_kwargs)
        else:
            return W4A16GPTQMarlinLLM(base_path, **common_kwargs)
    else:
        if config['apply_eagle']:
            return LLM_with_eagle(eagle_path, base_path, **common_kwargs, **eagle_kwargs)
        else:
            return LLM(base_path, **common_kwargs)

def setup_frspec_vocab(llm, frspec_path):
    """Setup frequency speculative vocabulary for Eagle models"""
    if frspec_path and os.path.exists(frspec_path):
        with open(frspec_path, 'rb') as f:
            token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
        llm._load("token_id_remap", token_id_remap, cls="eagle")
        return frspec_path
    return None 