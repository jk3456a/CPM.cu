#!/usr/bin/env python3
"""
MiniCPM4 Specific Configuration

Contains MiniCPM4-specific configurations including YARN factors
and model path conventions.
"""

import torch


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


def apply_minicpm4_yarn_config(llm):
    """Apply MiniCPM4 YARN configuration to model config"""
    yarn_factors = get_minicpm4_yarn_factors()
    
    # Create or modify rope_scaling configuration
    if not hasattr(llm.config, 'rope_scaling') or llm.config.rope_scaling is None:
        llm.config.rope_scaling = {}
        
    llm.config.rope_scaling['rope_type'] = 'longrope'
    llm.config.rope_scaling['long_factor'] = yarn_factors
    llm.config.rope_scaling['short_factor'] = yarn_factors
    print("Applied MiniCPM4 YARN rope_scaling parameters")


def get_minicpm4_model_paths(path_prefix="openbmb", use_quant=True, use_eagle=True, use_eagle_quant=True):
    """Get MiniCPM4 model paths based on configuration
    
    Args:
        path_prefix: Repository prefix (default: 'openbmb')
        use_quant: Use quantized base model
        use_eagle: Use Eagle speculative decoding
        use_eagle_quant: Use quantized Eagle model
    
    Returns:
        dict: Dictionary containing model paths
    """
    paths = {}
    
    # Base model path
    if use_quant:
        paths['model_path'] = f"{path_prefix}/MiniCPM4-8B-marlin-cpmcu"
    else:
        paths['model_path'] = f"{path_prefix}/MiniCPM4-8B"
    
    # Draft model path (Eagle)
    if use_eagle:
        if use_eagle_quant:
            paths['draft_model_path'] = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu"
        else:
            paths['draft_model_path'] = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec"
    
    return paths


def get_minicpm4_default_config(enable_yarn=True, **kwargs):
    """Get MiniCPM4-specific default configuration
    
    Args:
        enable_yarn: Enable YARN rope scaling
        **kwargs: Additional configuration overrides
    
    Returns:
        dict: MiniCPM4 configuration
    """
    config = {
        # MiniCPM4 specific settings
        "model_type": "minicpm4",
        "apply_speculative": True,
        "apply_quant": True,
        "apply_sparse": True,
        "apply_spec_quant": True,
        "enable_yarn": enable_yarn,
        
        # Optimized for MiniCPM4
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
        
        # Generation settings
        "num_generate": 256,
        "chunk_length": 2048,
        "memory_limit": 0.9,
        "cuda_graph": True,
        "dtype": torch.float16,
        "use_terminators": True,
        "temperature": 0.0,
        "use_stream": True,
        "use_enter": False,
        "use_decode_enter": False,
        "random_seed": None,
    }
    
    # Override with provided kwargs
    config.update(kwargs)
    
    return config 