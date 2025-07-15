"""
Logits Comparison Package

这个包提供了模块化的logits比较工具，用于分析不同投机解码配置的行为差异。

主要模块:
- config: 配置管理
- logits: Logits捕获和生成
- analysis: 数据分析和比较

使用示例:
    from comparison.config import ComparisonConfig
    from comparison.logits import run_generation_with_config
    from comparison.analysis import compare_logits_data
"""

from .config import (
    ComparisonConfig,
    create_comparison_config_parser,
    create_config_from_args,
    create_two_config_comparison_parser,
    create_configs_from_comparison_args
)

from .logits import (
    LogitsCapture,
    patch_model_for_logits_capture,
    run_generation_with_config,
    load_logits_data
)

from .analysis import (
    compare_logits_data,
    analyze_single_capture,
    compare_multiple_captures,
    generate_comparison_report
)

__version__ = "1.0.0"
__author__ = "CPM.cu Team"

__all__ = [
    # config module
    "ComparisonConfig",
    "create_comparison_config_parser", 
    "create_config_from_args",
    "create_two_config_comparison_parser",
    "create_configs_from_comparison_args",
    
    # logits module
    "LogitsCapture",
    "patch_model_for_logits_capture",
    "run_generation_with_config", 
    "load_logits_data",
    
    # analysis module
    "compare_logits_data",
    "analyze_single_capture",
    "compare_multiple_captures",
    "generate_comparison_report"
] 