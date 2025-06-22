#!/usr/bin/env python3
"""
MiniCPM4 Server Startup Script

Optimized server startup for MiniCPM4 models with YARN support and default configurations.
"""

import argparse
import sys
from pathlib import Path

# Smart import handling - supports both execution modes
if __package__ is None:
    # Direct script execution - add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from minicpm4.default_config import create_minicpm4_config, apply_minicpm4_yarn_config
    from cpmcu.server import launch_server
else:
    # Module execution - use relative import
    from .default_config import create_minicpm4_config, apply_minicpm4_yarn_config
    from ...cpmcu.server import launch_server


def create_minicpm4_server_parser():
    """Create argument parser with MiniCPM4 optimized defaults"""
    parser = argparse.ArgumentParser(description='MiniCPM4 Optimized Server')
    
    # Server configuration
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    # Model configuration with MiniCPM4 defaults
    parser.add_argument('--path-prefix', '--path_prefix', type=str, default='openbmb',
                       help='Model repository prefix (default: openbmb)')
    parser.add_argument('--apply-quant', '--apply_quant', type=lambda x: x.lower() == 'true', default=True,
                       help='Use quantized model (default: True)')
    parser.add_argument('--apply-spec-quant', '--apply_spec_quant', type=lambda x: x.lower() == 'true', default=True,
                       help='Use quantized speculative model (default: True)')
    parser.add_argument('--minicpm4-yarn', '--minicpm4_yarn', type=lambda x: x.lower() == 'true', default=True,
                       help='Enable MiniCPM4 YARN (default: True)')
    parser.add_argument('--apply-sparse', '--apply_sparse', type=lambda x: x.lower() == 'true', default=True,
                       help='Use sparse attention (default: True)')
    
    # Generation parameters
    parser.add_argument('--temperature', '--temp', type=float, default=0.0,
                       help='Generation temperature (default: 0.0)')
    parser.add_argument('--chunk-length', '--chunk_length', type=int, default=2048,
                       help='Chunk length (default: 2048)')
    
    # Optional overrides
    parser.add_argument('--frspec-path', '--frspec_path', type=str, default=None,
                       help='Path to frequency speculative vocabulary file')
    
    return parser


def main():
    """MiniCPM4 server main entry point"""
    parser = create_minicpm4_server_parser()
    args = parser.parse_args()
    
    # Create MiniCPM4 configuration
    config = create_minicpm4_config(
        path_prefix=args.path_prefix,
        apply_quant=args.apply_quant,
        apply_spec_quant=args.apply_spec_quant,
        minicpm4_yarn=args.minicpm4_yarn,
        apply_sparse=args.apply_sparse,
        temperature=args.temperature,
        chunk_length=args.chunk_length
    )
    
    # Add server-specific parameters
    config['host'] = args.host
    config['port'] = args.port
    
    # Add optional FRSpec path
    if args.frspec_path:
        config['frspec_path'] = args.frspec_path
    
    print("=" * 60)
    print("MiniCPM4 Optimized Server Configuration:")
    print("=" * 60)
    print(f"Model: {config['model_path']}")
    print(f"Draft Model: {config['draft_model_path']}")
    print(f"Features: speculative={config['apply_speculative']}, quant={config['apply_quant']}, sparse={config['apply_sparse']}")
    print(f"YARN: {config['minicpm4_yarn']}")
    print(f"Server: {args.host}:{args.port}")
    print("=" * 60)
    
    try:
        print("Starting MiniCPM4 server...")
        # Direct function call instead of subprocess
        launch_server(config)
        return 0
    except Exception as e:
        print(f"Server failed to start: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
        return 0


if __name__ == "__main__":
    sys.exit(main()) 