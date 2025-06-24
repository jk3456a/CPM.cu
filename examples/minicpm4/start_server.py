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
    from minicpm4.utils import create_minicpm4_config, apply_minicpm4_yarn_config, create_minicpm4_server_parser
    from cpmcu.server import launch_server
else:
    # Module execution - use relative import
    from .utils import create_minicpm4_config, apply_minicpm4_yarn_config, create_minicpm4_server_parser
    from ...cpmcu.server import launch_server


def main():
    """MiniCPM4 server main entry point"""
    parser = create_minicpm4_server_parser()
    args = parser.parse_args()
    
    # Create MiniCPM4 configuration using the args
    try:
        config = create_minicpm4_config(
            args=args,
            path_prefix=args.path_prefix,
            minicpm4_yarn=args.minicpm4_yarn
        )
    except Exception as e:
        print(f"Error creating MiniCPM4 configuration: {e}")
        return 1
    
    print("=" * 60)
    print("MiniCPM4 Optimized Server Configuration:")
    print("=" * 60)
    print(f"Model: {config['model_path']}")
    print(f"Draft Model: {config.get('draft_model_path', 'None')}")
    print(f"Features: speculative=auto, quant=auto, sparse={config.get('apply_sparse', False)}")
    print(f"YARN: {config.get('minicpm4_yarn', False)}")
    print(f"Server: {config.get('host', '0.0.0.0')}:{config.get('port', 8000)}")
    print("=" * 60)
    
    try:
        print("Starting MiniCPM4 server...")
        
        # Apply MiniCPM4-specific configurations before server startup
        if config.get('minicpm4_yarn', False):
            # Import and apply YARN configuration callback
            if __package__ is None:
                from minicpm4.utils import create_minicpm4_yarn_callback
            else:
                from .utils import create_minicpm4_yarn_callback
            config['model_init_callback'] = create_minicpm4_yarn_callback()
        
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