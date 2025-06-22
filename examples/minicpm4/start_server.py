#!/usr/bin/env python3
"""
MiniCPM4 Server Startup Script

This script starts the CPM.cu server with MiniCPM4-specific configurations.
"""

import sys
import os
import argparse

# Import from installed cpmcu package
from cpmcu.server import main as server_main
from cpmcu.utils import get_default_config

# Import local config module
from config import get_minicpm4_default_config, get_minicpm4_model_paths, apply_minicpm4_yarn_config


def create_minicpm4_server_parser():
    """Create MiniCPM4-specific server argument parser"""
    parser = argparse.ArgumentParser(description='MiniCPM4 Server')
    
    # Server parameters
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    
    # MiniCPM4 specific parameters
    parser.add_argument('--path-prefix', type=str, default='openbmb',
                       help='HuggingFace model path prefix (default: openbmb)')
    parser.add_argument('--enable-yarn', type=bool, default=True,
                       help='Enable YARN rope scaling for long context (default: True)')
    parser.add_argument('--use-quant', type=bool, default=True,
                       help='Use quantized models (default: True)')
    parser.add_argument('--use-eagle', type=bool, default=True,
                       help='Use Eagle speculative decoding (default: True)')
    parser.add_argument('--use-eagle-quant', type=bool, default=True,
                       help='Use quantized Eagle model (default: True)')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for generation (default: 0.0)')
    parser.add_argument('--num-generate', type=int, default=256,
                       help='Number of tokens to generate (default: 256)')
    
    return parser


def setup_minicpm4_server(args):
    """Setup MiniCPM4 server configuration"""
    # Get MiniCPM4 model paths
    model_paths = get_minicpm4_model_paths(
        path_prefix=args.path_prefix,
        use_quant=args.use_quant,
        use_eagle=args.use_eagle,
        use_eagle_quant=args.use_eagle_quant
    )
    
    # Get MiniCPM4 default configuration
    config = get_minicpm4_default_config(
        enable_yarn=args.enable_yarn,
        temperature=args.temperature,
        num_generate=args.num_generate
    )
    
    # Update with model paths
    config.update(model_paths)
    config['host'] = args.host
    config['port'] = args.port
    
    return config


def main():
    """Main entry point for MiniCPM4 server"""
    parser = create_minicpm4_server_parser()
    args = parser.parse_args()
    
    # Setup MiniCPM4 configuration
    config = setup_minicpm4_server(args)
    
    print("=" * 50)
    print("MiniCPM4 Server Configuration:")
    print("=" * 50)
    print(f"Base Model: {config.get('model_path', 'N/A')}")
    print(f"Draft Model: {config.get('draft_model_path', 'N/A')}")
    print(f"Quantization: Base={config['apply_quant']}, Eagle={config['apply_spec_quant']}")
    print(f"YARN Enabled: {config.get('enable_yarn', False)}")
    print(f"Server: {config['host']}:{config['port']}")
    print("=" * 50)
    
    # Set the global model configuration for cpmcu.server
    import cpmcu.server
    cpmcu.server.model_config = {"config": config}
    
    # Add MiniCPM4-specific lifecycle hook
    original_lifespan = cpmcu.server.lifespan
    
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def minicpm4_lifespan(app):
        """MiniCPM4-enhanced lifespan context manager"""
        # Call original lifespan setup
        async with original_lifespan(app):
            # Apply MiniCPM4 YARN configuration if enabled and model is loaded
            if config.get('enable_yarn', True) and cpmcu.server.model_instance:
                print("Applying MiniCPM4 YARN rope_scaling parameters")
                apply_minicpm4_yarn_config(cpmcu.server.model_instance)
            
            yield
    
    # Replace the lifespan in the app
    cpmcu.server.app.router.lifespan_context = minicpm4_lifespan
    
    # Start the server using uvicorn
    import uvicorn
    uvicorn.run(
        cpmcu.server.app,
        host=config.get('host', '0.0.0.0'),
        port=config.get('port', 8000),
        log_level="info"
    )


if __name__ == "__main__":
    main() 