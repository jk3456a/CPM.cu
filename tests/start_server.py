#!/usr/bin/env python3
"""
CPM.cu Server Startup Script

This script starts the CPM.cu server with configurable parameters.
Handles all configuration and argument processing before launching the server.
"""

import subprocess
import time
import sys
import os
import signal
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from cpmcu.utils import (
    get_default_config,
    check_or_download_model,
    get_model_paths,
    get_minicpm4_yarn_factors
)

def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(description='Start CPM.cu OpenAI API Server with configurable parameters')
    
    # Server arguments
    parser.add_argument('--port', type=int, default=8000, 
                       help='Server port (default: 8000)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Server host (default: localhost)')
    
    # Basic arguments
    parser.add_argument('--path-prefix', '--path_prefix', '-p', type=str, default='openbmb', 
                        help='Path prefix for model directories, you can use openbmb to download models, or your own path (default: openbmb)')

    # Model configuration boolean arguments
    parser.add_argument('--test-minicpm4', '--test_minicpm4', action='store_true',
                        help='Use MiniCPM4 model')
    parser.add_argument('--no-test-minicpm4', '--no_test_minicpm4', action='store_false', dest='test_minicpm4',
                        help='Do not use MiniCPM4 model')
    parser.add_argument('--apply-eagle', '--apply_eagle', action='store_true',
                        help='Use Eagle speculative decoding')
    parser.add_argument('--no-apply-eagle', '--no_apply_eagle', action='store_false', dest='apply_eagle',
                        help='Do not use Eagle speculative decoding')
    parser.add_argument('--apply-quant', '--apply_quant', action='store_true',
                        help='Use quantized model')
    parser.add_argument('--no-apply-quant', '--no_apply_quant', action='store_false', dest='apply_quant',
                        help='Do not use quantized model')
    parser.add_argument('--apply-sparse', '--apply_sparse', action='store_true',
                        help='Use sparse attention')
    parser.add_argument('--no-apply-sparse', '--no_apply_sparse', action='store_false', dest='apply_sparse',
                        help='Do not use sparse attention')
    parser.add_argument('--apply-eagle-quant', '--apply_eagle_quant', action='store_true',
                        help='Use quantized Eagle model')
    parser.add_argument('--no-apply-eagle-quant', '--no_apply_eagle_quant', action='store_false', dest='apply_eagle_quant',
                        help='Do not use quantized Eagle model')
    parser.add_argument('--apply-compress-lse', '--apply_compress_lse', action='store_true',
                        help='Apply LSE compression, only support on sparse attention, this will compress the stage 1 kv twice for LSE pre-computing')
    parser.add_argument('--no-apply-compress-lse', '--no_apply_compress_lse', action='store_false', dest='apply_compress_lse',
                        help='Do not apply LSE compression')
    parser.add_argument('--cuda-graph', '--cuda_graph', action='store_true',
                        help='Use CUDA graph optimization')
    parser.add_argument('--no-cuda-graph', '--no_cuda_graph', action='store_false', dest='cuda_graph',
                        help='Do not use CUDA graph optimization')
    parser.add_argument('--use-teminators', '--use_terminators', action='store_true',
                        help='Use teminators, if not specified, the generation will not be interrupted')
    parser.add_argument('--no-use-teminators', '--no_use_terminators', action='store_false', dest='use_terminators',
                        help='Do not use teminators')
    parser.add_argument('--minicpm4-yarn', '--minicpm4_yarn', action='store_true',
                        help='Use MiniCPM4 YARN, this is for very long context, such as > 32/64k tokens')
    parser.add_argument('--no-minicpm4-yarn', '--no_minicpm4_yarn', action='store_false', dest='minicpm4_yarn',
                        help='Do not use MiniCPM4 YARN')
    parser.add_argument('--use-enter', '--use_enter', action='store_true',
                        help='Use enter to generate')
    parser.add_argument('--no-use-enter', '--no_use_enter', action='store_false', dest='use_enter',
                        help='Do not use enter to generate')
    parser.add_argument('--use-decode-enter', '--use_decode_enter', action='store_true',
                        help='Use enter before decode phase')
    parser.add_argument('--no-use-decode-enter', '--no_use_decode_enter', action='store_false', dest='use_decode_enter',
                        help='Do not use enter before decode phase')

    # Model configuration numeric arguments
    parser.add_argument('--frspec-vocab-size', '--frspec_vocab_size', type=int, default=None,
                        help='Frequent speculation vocab size (default: 32768)')
    parser.add_argument('--eagle-window-size', '--eagle_window_size', type=int, default=None,
                        help='Eagle window size (default: 1024)')
    parser.add_argument('--eagle-num-iter', '--eagle_num_iter', type=int, default=None,
                        help='Eagle number of iterations (default: 2)')
    parser.add_argument('--eagle-topk-per-iter', '--eagle_topk_per_iter', type=int, default=None,
                        help='Eagle top-k per iteration (default: 10)')
    parser.add_argument('--eagle-tree-size', '--eagle_tree_size', type=int, default=None,
                        help='Eagle tree size (default: 12)')
    parser.add_argument('--sink-window-size', '--sink_window_size', type=int, default=None,
                        help='Sink window size of sparse attention (default: 1)')
    parser.add_argument('--block-window-size', '--block_window_size', type=int, default=None,
                        help='Block window size of sparse attention (default: 8)')
    parser.add_argument('--sparse-topk-k', '--sparse_topk_k', type=int, default=None,
                        help='Sparse attention top-k (default: 64)')
    parser.add_argument('--sparse-switch', '--sparse_switch', type=int, default=None,
                        help='Context length of dense and sparse attention switch (default: 1)')
    parser.add_argument('--chunk-length', '--chunk_length', type=int, default=None,
                        help='Chunk length for prefilling (default: 2048)')
    parser.add_argument('--memory-limit', '--memory_limit', type=float, default=None,
                        help='Memory limit for use (default: 0.9)')
    parser.add_argument('--temperature', '--temperature', type=float, default=None,
                        help='Temperature for processing (default: 0.0)')
    parser.add_argument('--dtype', type=str, default=None, choices=['float16', 'bfloat16'],
                        help='Model dtype (default: float16)')
    parser.add_argument('--random-seed', '--random_seed', type=int, default=None,
                        help='Random seed for processing (default: None)')
    
    return parser

def parse_and_merge_config(default_config):
    """Parse arguments and merge with default configuration"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Use default configuration
    config = default_config.copy()
    print("Using default configuration")
    
    # Set default values to None for boolean arguments that weren't specified
    bool_args = [key for key, value in config.items() if isinstance(value, bool)]
    for arg in bool_args:
        # Convert underscores to hyphens for command line argument names
        arg_hyphen = arg.replace('_', '-')
        # Check for both formats (hyphen and underscore)
        arg_specified = (f'--{arg_hyphen}' in sys.argv or f'--no-{arg_hyphen}' in sys.argv or
                        f'--{arg}' in sys.argv or f'--no-{arg}' in sys.argv)
        if not arg_specified:
            setattr(args, arg, None)

    # Define parameter mappings for automatic override (exclude dtype which needs special handling)
    auto_override_params = [key for key in config.keys() if key != 'dtype']

    # Override config values if arguments are provided
    for param in auto_override_params:
        arg_value = getattr(args, param, None)
        if arg_value is not None:
            config[param] = arg_value

    # Handle dtype separately due to type conversion
    if args.dtype is not None:
        config['dtype'] = args.dtype  # Keep as string for server arguments
    
    # Process model paths
    print("Processing model paths...")
    try:
        eagle_path, base_path, eagle_repo_id, base_repo_id = get_model_paths(args.path_prefix, config)
        config['eagle_path'] = eagle_path
        config['base_path'] = base_path
        config['eagle_repo_id'] = eagle_repo_id
        config['base_repo_id'] = base_repo_id
        print(f"Eagle model path: {eagle_path}")
        print(f"Base model path: {base_path}")
    except Exception as e:
        print(f"Error processing model paths: {e}")
        sys.exit(1)
    
    # Process MiniCPM4 YARN configuration if enabled
    if config['test_minicpm4'] and config['minicpm4_yarn']:
        print("Adding MiniCPM4 YARN configuration...")
        config['yarn_factors'] = get_minicpm4_yarn_factors()
    
    # Process frequency speculative vocabulary if enabled
    if config['apply_eagle'] and config['frspec_vocab_size'] > 0:
        print(f"Processing frequency speculative vocabulary (size: {config['frspec_vocab_size']})...")
        fr_path = f'{eagle_path}/freq_{config["frspec_vocab_size"]}.pt'
        if not os.path.exists(fr_path):
            print(f"Downloading frequency vocabulary file...")
            cache_dir = snapshot_download(
                eagle_repo_id,
                ignore_patterns=["*.bin", "*.safetensors"],
            )
            fr_path = os.path.join(cache_dir, f'freq_{config["frspec_vocab_size"]}.pt')
        config['frspec_vocab_path'] = fr_path
        print(f"Frequency vocabulary path: {fr_path}")
    
    return args, config

def print_config(config):
    """Print all configuration parameters"""
    print("=" * 50)
    print("Server Configuration Parameters:")
    print("=" * 50)
    print(f"Features: eagle={config['apply_eagle']}, quant={config['apply_quant']}, sparse={config['apply_sparse']}")
    print(f"Generation: chunk_length={config['chunk_length']}, use_terminators={config['use_terminators']}")
    print(f"Sampling: temperature={config['temperature']}, random_seed={config['random_seed']}")
    print(f"Demo: use_enter={config['use_enter']}, use_decode_enter={config['use_decode_enter']}")
    print(f"Others: dtype={config['dtype']}, minicpm4_yarn={config['minicpm4_yarn']}, cuda_graph={config['cuda_graph']}, memory_limit={config['memory_limit']}")
    if config['apply_sparse']:
        print(f"Sparse Attention: sink_window={config['sink_window_size']}, block_window={config['block_window_size']}, sparse_topk_k={config['sparse_topk_k']}, sparse_switch={config['sparse_switch']}, compress_lse={config['apply_compress_lse']}")
    if config['apply_eagle']:
        print(f"Eagle: eagle_num_iter={config['eagle_num_iter']}, eagle_topk_per_iter={config['eagle_topk_per_iter']}, eagle_tree_size={config['eagle_tree_size']}, apply_eagle_quant={config['apply_eagle_quant']}, window_size={config['eagle_window_size']}, frspec_vocab_size={config['frspec_vocab_size']}")
    print("=" * 50)
    print()

class ServerManager:
    def __init__(self, port=8000, host="localhost"):
        self.port = port
        self.host = host
        self.server_process = None
        

        
    def start_server(self, config=None):
        """Start the CPM.cu server"""
        print(f"Starting CPM.cu server on {self.host}:{self.port}...")
        
        # Build command
        cmd = [
            sys.executable, "-m", "cpmcu.server",
            "--port", str(self.port),
            "--host", self.host
        ]
        
        # Note: Configuration will use server's built-in defaults
        # since --config-file support has been removed from tests
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            print("Server is starting... Press Ctrl+C to stop")
            return True
            
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def wait_and_monitor(self):
        """Monitor server process and output logs"""
        try:
            # Print server output in real time
            while True:
                output = self.server_process.stdout.readline()
                if output == '' and self.server_process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    
        except KeyboardInterrupt:
            print("\nReceived interrupt signal, stopping server...")
            self.stop_server()
        except Exception as e:
            print(f"Error monitoring server: {e}")
            self.stop_server()
    
    def stop_server(self):
        """Stop the server process"""
        if self.server_process:
            print(f"Stopping server process (PID: {self.server_process.pid})...")
            
            try:
                # Try graceful shutdown first
                self.server_process.terminate()
                try:
                    self.server_process.wait(timeout=10)
                    print("Server stopped gracefully")
                except subprocess.TimeoutExpired:
                    print("Server didn't stop gracefully, forcing kill...")
                    self.server_process.kill()
                    self.server_process.wait()
                    print("Server killed")
            except Exception as e:
                print(f"Error stopping server: {e}")
            
            self.server_process = None
    
    def run_server(self, config=None):
        """Start and run the server"""
        print("=" * 60)
        print("CPM.cu Server Startup")
        print("=" * 60)
        
        try:
            # Start server
            if not self.start_server(config):
                return False
            
            # Monitor server output
            self.wait_and_monitor()
            
            return True
            
        finally:
            # Always try to stop server
            if self.server_process:
                self.stop_server()

def main():
    # Parse arguments and merge with default config
    args, config = parse_and_merge_config(get_default_config())
    
    # Print configuration
    print_config(config)
    
    # Create server manager
    server_manager = ServerManager(
        port=args.port,
        host=args.host
    )
    
    # Run server with processed config
    success = server_manager.run_server(config)
    
    if success:
        print("\nServer stopped successfully")
        sys.exit(0)
    else:
        print("\nServer startup failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 