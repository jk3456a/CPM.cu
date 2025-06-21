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
from cpmcu.args import parse_server_args, display_config_summary

class ServerManager:
    def __init__(self, port=8000, host="localhost", server_args=None):
        self.port = port
        self.host = host
        self.server_process = None
        self.server_args = server_args or []
        

        
    def start_server(self):
        """Start the CPM.cu server"""
        # Build command - pass through all server arguments
        cmd = [
            sys.executable, "-m", "cpmcu.server",
            "--port", str(self.port),
            "--host", self.host
        ]
        
        # Add all other server arguments
        if self.server_args:
            cmd.extend(self.server_args)
        
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
    
    def run_server(self):
        """Start and run the server"""
        try:
            # Start server
            if not self.start_server():
                return False
            
            # Monitor server output
            self.wait_and_monitor()
            
            return True
            
        finally:
            # Always try to stop server
            if self.server_process:
                self.stop_server()

def main():
    # Create a minimal parser for host and port only
    host_port_parser = argparse.ArgumentParser(add_help=False)
    host_port_parser.add_argument('--host', default='0.0.0.0')
    host_port_parser.add_argument('--port', type=int, default=8000)
    
    # Parse known args (host/port) and get remaining args
    host_port_args, server_args = host_port_parser.parse_known_args()
    
    # Also parse with full server parser for config
    args, config = parse_server_args()
    
    # Create server manager
    server_manager = ServerManager(
        port=host_port_args.port,
        host=host_port_args.host,
        server_args=server_args
    )
    
    # Run server
    success = server_manager.run_server()
    
    if success:
        print("\nServer stopped successfully")
        sys.exit(0)
    else:
        print("\nServer startup failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 