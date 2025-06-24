#!/usr/bin/env python3
"""
CPM.cu Server Launcher

Simple launcher for CPM.cu server startup.
This script forwards all arguments to the core server module.
"""

import sys
import subprocess


def main():
    """Launch the core server module with all arguments"""
    # Build command to run the core server module
    cmd = [sys.executable, "-m", "cpmcu.launch_server"] + sys.argv[1:]
    
    try:
        # Run the core server module with all arguments passed through
        process = subprocess.run(cmd, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Server failed to start: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
        return 0


if __name__ == "__main__":
    sys.exit(main()) 