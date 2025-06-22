#!/usr/bin/env python3
"""
CPM.cu Test Generation Launcher

Simple launcher for CPM.cu model generation testing.
This script forwards all arguments to the core generation module.
"""

import sys
import subprocess


def main():
    """Launch the core generation module with all arguments"""
    # Build command to run the core generation module
    cmd = [sys.executable, "-m", "cpmcu.generate"] + sys.argv[1:]
    
    try:
        # Run the core generation module with all arguments passed through
        process = subprocess.run(cmd, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Generation failed: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
