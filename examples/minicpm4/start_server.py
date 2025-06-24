#!/usr/bin/env python3
"""
MiniCPM4 Server Launcher

MiniCPM4-optimized server launcher that parses MiniCPM4-specific parameters 
and forwards all other arguments to the core server module.
"""

import sys
from utils import launch_minicpm4_module


def main():
    """Launch the core server module with MiniCPM4 parameter processing"""
    return launch_minicpm4_module("cpmcu.launch_server")


if __name__ == "__main__":
    sys.exit(main()) 