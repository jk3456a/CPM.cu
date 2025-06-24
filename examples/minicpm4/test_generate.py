#!/usr/bin/env python3
"""
MiniCPM4 Test Generation Launcher

MiniCPM4-optimized test generation launcher that parses MiniCPM4-specific parameters 
and forwards all other arguments to the core generation module.
"""

import sys
from utils import launch_minicpm4_module


def main():
    """Launch the core generation module with MiniCPM4 parameter processing"""
    return launch_minicpm4_module("cpmcu.generate")


if __name__ == "__main__":
    sys.exit(main()) 