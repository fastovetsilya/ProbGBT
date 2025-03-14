#!/usr/bin/env python3
"""
Simple script to run the ProbGBT example.
"""
import os
import sys
import warnings

# Suppress all PyGAM and convergence warnings
warnings.filterwarnings("ignore", message=".*[Dd]id not converge.*")
warnings.filterwarnings("ignore", message=".*[Cc]onvergence.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pygam")

# Add the current directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main function from the example module
from prob_gbt.example import main

if __name__ == "__main__":
    main() 