#!/usr/bin/env python3
"""
Script to run the ProbGBT save/load example.
"""
import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the main function from the example_save_load module
from prob_gbt.examples.example_save_load import main

if __name__ == "__main__":
    main() 