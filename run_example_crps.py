#!/usr/bin/env python3
"""
Run the CRPS evaluation example for ProbGBT.

This script is a wrapper that calls the main CRPS evaluation example 
functionality from the prob_gbt.example_crps module.

The example:
1. Trains models with different numbers of iterations/epochs
2. Evaluates each model using CRPS
3. Plots CRPS values across different epoch counts
4. Identifies the optimal number of iterations based on CRPS
"""

import sys
import os
import warnings

# Suppress all warnings, especially for PyGAM convergence issues
warnings.filterwarnings("ignore", message=".*[Dd]id not converge.*")
warnings.filterwarnings("ignore", message=".*[Cc]onvergence.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pygam")

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the run_crps_example function from the module
from prob_gbt.example_crps import run_crps_example

def run_example_crps():
    """
    Run the main CRPS evaluation example.
    
    This wrapper function calls the implementation in prob_gbt.example_crps,
    allowing the same code to be run both as a standalone script and as 
    an imported module.
    """
    print("=" * 80)
    print("Running ProbGBT CRPS evaluation example")
    print("=" * 80)
    
    # Configuration
    max_iterations = 10000  # Maximum number of iterations to test
    num_points = 1000       # Number of points for CRPS smoothing
    data_path = "./prob_gbt/data/california_housing_prices/housing.csv"
    output_dir = "./images"
    save_dir = "./saved_models"
    output_file = "./iterations_performance.csv"
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Run the main example function with these parameters
    results = run_crps_example(
        max_iterations=max_iterations,
        num_points=num_points,
        data_path=data_path,
        output_dir=output_dir,
        save_dir=save_dir,
        output_file=output_file
    )
    
    # Display a summary of results
    print("\n" + "=" * 80)
    print("CRPS Evaluation Example Summary")
    print("=" * 80)
    print(f"Best model found at {results['best_iterations']} iterations")
    print(f"Best CRPS value: {results['best_crps']:.6f}")
    print(f"Final model RMSE: {results['evaluation']['rmse']:.2f}")
    print(f"Final model 95% confidence interval coverage: {results['evaluation']['coverage']:.2%}")
    print(f"Results saved to {output_file}")
    print(f"Model saved to {save_dir}/best_iteration_model.pkl")
    print(f"Plots saved to {output_dir}/")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    run_example_crps() 