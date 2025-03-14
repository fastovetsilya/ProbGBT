import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import warnings

# Use relative imports for the package
from .prob_gbt import ProbGBT

def load_data(data_path="./prob_gbt/data/california_housing_prices/housing.csv"):
    """
    Load and prepare the California housing dataset.
    
    Args:
        data_path: Path to the dataset file
        
    Returns:
        X_train_subset, y_train_subset, X_val_subset, y_val_subset, X_test_subset, y_test_subset, 
        X_test, y_test, cat_features
    """
    print("Loading California housing prices dataset...")
    housing_df = pd.read_csv(data_path)
    
    # Handle missing values if any
    housing_df = housing_df.dropna()
    
    # Split the data into features and target
    X = housing_df.drop('median_house_value', axis=1)
    y = np.array(housing_df['median_house_value'])
    
    # Convert 'ocean_proximity' to categorical codes
    X['ocean_proximity'] = X['ocean_proximity'].astype('category').cat.codes
    
    # Define categorical features
    cat_features = ['ocean_proximity']

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Use a reduced dataset for faster training
    print("Using a subset of data for faster training...")
    subset_size = min(5000, len(X_train))
    idx = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_subset = X_train.iloc[idx]
    y_train_subset = y_train[idx]
    
    val_subset_size = min(1000, len(X_val))
    idx = np.random.choice(len(X_val), val_subset_size, replace=False)
    X_val_subset = X_val.iloc[idx]
    y_val_subset = y_val[idx]
    
    test_subset_size = min(1000, len(X_test))
    idx = np.random.choice(len(X_test), test_subset_size, replace=False)
    X_test_subset = X_test.iloc[idx]
    y_test_subset = y_test[idx]
    
    print(f"Using {len(X_train_subset)} training, {len(X_val_subset)} validation, and {len(X_test_subset)} test samples")
    
    return X_train_subset, y_train_subset, X_val_subset, y_val_subset, X_test_subset, y_test_subset, X_test, y_test, cat_features

def train_iteration_models(X_train_subset, y_train_subset, X_val_subset, y_val_subset, X_test_subset, y_test_subset, 
                           cat_features, num_points=10000, max_iterations=10000):
    """
    Train models with different numbers of iterations and evaluate their performance.
    
    Args:
        X_train_subset: Training features
        y_train_subset: Training targets
        X_val_subset: Validation features
        y_val_subset: Validation targets
        X_test_subset: Test features
        y_test_subset: Test targets
        cat_features: List of categorical feature names
        num_points: Number of points for CRPS calculation
        max_iterations: Maximum number of iterations to test
        
    Returns:
        iteration_values, crps_values, rmse_values, best_iterations, best_crps
    """
    # Define iteration values - even distribution up to max_iterations
    if max_iterations >= 10000:
        iteration_values = [10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    elif max_iterations >= 5000:
        iteration_values = [10, 100, 500, 1000, 2000, 3000, 4000, 5000]
    elif max_iterations >= 3000:
        iteration_values = [10, 100, 500, 1000, 1500, 2000, 2500, 3000]
    else:
        iteration_values = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000]
    
    # Keep only iterations up to max_iterations
    iteration_values = [i for i in iteration_values if i <= max_iterations]
    
    # Store results
    crps_values = []
    rmse_values = []
    
    print(f"\n--- Training models with different numbers of iterations (up to {max_iterations}) ---")
    for iterations in tqdm(iteration_values, desc="Training models with varied iterations"):
        # Create and train model
        model = ProbGBT(
            num_quantiles=20,    # Good quantile coverage
            iterations=iterations,
            learning_rate=0.03,  # Lower learning rate for stability with high iterations
            subsample=0.8,
            random_seed=42
        )
        
        model.train(
            X_train_subset, 
            y_train_subset, 
            cat_features=cat_features, 
            eval_set=(X_val_subset, y_val_subset), 
            use_best_model=False,  # Disabled model shrinking
            verbose=False,  # Less verbose for multiple runs
            early_stopping_rounds=None  # No early stopping
        )
        
        # Evaluate with CRPS
        crps = model.evaluate_crps(X_test_subset, y_test_subset, num_points=num_points, verbose=False)
        crps_values.append(crps)
        
        # Evaluate with RMSE
        y_pred = model.predict(X_test_subset)
        rmse = np.sqrt(np.mean((y_test_subset - y_pred) ** 2))
        rmse_values.append(rmse)
        
        print(f"Iterations: {iterations}, CRPS: {crps:.6f}, RMSE: {rmse:.2f}")
    
    # Find best iteration based on CRPS
    best_idx = np.argmin(crps_values)
    best_iterations = iteration_values[best_idx]
    best_crps = crps_values[best_idx]
    
    return iteration_values, crps_values, rmse_values, best_iterations, best_crps

def plot_results(iteration_values, crps_values, rmse_values, best_iterations, best_crps, output_dir="./images"):
    """
    Plot the results of the iteration study.
    
    Args:
        iteration_values: List of iteration values tested
        crps_values: List of CRPS values for each iteration
        rmse_values: List of RMSE values for each iteration
        best_iterations: The optimal number of iterations found
        best_crps: The best CRPS value found
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best iteration based on RMSE
    best_rmse_idx = np.argmin(rmse_values)
    best_rmse_iterations = iteration_values[best_rmse_idx]
    best_rmse = rmse_values[best_rmse_idx]
    
    # Plot CRPS vs iterations - linear scale only
    plt.figure(figsize=(15, 8))
    plt.plot(iteration_values, crps_values, '-o', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('CRPS (lower is better)', fontsize=14)
    plt.title(f'CRPS vs Number of Iterations (up to {iteration_values[-1]})', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Ensure all iteration labels are visible
    plt.xticks(iteration_values, [str(it) for it in iteration_values], rotation=45)
    
    plt.axvline(x=best_iterations, color='r', linestyle='--')
    plt.axhline(y=best_crps, color='r', linestyle='--')
    plt.annotate(f'Best: {best_iterations} iterations\nCRPS={best_crps:.6f}', 
                xy=(best_iterations, best_crps),
                xytext=(best_iterations*1.1, best_crps*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/crps_vs_iterations.png', dpi=300, bbox_inches='tight')
    print(f"Saved CRPS vs iterations plot to {output_dir}/crps_vs_iterations.png")
    
    # Plot RMSE vs iterations - linear scale only
    plt.figure(figsize=(15, 8))
    plt.plot(iteration_values, rmse_values, '-o', linewidth=2, markersize=8, color='green')
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('RMSE (lower is better)', fontsize=14)
    plt.title(f'RMSE vs Number of Iterations (up to {iteration_values[-1]})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(iteration_values, [str(it) for it in iteration_values], rotation=45)
    
    plt.axvline(x=best_rmse_iterations, color='r', linestyle='--')
    plt.axhline(y=best_rmse, color='r', linestyle='--')
    plt.annotate(f'Best: {best_rmse_iterations} iterations\nRMSE={best_rmse:.2f}', 
                xy=(best_rmse_iterations, best_rmse),
                xytext=(best_rmse_iterations*1.1, best_rmse*1.05),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rmse_vs_iterations.png', dpi=300, bbox_inches='tight')
    print(f"Saved RMSE vs iterations plot to {output_dir}/rmse_vs_iterations.png")
    
    # Plot both metrics together (normalized) - linear scale only
    plt.figure(figsize=(15, 8))
    
    # Normalize values for comparison
    crps_norm = np.array(crps_values) / max(crps_values)
    rmse_norm = np.array(rmse_values) / max(rmse_values)
    
    plt.plot(iteration_values, crps_norm, '-o', linewidth=2, markersize=8, label='Normalized CRPS')
    plt.plot(iteration_values, rmse_norm, '-s', linewidth=2, markersize=8, label='Normalized RMSE')
    
    plt.axvline(x=best_iterations, color='blue', linestyle='--', alpha=0.7,
                label=f'Best CRPS: {best_iterations} iterations')
    plt.axvline(x=best_rmse_iterations, color='green', linestyle='--', alpha=0.7,
                label=f'Best RMSE: {best_rmse_iterations} iterations')
    
    plt.xlabel('Number of Iterations', fontsize=14)
    plt.ylabel('Normalized Score (lower is better)', fontsize=14)
    plt.title(f'Comparison of CRPS and RMSE vs Number of Iterations (up to {iteration_values[-1]})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(iteration_values, [str(it) for it in iteration_values], rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/crps_rmse_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison plot to {output_dir}/crps_rmse_comparison.png")
    print(f"Plot shows iterations: {iteration_values}")
    
    # Return a summary of the results
    return {
        "best_crps_iterations": best_iterations,
        "best_crps_value": best_crps,
        "best_rmse_iterations": best_rmse_iterations,
        "best_rmse_value": best_rmse
    }

def train_final_model(X_train_subset, y_train_subset, X_val_subset, y_val_subset, 
                     cat_features, best_iterations, save_dir="./saved_models"):
    """
    Train a final model with the optimal number of iterations.
    
    Args:
        X_train_subset: Training features
        y_train_subset: Training targets
        X_val_subset: Validation features
        y_val_subset: Validation targets
        cat_features: List of categorical feature names
        best_iterations: The optimal number of iterations to use
        save_dir: Directory to save the model
        
    Returns:
        The trained model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- Training final model with optimal iterations ({best_iterations}) based on CRPS ---")
    final_model = ProbGBT(
        num_quantiles=50,   # Full number of quantiles for better results
        iterations=best_iterations,
        learning_rate=0.03,
        subsample=0.8,
        random_seed=42
    )
    
    final_model.train(
        X_train_subset, 
        y_train_subset, 
        cat_features=cat_features, 
        eval_set=(X_val_subset, y_val_subset), 
        use_best_model=False,  # Disabled model shrinking
        verbose=True
    )
    
    # Save the best model
    try:
        model_path = f"{save_dir}/best_iteration_model.pkl"
        final_model.save(model_path)
        print(f"Saved best iterations model to {model_path}")
    except Exception as e:
        print(f"Error saving best iterations model: {str(e)}")
    
    return final_model

def save_results(iteration_values, crps_values, rmse_values, output_file="./iterations_performance.csv"):
    """
    Save the results to a CSV file.
    
    Args:
        iteration_values: List of iteration values tested
        crps_values: List of CRPS values for each iteration
        rmse_values: List of RMSE values for each iteration
        output_file: Path to the output CSV file
    """
    results_df = pd.DataFrame({
        'Iterations': iteration_values,
        'CRPS': crps_values,
        'RMSE': rmse_values
    })
    results_df.to_csv(output_file, index=False)
    print(f"Saved performance metrics to {output_file}")
    return results_df

def evaluate_model(model, X_test, y_test, num_points=10000):
    """
    Evaluate a model with CRPS and RMSE.
    
    Args:
        model: The model to evaluate
        X_test: Test features
        y_test: Test targets
        num_points: Number of points for CRPS calculation
        
    Returns:
        Dictionary with evaluation metrics
    """
    # CRPS with more points for better accuracy
    print("Evaluating model with CRPS...")
    crps = model.evaluate_crps(X_test, y_test, subset_fraction=0.2, num_points=num_points, verbose=True)
    
    # RMSE
    print("Evaluating model with RMSE...")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    
    # Confidence intervals
    print("Calculating confidence intervals...")
    lower_bounds, upper_bounds = model.predict_interval(X_test, confidence_level=0.95)
    coverage = np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds))
    
    return {
        "crps": crps,
        "rmse": rmse,
        "coverage": coverage,
        "predictions": y_pred,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds
    }

def run_crps_example(max_iterations=10000, num_points=10000, data_path="./prob_gbt/data/california_housing_prices/housing.csv",
                    output_dir="./images", save_dir="./saved_models", output_file="./iterations_performance.csv"):
    """
    Run the complete CRPS evaluation example.
    
    Args:
        max_iterations: Maximum number of iterations to test
        num_points: Number of points for CRPS calculation
        data_path: Path to the dataset file
        output_dir: Directory to save the plots
        save_dir: Directory to save the model
        output_file: Path to the output CSV file
        
    Returns:
        Dictionary with results and the trained model
    """
    # Suppress warnings
    warnings.filterwarnings("ignore", message=".*[Dd]id not converge.*")
    warnings.filterwarnings("ignore", message=".*[Cc]onvergence.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pygam")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    X_train_subset, y_train_subset, X_val_subset, y_val_subset, X_test_subset, y_test_subset, X_test, y_test, cat_features = load_data(data_path)
    
    # Train models with different iterations
    iteration_values, crps_values, rmse_values, best_iterations, best_crps = train_iteration_models(
        X_train_subset, y_train_subset, X_val_subset, y_val_subset, X_test_subset, y_test_subset, 
        cat_features, num_points=num_points, max_iterations=max_iterations
    )
    
    # Plot results
    plot_summary = plot_results(iteration_values, crps_values, rmse_values, best_iterations, best_crps, output_dir)
    
    # Save results to CSV
    results_df = save_results(iteration_values, crps_values, rmse_values, output_file)
    
    # Train final model
    final_model = train_final_model(X_train_subset, y_train_subset, X_val_subset, y_val_subset, 
                                   cat_features, best_iterations, save_dir)
    
    # Evaluate final model
    evaluation = evaluate_model(final_model, X_test, y_test, num_points)
    
    print("\n--- Final Model Evaluation ---")
    print(f"CRPS: {evaluation['crps']:.6f}")
    print(f"RMSE: {evaluation['rmse']:.2f}")
    print(f"95% Confidence Interval Coverage: {evaluation['coverage']:.2%}")
    
    # Summary and recommendations
    print("\n--- Summary ---")
    print(f"Best iterations for CRPS: {best_iterations} (CRPS: {best_crps:.6f})")
    print(f"Best iterations for RMSE: {plot_summary['best_rmse_iterations']} (RMSE: {plot_summary['best_rmse_value']:.2f})")
    
    if best_iterations == iteration_values[-1]:
        print("\nNote: The optimal number of iterations for CRPS is at the maximum tested value.")
        print(f"Consider testing with more than {max_iterations} iterations to find the true optimum.")
    
    print("\nExperiment complete!")
    
    return {
        "best_iterations": best_iterations,
        "best_crps": best_crps,
        "evaluation": evaluation,
        "model": final_model,
        "results_df": results_df
    }

def main():
    """
    Main function to run the example.
    """
    return run_crps_example()

if __name__ == "__main__":
    main() 