import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Use relative imports for the package
from .prob_gbt import ProbGBT

def main():
    # Create output directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)

    print("Loading California housing prices dataset...")
    # Load data from CSV file instead of using sklearn's fetch_california_housing()
    housing_df = pd.read_csv("./prob_gbt/data/california_housing_prices/housing.csv")
    
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

    # Example 1: Save and load single model (MultiQuantile approach)
    print("\n--- Example 1: Single Model Approach (MultiQuantile) ---")
    print("Training ProbGBT model with MultiQuantile loss...")
    model_single = ProbGBT(
        num_quantiles=20,
        iterations=100,  # Using fewer iterations for this example
        subsample=1.0,
        random_seed=42,
        calibrate=True,
        train_separate_models=False
    )

    model_single.train(
        X_train, 
        y_train, 
        cat_features=cat_features,  # Added cat_features parameter
        eval_set=(X_val, y_val), 
        use_best_model=True, 
        verbose=True
    )

    # Make predictions with the original model
    y_pred_single_orig = model_single.predict(X_test)
    
    # Save the model
    print("\nSaving single model...")
    single_model_path = "saved_models/single_model.cbm"
    model_single.save(single_model_path)
    print(f"Model saved to {single_model_path}")
    
    # Load the model
    print("\nLoading single model...")
    loaded_model_single = ProbGBT()
    loaded_model_single.load(single_model_path)
    
    # Make predictions with the loaded model
    y_pred_single_loaded = loaded_model_single.predict(X_test)
    
    # Check if predictions are the same
    single_model_diff = np.abs(y_pred_single_orig - y_pred_single_loaded).max()
    print(f"Maximum difference between original and loaded model predictions: {single_model_diff:.10f}")
    
    # Example 2: Save and load separate models
    print("\n--- Example 2: Separate Models Approach ---")
    print("Training ProbGBT model with separate quantile models...")
    model_separate = ProbGBT(
        num_quantiles=50,  # Using fewer quantiles for this example
        iterations=100,     # Using fewer iterations for this example
        subsample=1.0,
        random_seed=42,
        calibrate=True,
        train_separate_models=True
    )

    model_separate.train(
        X_train, 
        y_train, 
        cat_features=cat_features,  # Added cat_features parameter
        eval_set=(X_val, y_val), 
        use_best_model=True, 
        verbose=True
    )

    # Make predictions with the original model
    y_pred_separate_orig = model_separate.predict(X_test)
    
    # Save the model
    print("\nSaving separate models...")
    separate_model_path = "saved_models/separate_models.tar.xz"
    model_separate.save(separate_model_path, compression_level=3)  # Using lower compression for faster example
    print(f"Models saved to {separate_model_path}")
    
    # Load the model
    print("\nLoading separate models...")
    loaded_model_separate = ProbGBT()
    loaded_model_separate.load(separate_model_path)
    
    # Make predictions with the loaded model
    y_pred_separate_loaded = loaded_model_separate.predict(X_test)
    
    # Check if predictions are the same
    separate_model_diff = np.abs(y_pred_separate_orig - y_pred_separate_loaded).max()
    print(f"Maximum difference between original and loaded model predictions: {separate_model_diff:.10f}")
    
    # Plot comparison of predictions
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_single_orig, alpha=0.5, label='Original')
    plt.scatter(y_test, y_pred_single_loaded, alpha=0.5, label='Loaded')
    plt.plot([0, 5], [0, 5], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Single Model: Original vs Loaded')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_separate_orig, alpha=0.5, label='Original')
    plt.scatter(y_test, y_pred_separate_loaded, alpha=0.5, label='Loaded')
    plt.plot([0, 5], [0, 5], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Separate Models: Original vs Loaded')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('saved_models/prediction_comparison.png')
    plt.close()
    
    print("\nExample completed. Check the 'saved_models' directory for the saved models.")
    print("A comparison plot was saved to 'saved_models/prediction_comparison.png'")

if __name__ == "__main__":
    main()
