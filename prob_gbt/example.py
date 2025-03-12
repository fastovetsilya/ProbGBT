import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split

# Use relative imports for the package
from .data.data_generator import generate_house_prices_dataset
from .prob_gbt import ProbGBT

def main():
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    # Generate synthetic house prices dataset
    print("Generating synthetic house prices dataset...")
    house_prices_df = generate_house_prices_dataset(num_samples=5000, random_seed=42)

    # Split the data into features and target
    X = house_prices_df.drop('house_price', axis=1)
    y = np.array(house_prices_df['house_price'])

    # Define categorical features
    cat_features = ['location_category', 'property_condition']

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Initialize and train the ProbGBT model
    print("\nTraining ProbGBT model...")
    model = ProbGBT(
        num_quantiles=50,
        iterations=500,  # Reduced for faster execution
        subsample=0.8,
        random_seed=42
    )

    model.train(
        X_train, 
        y_train, 
        cat_features=cat_features, 
        eval_set=(X_val, y_val), 
        use_best_model=True, 
        verbose=True
    )

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Predict confidence intervals
    print("\nPredicting confidence intervals...")
    lower_bounds, upper_bounds = model.predict_interval(X_test, confidence_level=0.95)

    # Calculate coverage (percentage of true values within the confidence interval)
    coverage = np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds))
    print(f"95% Confidence Interval Coverage: {coverage:.2%}")

    # Plot predictions vs actual for a subset of test samples
    print("\nPlotting predictions vs actual values...")
    sample_indices = np.random.choice(len(y_test), size=10, replace=False)

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, label='All predictions')
    plt.scatter(y_test[sample_indices], y_pred[sample_indices], color='red', label='Selected samples')

    for i in sample_indices:
        plt.plot([y_test[i], y_test[i]], [lower_bounds[i], upper_bounds[i]], 'r-', alpha=0.7)

    plt.xlabel('Actual House Price')
    plt.ylabel('Predicted House Price')
    plt.title('ProbGBT: Predicted vs Actual House Prices with 95% Confidence Intervals')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print("Saved predictions vs actual plot to ./images/predictions_vs_actual.png")

    # Plot PDF for a single example
    print("\nPlotting probability density function for a single example...")
    sample_idx = sample_indices[0]
    pdfs = model.predict_pdf(X_test.iloc[[sample_idx]])
    x_values, pdf_values = pdfs[0]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, pdf_values, label='Predicted PDF')
    plt.axvline(x=y_test[sample_idx], color='r', linestyle='--', label='Actual value')
    plt.axvline(x=y_pred[sample_idx], color='g', linestyle='--', label='Predicted value')
    plt.fill_between(x_values, pdf_values, where=(x_values >= lower_bounds[sample_idx]) & (x_values <= upper_bounds[sample_idx]), 
                    alpha=0.3, color='blue', label='95% Confidence Interval')
    plt.xlabel('House Price')
    plt.ylabel('Probability Density')
    plt.title(f'Predicted Probability Distribution for Sample {sample_idx}')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/predicted_pdf.png', dpi=300, bbox_inches='tight')
    print("Saved PDF plot to ./images/predicted_pdf.png")

    print("\nExample completed. Check the generated plots in the images directory.")

if __name__ == "__main__":
    # Add the parent directory to the path to make imports work when run directly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main() 