import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Use relative imports for the package
from .prob_gbt import ProbGBT

def main():
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    # Load California housing prices dataset
    print("Loading California housing prices dataset...")
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
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1234)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1234)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Initialize and train the ProbGBT model
    print("\nTraining ProbGBT model...")
    model = ProbGBT(
        num_quantiles=100, # Use less quantiles to speed up training
        iterations=500, # Use less iterations to reduce overfitting
        subsample=1.0,
        random_seed=1234,
        calibrate=False,
        train_separate_models=False # Train a single model for all quantiles
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
    lower_bounds, upper_bounds = model.predict_interval(X_test, confidence_level=0.95, method='gmm')
    
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
    plt.title('ProbGBT: Predicted vs Actual California House Prices with 95% Confidence Intervals')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print("Saved predictions vs actual plot to ./images/predictions_vs_actual.png")

    # Plot PDF for a single example
    print("\nPlotting probability density function for a single example...")
    sample_idx = sample_indices[0]
    pdfs = model.predict_pdf(X_test.iloc[[sample_idx]], method='gmm')
    x_values, pdf_values = pdfs[0]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, pdf_values, label='Predicted PDF')
    plt.axvline(x=y_test[sample_idx], color='r', linestyle='--', label='Actual value')
    plt.axvline(x=y_pred[sample_idx], color='g', linestyle='--', label='Predicted value')
    plt.fill_between(x_values, pdf_values, where=(x_values >= lower_bounds[sample_idx]) & (x_values <= upper_bounds[sample_idx]), 
                    alpha=0.3, color='blue', label='95% Confidence Interval')
    
    # Set y-axis limit to 1e-4 if there are values higher than that
    y_max = np.max(pdf_values)
    if y_max > 1e-4:
        plt.ylim(0, 1e-4)
    
    plt.xlabel('House Price')
    plt.ylabel('Probability Density')
    plt.title(f'Predicted Probability Distribution for California House Sample {sample_idx}')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/predicted_pdf.png', dpi=300, bbox_inches='tight')
    print("Saved PDF plot to ./images/predicted_pdf.png")

    # NEW VISUALIZATIONS
    
    # 1. Multiple PDFs with different characteristics
    print("\nPlotting multiple PDFs with different characteristics...")
    # Select samples with different characteristics (low, medium, high prices)
    sorted_indices = np.argsort(y_test)
    diverse_indices = [
        sorted_indices[len(sorted_indices) // 10],  # Low price
        sorted_indices[len(sorted_indices) // 2],   # Medium price
        sorted_indices[int(len(sorted_indices) * 0.9)]  # High price
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Second pass to plot without y-axis limit adjustments
    for i, idx in enumerate(diverse_indices):
        pdfs = model.predict_pdf(X_test.iloc[[idx]], method='gmm')
        x_values, pdf_values = pdfs[0]
        
        axes[i].plot(x_values, pdf_values, label='PDF')
        axes[i].axvline(x=y_test[idx], color='r', linestyle='--', label='Actual')
        axes[i].axvline(x=y_pred[idx], color='g', linestyle='--', label='Predicted')
        axes[i].fill_between(x_values, pdf_values, 
                           where=(x_values >= lower_bounds[idx]) & (x_values <= upper_bounds[idx]), 
                           alpha=0.3, color='blue', label='95% CI')
        
        # Set y-axis limit to 1e-4 if there are values higher than that
        y_max = np.max(pdf_values)
        if y_max > 1e-4:
            axes[i].set_ylim(0, 1e-4)
        
        price_category = "Low" if i == 0 else "Medium" if i == 1 else "High"
        axes[i].set_title(f'{price_category} Price Example')
        axes[i].set_xlabel('House Price')
        if i == 0:
            axes[i].set_ylabel('Probability Density')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('./images/multiple_pdfs.png', dpi=300, bbox_inches='tight')
    print("Saved multiple PDFs plot to ./images/multiple_pdfs.png")
    
    # 2. Confidence interval width vs. prediction error
    print("\nPlotting confidence interval width vs. prediction error...")
    ci_widths = upper_bounds - lower_bounds
    prediction_errors = np.abs(y_test - y_pred)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(ci_widths, prediction_errors, alpha=0.5)
    plt.xlabel('Confidence Interval Width')
    plt.ylabel('Absolute Prediction Error')
    plt.title('Relationship Between Uncertainty and Prediction Error')
    plt.grid(True)
    
    # Add trend line
    z = np.polyfit(ci_widths, prediction_errors, 1)
    p = np.poly1d(z)
    plt.plot(ci_widths, p(ci_widths), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    plt.legend()
    
    plt.savefig('./images/uncertainty_vs_error.png', dpi=300, bbox_inches='tight')
    print("Saved uncertainty vs. error plot to ./images/uncertainty_vs_error.png")
    
    # 3. Feature importance and uncertainty
    print("\nPlotting feature importance and uncertainty relationship...")
    # Get feature importances from the model
    if model.train_separate_models:
        # For separate models, average the feature importances across all models
        feature_importances = np.zeros(X_train.shape[1])
        for q, m in model.trained_models.items():
            feature_importances += m.get_feature_importance()
        feature_importances /= len(model.trained_models)
    else:
        feature_importances = model.model.get_feature_importance()
    
    feature_names = X_train.columns
    
    # Create a grid of subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)  # Changed from 2x2 to 2x3 grid
    
    # Feature importance plot
    ax1 = fig.add_subplot(gs[0, :])
    sorted_idx = np.argsort(feature_importances)
    ax1.barh(np.array(feature_names)[sorted_idx], feature_importances[sorted_idx])
    ax1.set_title('Feature Importance')
    ax1.set_xlabel('Importance')
    
    # Select top 3 important features
    top_features_idx = np.argsort(feature_importances)[-3:]
    top_features = np.array(feature_names)[top_features_idx]
    
    # Plot uncertainty vs. feature value for top features
    for i, feature_idx in enumerate(top_features_idx):
        feature_name = feature_names[feature_idx]
        ax = fig.add_subplot(gs[1, i])
        
        if feature_name in cat_features:
            # For categorical features, calculate average CI width per category
            categories = X_test[feature_name].unique()
            avg_widths = []
            for cat in categories:
                mask = X_test[feature_name] == cat
                if np.sum(mask) > 0:  # Ensure there are samples in this category
                    avg_widths.append(np.mean(ci_widths[mask]))
                else:
                    avg_widths.append(0)
            
            ax.bar(categories, avg_widths)
            ax.set_title(f'Uncertainty vs. {feature_name}')
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Avg. CI Width')
        else:
            # For numerical features, scatter plot
            ax.scatter(X_test[feature_name], ci_widths, alpha=0.5)
            ax.set_title(f'Uncertainty vs. {feature_name}')
            ax.set_xlabel(feature_name)
            ax.set_ylabel('CI Width')
    
    plt.tight_layout()
    plt.savefig('./images/feature_uncertainty.png', dpi=300, bbox_inches='tight')
    print("Saved feature importance and uncertainty plot to ./images/feature_uncertainty.png")
    
    # Calculate coverage from intervals (using the direct interval method)
    print("\nCalculating confidence interval coverage...")
    coverage = np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds))
    print(f"95% Confidence Interval Coverage: {coverage:.2%}")
    
    # 4. Calibration plot - checking if confidence intervals are well-calibrated
    print("\nCreating calibration plot for confidence intervals...")
    # Create confidence levels array with 0.95 included exactly
    confidence_levels_1 = np.linspace(0.1, 0.94, 15)
    confidence_levels_2 = np.linspace(0.96, 0.99, 4)
    confidence_levels = np.concatenate([confidence_levels_1, np.array([0.95]), confidence_levels_2])
    observed_coverages = []
    
    # Use tqdm for a single progress bar, with no extra print output
    print("Calculating coverage for different confidence levels...")
    for conf_level in tqdm(confidence_levels, desc="Evaluating confidence levels"):
        # Get interval bounds for this confidence level using the same method
        lower, upper = model.predict_interval(X_test, confidence_level=conf_level, method='gmm')
        
        # Calculate coverage
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        observed_coverages.append(coverage)
        print(f"Coverage for {conf_level}: {coverage:.2%}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(confidence_levels, observed_coverages, 'o-', label='Observed coverage')
    plt.plot([0, 1], [0, 1], 'k--', label='Ideal calibration')
    
    # Add vertical line at 95% confidence level
    plt.axvline(x=0.95, color='r', linestyle='--', alpha=0.7, label='95% confidence level')
    
    # Find the observed coverage at 95% confidence level (or closest to it)
    idx_95 = np.abs(confidence_levels - 0.95).argmin()
    coverage_at_95 = observed_coverages[idx_95]
    
    # Add horizontal line from the 95% point to the y-axis
    plt.plot([0, 0.95], [coverage_at_95, coverage_at_95], 'r--', alpha=0.7)
    
    # Add annotation for the 95% coverage value
    plt.annotate(f'Coverage: {coverage_at_95:.2%}', 
                 xy=(0.95, coverage_at_95),
                 xytext=(0.7, coverage_at_95 - 0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))
    
    plt.xlabel('Expected Coverage (Confidence Level)')
    plt.ylabel('Observed Coverage')
    plt.title('Calibration Plot for Confidence Intervals')
    
    # Add finer grid
    plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    
    # Set axis limits for better visualization
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('./images/calibration_plot.png', dpi=300, bbox_inches='tight')
    print("Saved calibration plot to ./images/calibration_plot.png")

    print("\nExample completed. Check the generated plots in the images directory.")

if __name__ == "__main__":
    # Add the parent directory to the path to make imports work when run directly
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main() 