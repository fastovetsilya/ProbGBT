import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec
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
    y_raw = np.array(housing_df['median_house_value'])
    
    # Apply log transformation to the target variable to ensure positive predictions
    print("Applying log transformation to house prices...")
    y = np.log1p(y_raw)  # log1p is log(1+x) which handles zero values gracefully
    
    # Define categorical features - no need to convert to codes, CatBoost handles text categories
    cat_features = ['ocean_proximity']

    # Split the data into train, validation, and test sets
    # Add index as a column to track samples (preserves data types)
    X = X.reset_index(drop=True)  # Reset index to ensure it starts from 0
    X_index = X.copy()
    X_index['original_index'] = X_index.index
    
    # Split data with indices
    X_train_idx, X_temp_idx, y_train, y_temp = train_test_split(
        X_index, y, test_size=0.3, random_state=1234)
    X_val_idx, X_test_idx, y_val, y_test = train_test_split(
        X_temp_idx, y_temp, test_size=0.5, random_state=1234)
    
    # Extract indices
    train_indices = X_train_idx['original_index'].values
    val_indices = X_val_idx['original_index'].values
    test_indices = X_test_idx['original_index'].values
    
    # Remove index column
    X_train = X_train_idx.drop('original_index', axis=1)
    X_val = X_val_idx.drop('original_index', axis=1)
    X_test = X_test_idx.drop('original_index', axis=1)
    
    # Get the corresponding raw target values
    y_raw_train = y_raw[train_indices]
    y_raw_val = y_raw[val_indices]
    y_raw_test = y_raw[test_indices]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Categorical feature types: {X_train[cat_features].dtypes.to_dict()}")

    # Initialize and train the ProbGBT model
    print("\nTraining ProbGBT model...")
    model = ProbGBT(
        num_quantiles=100,
        iterations=1000,
        subsample=1.0,
        random_seed=1234,
        calibrate=True, 
        train_separate_models=False
    )

    model.train(
        X_train, 
        y_train, 
        cat_features=cat_features, 
        eval_set=(X_val, y_val), 
        calibration_set=None, 
        use_best_model=True, 
        verbose=True
    )

    # Define smoothing method for all predictions
    smoothing_method = 'sample_kde'  # Use GMM smoothing on top of spline for better curves

    # Make predictions in log space
    print("\nMaking predictions...")
    y_pred_log = model.predict(X_test)
    
    # Convert predictions back to original scale
    y_pred = np.expm1(y_pred_log)
    
    # Calculate RMSE and MAE for the point predictions on original scale
    rmse = np.sqrt(np.mean((y_raw_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_raw_test - y_pred))
    print(f"Point predictions from PDF median (back-transformed):")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Predict confidence intervals in log space
    print("\nPredicting confidence intervals...")
    lower_bounds_log, upper_bounds_log = model.predict_interval(X_test, confidence_level=0.95, method=smoothing_method)
    
    # Transform bounds back to original scale
    lower_bounds = np.expm1(lower_bounds_log)
    upper_bounds = np.expm1(upper_bounds_log)
    
    # Plot predictions vs actual for a subset of test samples
    print("\nPlotting predictions vs actual values...")
    sample_indices = np.random.choice(len(y_raw_test), size=10, replace=False)

    plt.figure(figsize=(12, 8))
    plt.scatter(y_raw_test, y_pred, alpha=0.5, label='All predictions')
    plt.scatter(y_raw_test[sample_indices], y_pred[sample_indices], color='red', label='Selected samples')

    for i in sample_indices:
        plt.plot([y_raw_test[i], y_raw_test[i]], [lower_bounds[i], upper_bounds[i]], 'r-', alpha=0.7)

    plt.xlabel('Actual House Price')
    plt.ylabel('Predicted House Price')
    plt.title('ProbGBT: Predicted vs Actual California House Prices with 95% Confidence Intervals')
    plt.plot([y_raw_test.min(), y_raw_test.max()], [y_raw_test.min(), y_raw_test.max()], 'k--', label='Perfect prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print("Saved predictions vs actual plot to ./images/predictions_vs_actual.png")

    # Plot PDF for a single example
    print("\nPlotting probability density function for a single example...")
    sample_idx = sample_indices[0]
    
    pdfs_log = model.predict_pdf(X_test.iloc[[sample_idx]], method=smoothing_method)
    x_values_log, pdf_values_log = pdfs_log[0]
    
    # Transform x-values back to original scale
    x_values = np.expm1(x_values_log)
    
    # The PDF transformation formula is:
    # pdf_new(y) = pdf_old(log1p(y)) * d/dy(log1p(y))
    # where d/dy(log1p(y)) = 1/(1+y)
    pdf_values = pdf_values_log / (1 + x_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, pdf_values, label='Predicted PDF')
    plt.axvline(x=y_raw_test[sample_idx], color='r', linestyle='--', label='Actual value')
    plt.axvline(x=y_pred[sample_idx], color='g', linestyle='--', label='Predicted value')
    plt.fill_between(x_values, pdf_values, 
                    where=(x_values >= lower_bounds[sample_idx]) & (x_values <= upper_bounds[sample_idx]), 
                    alpha=0.3, color='blue', label='95% Confidence Interval')
    
    # Set y-axis limit to improve visualization
    plt.ylim(bottom=0)
    y_max = np.max(pdf_values)
    if y_max > 1e-4:
        plt.ylim(top=min(y_max*1.1, 1e-4))
    
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
    sorted_indices = np.argsort(y_raw_test)
    diverse_indices = [
        sorted_indices[len(sorted_indices) // 10],  # Low price
        sorted_indices[len(sorted_indices) // 2],   # Medium price
        sorted_indices[int(len(sorted_indices) * 0.9)]  # High price
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot each distribution separately with correct data
    for i, idx in enumerate(diverse_indices):
        # Get PDF for this specific sample
        sample_pdfs_log = model.predict_pdf(X_test.iloc[[idx]], method=smoothing_method)
        x_vals_log, pdf_vals_log = sample_pdfs_log[0]
        
        # Transform x-values back to original scale
        x_vals = np.expm1(x_vals_log)
        
        # Adjust PDF values for the change of variable
        pdf_vals = pdf_vals_log / (1 + x_vals)
        
        # Get prediction for this sample
        pred_val = y_pred[idx]
        
        # Plot the distribution and vertical lines
        axes[i].plot(x_vals, pdf_vals, label='PDF')
        axes[i].axvline(x=y_raw_test[idx], color='r', linestyle='--', label='Actual')
        axes[i].axvline(x=pred_val, color='g', linestyle='--', label='Predicted')
        
        # Fill the 95% confidence interval
        axes[i].fill_between(x_vals, pdf_vals, 
                          where=(x_vals >= lower_bounds[idx]) & (x_vals <= upper_bounds[idx]), 
                          alpha=0.3, color='blue', label='95% CI')
        
        # Set y-axis limits for better visualization
        axes[i].set_ylim(bottom=0)
        y_max = np.max(pdf_vals)
        if y_max > 1e-4:
            axes[i].set_ylim(top=min(y_max*1.1, 1e-4))
        
        price_category = "Low" if i == 0 else "Medium" if i == 1 else "High"
        axes[i].set_title(f'{price_category} Price Example')
        axes[i].set_xlabel('House Price')
        if i == 0:
            axes[i].set_ylabel('Probability Density')
        axes[i].grid(True)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('./images/multiple_pdfs.png', dpi=300, bbox_inches='tight')
    print("Saved multiple PDFs plot to ./images/multiple_pdfs.png")
    
    # 2. Confidence interval width vs. prediction error
    print("\nPlotting confidence interval width vs. prediction error...")
    ci_widths = upper_bounds - lower_bounds
    
    # Check for any invalid (negative) confidence interval widths
    invalid_indices = np.where(ci_widths < 0)[0]
    if len(invalid_indices) > 0:
        print(f"WARNING: Found {len(invalid_indices)} samples with negative confidence interval widths!")
        for idx in invalid_indices:
            print(f"Sample {idx}: Lower bound ({lower_bounds[idx]:.2f}) > Upper bound ({upper_bounds[idx]:.2f})")
            print(f"  Actual value: {y_raw_test[idx]:.2f}, Predicted value: {y_pred[idx]:.2f}")
            # Fix by swapping lower and upper bounds for this sample
            lower_bounds[idx], upper_bounds[idx] = upper_bounds[idx], lower_bounds[idx]
        
        # Recalculate widths after fixing
        ci_widths = upper_bounds - lower_bounds
    
    prediction_errors = np.abs(y_raw_test - y_pred)
    
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
    coverage = np.mean((y_raw_test >= lower_bounds) & (y_raw_test <= upper_bounds))
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
        lower_log, upper_log = model.predict_interval(X_test, confidence_level=conf_level, method=smoothing_method)
        
        # Transform bounds back to original scale
        lower = np.expm1(lower_log)
        upper = np.expm1(upper_log)
        
        # Calculate coverage
        coverage = np.mean((y_raw_test >= lower) & (y_raw_test <= upper))
        print(f"Coverage for {conf_level:.2f}: {coverage:.2%}")
        observed_coverages.append(coverage)
    
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
    