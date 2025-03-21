# ProbGBT: Probabilistic Gradient Boosted Trees

ProbGBT is a fully nonparametric probabilistic machine learning model that extends gradient boosted trees to provide uncertainty estimates. It is built on top of quantile regression functionality provided by Catboost package.

First, the model is trained for multiple quantiles. At this step, we implement a non-uniform quantile selection technique to reduce the number of quantiles and improve stability on the tails of the distributions. Then, we reconstruct a probability density function (PDF) from these quantiles without making any assumptions about the underlying distribution shape. To achieve this, we propose three different methods. By default, the model part of the training dataset to calibrate the distributions to achieve reliable coverage for intervals with different confidence levels.

The main goal of this project is to create a tool that is easy to use and flexible enough to be adapted for various datasets. Instead of trying to build the perfect model for a particular problem, we emphasize simplicity as a way to design tools to be used with autonomous LLM agents. Although customizable, the model is ready to be used with default parameters.

## Features

- **Uncertainty Estimation**: Provides smooth or detailed probability distributions
- **Non-parametric**: Makes no assumptions about the shape of the target distribution
- **Flexible**: Works with both numerical and categorical features, can be used out of box with default parameters
- **Multiple Training Strategies**: Supports both single model with MultiQuantile loss and separate models for each quantile
- **Calibrated Predictions**: Uses conformal prediction to ensure statistically valid confidence intervals

## Example Visualizations

### Predictions with Confidence Intervals

![Predictions vs Actual with Confidence Intervals](prob_gbt/examples/images/predictions_vs_actual.png)

*Scatter plot showing predicted vs actual values with 95% confidence intervals for selected samples.*

### Probability Density Functions

<table>
  <tr>
    <td><img src="prob_gbt/examples/images/predicted_pdf.png" alt="Single PDF Example" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="prob_gbt/examples/images/multiple_pdfs.png" alt="Multiple PDFs for Different Price Ranges" width="100%"/></td>
  </tr>
</table>

*Top: Detailed probability density function for a single example, showing the predicted distribution, actual value, and 95% confidence interval.*

*Bottom: PDFs for houses in different price ranges (low, medium, high), demonstrating how the model captures different uncertainty patterns.*

### Uncertainty Analysis

![Uncertainty vs Error](prob_gbt/examples/images/uncertainty_vs_error.png)

*Relationship between confidence interval width and prediction error, showing how larger uncertainty correlates with larger errors.*

![Feature Importance and Uncertainty](prob_gbt/examples/images/feature_uncertainty.png)

*Feature importance and its relationship with prediction uncertainty, demonstrating which features contribute most to the model's predictions and uncertainty.*

### Calibration Analysis

<table>
  <tr>
    <td><img src="prob_gbt/examples/images/calibration_plot.png" alt="Calibration Plot" width="100%"/></td>
  </tr>
  <tr>
    <td><img src="prob_gbt/examples/images/calibration_error_plot.png" alt="Calibration Error Plot" width="100%"/></td>
  </tr>
</table>

*Top: Calibration plot showing the relationship between expected coverage (confidence level) and observed coverage. The blue line represents the observed coverage at different confidence levels, while the dashed black line indicates perfect calibration. The plot also shows how confidence interval width increases with confidence level.*

*Bottom: Calibration error plot showing the difference between observed and expected coverage across confidence levels. Points highlight key confidence levels (50%, 80%, 90%, 95%, 99%), and the green band indicates an acceptable error range of ±2%. This plot helps identify where the model might be over-confident (negative values) or under-confident (positive values).*

The calibration analysis is crucial for assessing the reliability of the model's uncertainty estimates:

1. **Perfect calibration** would follow the diagonal line in the calibration plot, meaning that for any confidence level α, exactly α% of the true values fall within the predicted intervals.

2. **Over-confidence** occurs when the blue line falls below the diagonal, indicating that the predicted intervals are too narrow and contain fewer true values than expected.

3. **Under-confidence** occurs when the blue line is above the diagonal, indicating that the predicted intervals are unnecessarily wide.

4. **The calibration error plot** makes it easier to identify miscalibration patterns across different confidence levels, with values closer to zero indicating better calibration.

5. **Confidence interval width** (shown in the bottom part of the calibration plot) typically increases as the confidence level increases, but the rate of increase can reveal patterns in the model's uncertainty estimation.

ProbGBT's conformal calibration mechanism aims to ensure that the calibration curve stays close to the diagonal, providing statistically valid uncertainty estimates.

### Geospatial Probabilistic Predictions

ProbGBT provides fully nonparametric probabilistic predictions for geospatial data, enabling smooth uncertainty quantification across geographic regions. Below is an example with California housing prices where the model is trained using only latitude and longitude as features:

<img src="prob_gbt/examples/images/california_prices_map.png" alt="Predicted House Prices Map" width="80%"/>

*Geographic distribution of predicted house prices across California.*

<img src="prob_gbt/examples/images/california_uncertainty_map.png" alt="Prediction Uncertainty Map" width="80%"/>

*Relative uncertainty (95% CI width / predicted price) showing which regions have more uncertain predictions.*

<img src="prob_gbt/examples/images/california_absolute_uncertainty_map.png" alt="Absolute Uncertainty Map" width="80%"/>

*Absolute uncertainty (95% CI width in dollars) highlighting areas with the widest prediction intervals.*

<img src="prob_gbt/examples/images/california_combined_map.png" alt="Combined Visualization" width="100%"/>

*Combined visualization showing all three metrics for easier comparison.*


## Quick Start

### Installation

```bash
# Recommended: Install with Poetry
poetry install

# Alternative: Install with pip
pip install -r requirements.txt
```

### Running Examples

```bash
# Using Poetry (recommended)
poetry run example           # Run the main example with plots
poetry run example-save-load # Run the save/load example
poetry run example-geospatial # Run the geospatial visualization example

# Alternative: Running files directly
python prob_gbt/examples/example.py
python prob_gbt/examples/example_save_load.py
python prob_gbt/examples/example_geospatial.py
```

This will:
1. Load the California housing prices dataset
2. Initialize and train a ProbGBT model
3. Make predictions with uncertainty estimates
4. Calculate performance metrics
5. Generate visualizations in the 'images' directory

For detailed usage examples, please refer to the example files in the `examples/` directory.

## How It Works

ProbGBT works by:

1. **Quantile Transformation**: Generating non-uniformly spaced quantiles with more focus on the tails of the distribution
2. **Multi-Quantile Regression**: Training a CatBoost model to predict multiple quantiles simultaneously (or separate models for each quantile)
3. **PDF Estimation**: Computing the probability density function from quantiles
4. **Calibration** (enabled by default): Using conformal prediction to ensure statistical validity of the confidence intervals

### Technical Details on PDF Generation

The PDF generation process in ProbGBT involves several steps:

1. **Non-uniform Quantile Generation**: 
   - Instead of using uniformly spaced quantiles, ProbGBT transforms them using the normal distribution's PPF (Percent Point Function) and CDF (Cumulative Distribution Function)
   - This places more focus on the tails of the distribution, improving the model's ability to capture extreme values
   - The transformation uses: `non_uniform_quantiles = norm.cdf(norm.ppf(uniform_quantiles) * 1.5)`

2. **Quantile Prediction with CatBoost**:
   - The model can use either:
     - CatBoost's MultiQuantile loss function to predict all quantiles simultaneously (default)
     - Separate CatBoost models for each quantile (when `train_separate_models=True`)

3. **PDF Generation Methods**:
   ProbGBT offers three methods for transforming quantile predictions into PDFs, each with different characteristics:
   
   - **Sampling with KDE (`sample_kde`)**: The default method
     - Fully nonparametric approach that makes no assumptions about the distribution shape
     - Generates samples from the empirical CDF and applies Kernel Density Estimation (with `method='Silverman'`)
     - Produces smooth distributions
     - Most computationally intensive method (slowest)
     - Prone to under confidence (distributions too wide), but effectively fixed with conformal calibration (enabled by default)
   
   - **Spline-based smoothing (`spline`)**:
     - Nonparametric approach that directly approximates empirical CDF using splines
     - Fits a splines with monotonicity constraints using PyGAM package
     - Fastest method with lowest computational overhead
     - May produce peaky or irregular distributions in some cases
     - Recommended when no calibration is used: produces most calibrated distributions
   
   - **Gaussian Mixture Model (`gmm`)**:
     - Semi-parametric approach that combines spline smoothing with Gaussian mixtures
     - First applies spline-based smoothing, then fits a mixture of Gaussian distributions
     - Provides a good balance between flexibility and smoothness
     - Moderate computational cost (between `spline` and `sample_kde` in terms of speed)
     - Produces smoother distributions than the spline method
     - May not capture extremely complex distributions as well as `sample_kde`
     - When GMM is selected but fails to converge, spline results are used instead

   The choice of smoothing method depends on your specific needs:
      - Use `sample_kde` (default) for the most accurate representation when computational cost is not a concern
      - Use `spline` for the fastest performance when dealing with large datasets or when no calibration is used
      - Use `gmm` for a good balance between smooth distributions and reasonable performance

### Calibration Process

The calibration feature in ProbGBT uses conformal prediction to ensure that the predicted confidence intervals have the correct coverage probability. Here's how it works:

1. **Calibration Set Creation**:
   - During training, a calibration set is required to compute nonconformity scores
   - ProbGBT supports two approaches for obtaining this calibration set:
     
     a) **Automatic Splitting** (when `calibration_set=None`):
     - A portion of the training data is reserved for calibration (`calibration_size` parameter, default: 0.2)
     - The data is randomly shuffled (controlled by `random_seed`)
     - The first `calibration_size` fraction becomes the calibration set
     - The model is trained on the remaining (1-`calibration_size`) fraction
     - This approach is simple but reduces the effective training set size
     
     b) **Separate Calibration Set** (when `calibration_set=(X_cal, y_cal)`):
     - You provide a pre-defined calibration dataset
     - The model is trained on the full training set
     - Calibration is performed using the provided calibration set
     - This approach is better when you have sufficient data or for time series where random splitting is inappropriate
     - The calibration set should ideally come from the same distribution as the training data

2. **Nonconformity Scores**:
   - For each quantile level $\alpha$, the model computes nonconformity scores on the calibration set
   - For the quantile $q_{\alpha}(X)$, the nonconformity score is: $E_i = y_i - q_{\alpha}(X_i)$
   - These scores represent how much the true values deviate from the predicted quantiles

3. **Empirical Quantile Adjustment**:
   - For each desired quantile level $\alpha$, the model finds $\hat{s}$ such that:
     $P(y - q_{\alpha}(X) \leq \hat{s}) = \alpha$
   - This is calculated as the empirical $(n+1)\alpha/n$ quantile of the nonconformity scores
   - The adjusted prediction is then: $q_{\alpha}^{\text{cal}}(X) = q_{\alpha}(X) + \hat{s}$

4. **PDF from Calibrated Quantiles**:
   - If ProbGBT() was initialized with `calibrate=True` (default), the smoothed PDF is constructed from the calibrated quantiles
   - This ensures the PDF is consistent with the calibrated confidence intervals
   - The resulting PDFs maintain statistical validity while providing smooth, continuous distributions


## API Reference

### ProbGBT Class

```python
ProbGBT(
    num_quantiles=100,
    iterations=1000,
    learning_rate=None,
    depth=None,
    subsample=1.0,
    random_seed=None,
    train_separate_models=False,
    calibrate=True
)
```

#### Parameters:

- `num_quantiles`: Number of quantiles to predict (default: 100)
- `iterations`: Maximum number of trees to build (default: 1000)
- `learning_rate`: Learning rate for the gradient boosting algorithm (default: None, uses CatBoost default)
- `depth`: Depth of the trees (default: None, uses CatBoost default)
- `subsample`: Subsample ratio of the training instances (default: 1.0)
- `random_seed`: Random seed for reproducibility (default: None)
- `train_separate_models`: If True, train separate models for each quantile instead of using MultiQuantile loss (default: False)
- `calibrate`: If True, use conformal prediction to calibrate predicted quantiles (default: True)

#### Methods:

- `train(X, y, cat_features=None, eval_set=None, use_best_model=True, verbose=True, calibration_set=None, calibration_size=0.2)`: Train the model
- `predict_raw(X)`: Get raw quantile predictions (with calibration applied if enabled)
- `predict(X, method='sample_kde', num_points=1000)`: Predict probability density functions and CDFs for the given samples. Returns a list of tuples: `[(x_values, pdf_values, cdf_values), ...]` for each sample, providing both PDF and CDF in a single call.
- `evaluate_calibration(X_val, y_val, confidence_levels=None)`: Evaluate calibration quality on validation data
- `save(filepath, format='cbm', compression_level=6)`: Save the trained model to a file
- `load(filepath, format='cbm')`: Load a saved model from a file


### Save and Load Functionality

ProbGBT models can be saved to disk and loaded later for inference:

```python
# Save a trained model
model.save("model.cbm")  # For single model approach
model.save("models.tar.xz")  # For separate models approach

# Load a model
loaded_model = ProbGBT()
loaded_model.load("model.cbm")  # Load single model
loaded_model.load("models.tar.xz")  # Load separate models
```

#### Save Method

```python
save(filepath, format='cbm', compression_level=6)
```

- `filepath`: Path to save the model. For separate models, use a `.tar.xz` extension.
- `format`: Format for saving individual models. Options: 'cbm' (CatBoost binary), 'json'
- `compression_level`: Compression level for xz compression (1-9, where 9 is highest). Only used with separate models.

For the single model approach (when `train_separate_models=False`), this uses CatBoost's native `save_model` method.
For the separate models approach (when `train_separate_models=True`), this saves each model separately and compresses them into a tar.xz file.

#### Load Method

```python
load(filepath, format='cbm')
```

- `filepath`: Path to the saved model file.
- `format`: Format of the saved model. Only used for loading individual models from a tar.xz archive.

The method automatically detects whether the file is a single model or an archive of separate models based on the file extension.

## License

MIT