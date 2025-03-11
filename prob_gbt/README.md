# ProbGBT: Probabilistic Gradient Boosted Trees

ProbGBT is a probabilistic machine learning model that extends gradient boosted trees to provide uncertainty estimates. It uses CatBoost's MultiQuantile loss function to predict multiple quantiles of the target distribution, then constructs a probability density function (PDF) from these quantiles.

## Features

- **Uncertainty Estimation**: Provides confidence intervals and probability distributions for predictions
- **Non-parametric**: Makes no assumptions about the shape of the target distribution
- **Flexible**: Works with both numerical and categorical features
- **Efficient**: Built on top of CatBoost's fast gradient boosting implementation

## Example Visualizations

### Predictions with Confidence Intervals

![Predictions vs Actual with Confidence Intervals](./images/predictions_vs_actual.png)

*Scatter plot showing predicted vs actual values with 95% confidence intervals for selected samples.*

### Probability Density Function

![Probability Density Function](./images/predicted_pdf.png)

*Probability density function for a single example, showing the predicted distribution, actual value, and 95% confidence interval.*

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ProbGBT.git
cd ProbGBT

# Install dependencies
pip install -r requirements.txt
```

Or using Poetry:

```bash
# Install dependencies with Poetry
poetry install

# Run the example
poetry run run-example
```

## Dependencies

- numpy
- pandas
- catboost
- scipy
- scikit-learn
- pygam
- matplotlib

## Usage

### Basic Example

```python
from prob_gbt import ProbGBT
from sklearn.model_selection import train_test_split

# Prepare your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the model
model = ProbGBT(num_quantiles=50, iterations=1000)
model.train(X_train, y_train, cat_features=cat_features)

# Make point predictions
y_pred = model.predict(X_test)

# Get confidence intervals
lower_bounds, upper_bounds = model.predict_interval(X_test, confidence_level=0.95)

# Get full probability distributions
pdfs = model.predict_pdf(X_test)
```

### Example Script

The repository includes an example script that demonstrates how to use ProbGBT with a synthetic house prices dataset:

```bash
# Using Python directly
python -m prob_gbt.example

# Using Poetry
poetry run run-example
```

This will:
1. Generate a synthetic dataset
2. Train a ProbGBT model
3. Make predictions with uncertainty estimates
4. Calculate performance metrics
5. Generate visualizations

## How It Works

ProbGBT works by:

1. **Quantile Transformation**: Generating non-uniformly spaced quantiles with more focus on the tails of the distribution
2. **Multi-Quantile Regression**: Training a CatBoost model to predict multiple quantiles simultaneously
3. **Smoothing**: Using Generalized Additive Models (GAMs) to smooth the quantile function
4. **PDF Estimation**: Computing the derivative of the smoothed quantile function to obtain the probability density function

## API Reference

### ProbGBT Class

```python
ProbGBT(
    num_quantiles=50,
    iterations=1000,
    learning_rate=None,
    depth=None,
    subsample=1.0,
    random_seed=42
)
```

#### Parameters:

- `num_quantiles`: Number of quantiles to predict (default: 50)
- `iterations`: Maximum number of trees to build (default: 1000)
- `learning_rate`: Learning rate for the gradient boosting algorithm (default: None, uses CatBoost default)
- `depth`: Depth of the trees (default: None, uses CatBoost default)
- `subsample`: Subsample ratio of the training instances (default: 1.0)
- `random_seed`: Random seed for reproducibility (default: 42)

#### Methods:

- `train(X, y, cat_features=None, eval_set=None, use_best_model=True, verbose=True)`: Train the model
- `predict(X, return_quantiles=False)`: Make predictions
- `predict_interval(X, confidence_level=0.95)`: Predict confidence intervals
- `predict_pdf(X, num_points=1000)`: Predict probability density functions

## License

MIT

## Acknowledgements

This implementation is based on the concepts of quantile regression and uncertainty estimation in gradient boosted trees. 