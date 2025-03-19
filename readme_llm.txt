# ProbGBT: Token-Efficient Guide

## What
- Probabilistic gradient boosted trees for uncertainty estimates
- Built on CatBoost quantile regression
- Non-parametric, works with numerical/categorical features

## Install
```
poetry install  # Recommended
# OR
pip install -r requirements.txt
```

## Usage
```python
from prob_gbt import ProbGBT

# Init (all defaults)
model = ProbGBT()

# Train
model.train(X_train, y_train)

# Predict
predictions = model.predict(X_test)
# Returns [(x_values, pdf_values, cdf_values), ...] for each sample

# Save/load
model.save("model.cbm")
loaded_model = ProbGBT()
loaded_model.load("model.cbm")
```

## Default Parameters
- num_quantiles=100
- iterations=1000
- learning_rate=None (uses CatBoost default)
- depth=None (uses CatBoost default)
- subsample=1.0
- train_separate_models=False
- calibrate=True

## Methods
- train(X, y, cat_features=None, eval_set=None, use_best_model=True, verbose=True, calibration_set=None, calibration_size=0.2)
- predict(X, method='sample_kde', num_points=1000)
- predict_interval(X, confidence_level=0.95)
- evaluate_calibration(X_val, y_val)
- save(filepath, format='cbm')
- load(filepath) 