# Getting Started with modeltime_resample_py

This guide will help you get started with the modeltime_resample_py package for time series cross-validation and model evaluation.

## Installation

```bash
pip install modeltime-resample-py
```

Or install from source:

```bash
git clone https://github.com/your_username/modeltime_resample_py.git
cd modeltime_resample_py
pip install -e .
```

## Quick Start

### 1. Basic Time Series Split

```python
import pandas as pd
import numpy as np
from modeltime_resample_py import time_series_split

# Create sample data
dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
values = np.random.randn(365).cumsum()
data = pd.Series(values, index=dates, name='value')

# Create a single train/test split
train, test = time_series_split(
    data,
    initial='9 months',  # Training set size
    assess='3 months'    # Test set size
)

print(f"Train size: {len(train)}, Test size: {len(test)}")
```

### 2. Time Series Cross-Validation

```python
from modeltime_resample_py import time_series_cv, plot_time_series_cv_plan

# Create multiple CV splits
cv_splits = time_series_cv(
    data,
    initial='6 months',
    assess='1 month',
    skip='1 month',
    cumulative=False,  # Use rolling window
    slice_limit=5      # Maximum 5 splits
)

# Visualize the CV plan
plot_time_series_cv_plan(data, cv_splits)
```

### 3. Model Evaluation

```python
from sklearn.linear_model import LinearRegression
from modeltime_resample_py import evaluate_model

# Create features from date index
df = data.to_frame()
df['day_of_year'] = df.index.dayofyear
df['month'] = df.index.month

# Evaluate model with cross-validation
model = LinearRegression()
results = evaluate_model(
    data=df,
    model=model,
    initial='6 months',
    assess='1 month',
    target_column='value',
    feature_columns=['day_of_year', 'month'],
    metrics=['mae', 'rmse']
)

# View average performance
print(results.groupby('metric_name')['metric_value'].mean())
```

### 4. Compare Multiple Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from modeltime_resample_py import compare_models

# Define models to compare
models = {
    'linear': LinearRegression(),
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=0.1)
}

# Compare models
comparison = compare_models(
    data=df,
    models=models,
    initial='6 months',
    assess='1 month',
    target_column='value',
    feature_columns=['day_of_year', 'month']
)

# View results by model
print(comparison.groupby(['model_id', 'metric_name'])['metric_value'].mean())
```

### 5. Visualize Model Performance

```python
from modeltime_resample_py import fit_resamples, plot_resamples

# Get detailed resample results
resamples_df = fit_resamples(
    cv_splits=cv_splits,
    model_spec=model,
    data=df,
    target_column='value',
    feature_columns=['day_of_year', 'month']
)

# Plot actuals vs predictions
plot_resamples(resamples_df, max_splits_to_plot=3)
```

## Key Concepts

### Period Specifications

You can specify periods using:
- **Integers**: Number of observations (e.g., `initial=250`)
- **Time strings**: Duration strings (e.g., `initial='6 months'`, `assess='30 days'`)

Supported time strings:
- Years: `'1 year'`, `'2 years'`
- Months: `'6 months'`, `'3 months'`
- Weeks: `'4 weeks'`, `'2 weeks'`
- Days: `'30 days'`, `'7 days'`

### Cross-Validation Types

1. **Rolling Window** (`cumulative=False`): Training set size stays constant
2. **Expanding Window** (`cumulative=True`): Training set grows with each split

### Working with DataFrames

```python
# DataFrame with date column
df = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=365, freq='D'),
    'value': np.random.randn(365).cumsum(),
    'feature1': np.random.randn(365),
    'feature2': np.random.randn(365)
})

# Specify date column if not using DatetimeIndex
cv_splits = time_series_cv(
    df,
    date_column='date',
    initial='6 months',
    assess='1 month'
)
```

## Next Steps

- Explore the [API Reference](api_reference.md) for detailed function documentation
- Check out [Advanced Examples](advanced_examples.md) for more complex use cases
- Learn about [Custom Metrics](custom_metrics.md) for specialized evaluation 