# Modeltime Resample Py

A Python package for time series cross-validation, resampling, model fitting, and evaluation, inspired by the R `modeltime.resample` and `rsample` packages.

[![Documentation Status](https://readthedocs.org/projects/modeltime-resample-py/badge/?version=latest)](https://modeltime-resample-py.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/your_username/modeltime-resample-py/workflows/CI/badge.svg)](https://github.com/your_username/modeltime-resample-py/actions)
<!-- [![PyPI version](https://badge.fury.io/py/modeltime-resample-py.svg)](https://badge.fury.io/py/modeltime-resample-py) -->
<!-- [![codecov](https://codecov.io/gh/your_username/modeltime-resample-py/branch/main/graph/badge.svg)](https://codecov.io/gh/your_username/modeltime-resample-py) -->

## Overview

`modeltime-resample-py` provides tools to:
*   Effectively split time series data for model training and evaluation.
*   Generate robust backtesting strategies using various train/test splits and cross-validation folds.
*   Fit models to resamples and evaluate their performance.
*   Visualize resampling plans and model performance on resamples.

## Features

*   **Single Time Series Splitting (`time_series_split`):** Create a single train/test division of your time series.
*   **Time Series Cross-Validation (`time_series_cv`):**
    *   Generate multiple splits using a rolling origin (fixed window) or expanding window approach.
    *   Control initial training size, assessment (test) size, and the period to skip between folds.
*   **Flexible Period Specification:** Define `initial`, `assess`, and `skip` parameters using:
    *   Integer counts (number of samples).
    *   Time-based period strings (e.g., '5 years', '3 months', '10 days', '12W', '2M').
*   **Modeling Utilities:**
    *   **Fit to Resamples (`fit_resamples`):** Fit a model to each cross-validation split, returning a long-format DataFrame with actuals, fitted values, predictions, and residuals.
    *   **Resample Accuracy (`resample_accuracy`):** Calculate performance metrics (e.g., MAE, RMSE, or custom) across all resamples. Allows specification of period types ('train', 'test', or both) for evaluation via the `period_types_to_evaluate` argument. The output includes a `period_type` column.
    *   **Parallel Processing:** Speed up model fitting with `fit_resamples_parallel` and `evaluate_model_parallel` using multiple CPU cores.
    *   **Interactive Dashboards:** Real-time exploration of results with filtering, multiple views, and export capabilities.
    *   **Model Comparison Matrix:** Visual comparison of model performance using heatmaps, radar charts, and parallel coordinates.
*   **Visualization:**
    *   **CV Plan Plotting (`plot_time_series_cv_plan`):** Plot the cross-validation plan to visually inspect the train/test splits over time.
    *   **Resample Plotting (`plot_resamples`):** Visualize actuals, fitted values (on train data), and predictions (on test data) from `fit_resamples` output. Supports both Matplotlib (default) and interactive Plotly (`engine='plotly'`) backends.
*   **Convenience Functions:**
    *   **`evaluate_model`:** High-level function to perform CV and calculate metrics in one call.
    *   **`compare_models`:** Compare multiple models using the same CV splits.
    *   **`quick_cv_split`:** Simple wrapper for single train/test split.
*   **Data Handling:**
    *   Works with both `pandas.Series` (with `DatetimeIndex`) and `pandas.DataFrame` (with a specified date column).
    *   Automatically handles unsorted time series data by sorting it and issuing a warning.

## Installation

Currently, you can install the package directly from the source:

```bash
git clone https://your-repository-url/modeltime-resample-py.git # Replace with your repo URL
cd modeltime-resample-py
pip install -e .
```

To include development dependencies (for running tests):
```bash
pip install -e ".[dev]"
```

<!--
Once published to PyPI:
```bash
pip install modeltime-resample-py
```
-->

**Core Dependencies:**

*   pandas >= 1.1.0
*   numpy >= 1.19.0
*   matplotlib >= 3.3.0 (For plotting utilities)

**Optional Dependencies:**

*   scikit-learn (Recommended for model cloning in `fit_resamples`. If not present, a warning is issued, and models are deepcopied or used as-is.)
*   plotly >= 5.0.0 (For interactive plotting with `plot_resamples(engine='plotly')`)


## Quick Start

Here's a brief example of `time_series_cv` and `plot_time_series_cv_plan`:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modeltime_resample_py import time_series_cv, plot_time_series_cv_plan

# 1. Create Sample Data
periods = 365 * 2 # 2 years of daily data
dates = pd.date_range(start='2021-01-01', periods=periods, freq='D')
values = np.random.randn(periods).cumsum()
data = pd.Series(values, index=dates, name='value')

# 2. Create Cross-Validation Splits
cv_splits = time_series_cv(
    data,
    initial='1 year',
    assess='3 months',
    skip='3 months',
    cumulative=False, # Rolling window
    slice_limit=4     # Generate up to 4 folds
)

# 3. Visualize the CV Plan
fig, ax = plot_time_series_cv_plan(
    data,
    cv_splits,
    title="Time Series CV Plan Example"
)
plt.show()
```

For fitting models to these splits and visualizing their performance (including interactive Plotly plots), see the cookbook. Example:

```python
# (Assuming cv_splits and data from above, and a trained 'model_spec')
# from modeltime_resample_py import fit_resamples, plot_resamples
# from sklearn.linear_model import LinearRegression # Example model

# model_spec = LinearRegression() 
# resamples_df = fit_resamples(cv_splits, model_spec, data, target_column='value')
# fig_resamples = plot_resamples(resamples_df, engine='matplotlib') # or engine='plotly'
# plt.show() # For matplotlib
# if engine == 'plotly': fig_resamples.show() 
```


## Examples / Cookbook

For more detailed examples covering all functionalities, including:
*   `time_series_split` for single splits.
*   Integer and period-based parameters.
*   Rolling vs. Expanding windows for CV.
*   Usage with `pandas.DataFrame` and `date_column`.
*   Handling unsorted data.
*   Fitting models with `fit_resamples`.
*   Calculating accuracy with `resample_accuracy` (including evaluation on different period types like 'train' and 'test').
*   Visualizing resamples with `plot_resamples` (Matplotlib and Plotly engines).

Please refer to the cookbook script: [`examples/cookbook.py`](examples/cookbook.py)

To run the cookbook:
```bash
python examples/cookbook.py
```
(Ensure you have scikit-learn and plotly installed for all cookbook examples to run: `pip install scikit-learn plotly`)

## Documentation

Full documentation is available at [Read the Docs](https://modeltime-resample-py.readthedocs.io).

To build the documentation locally:

```bash
cd docs
make html
```

## 🎨 Advanced Visualization Features

### Interactive Dashboard

Explore your model results interactively with a full-featured dashboard:

```python
from modeltime_resample_py import create_interactive_dashboard

# Create and launch dashboard
dashboard = create_interactive_dashboard(
    resamples_df=results,
    accuracy_df=accuracy,
    title="My Time Series Analysis"
)

# Run the dashboard server
dashboard.run(port=8050)
```

Features include:
- Filter by model, time slice, and date range
- Multiple view types: time series, residuals, metrics
- Interactive plots with zoom and hover details
- Export capabilities for plots and data

### Model Comparison Matrix

Visualize model performance across multiple metrics:

```python
from modeltime_resample_py import plot_model_comparison_matrix, create_comparison_report

# Create heatmap comparison
fig = plot_model_comparison_matrix(
    accuracy_df=accuracy,
    plot_type='heatmap',  # or 'radar', 'parallel'
    metrics=['rmse', 'mae', 'mape']
)

# Generate comprehensive report
report = create_comparison_report(
    accuracy_df=accuracy,
    output_path='model_comparison.html',
    include_plots=['heatmap', 'radar', 'parallel']
)
```

## 🚀 Parallel Processing

Speed up model evaluation using parallel processing:

```python
from modeltime_resample_py import fit_resamples_parallel

# Use all CPU cores
results = fit_resamples_parallel(
    data_prep=data_prep,
    models=models,
    initial_window=180,
    assess_period=30,
    n_jobs=-1,  # Use all cores
    verbose=True
)
```

## Contributing

Contributions are welcome! If you'd like to contribute, please:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Add tests for your changes.
4. Ensure all tests pass.
5. Open a pull request.

Please open an issue first to discuss any significant changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.