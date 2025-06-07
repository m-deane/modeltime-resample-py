# Usage Examples for modeltime_resample_py

This guide shows you how to use all the features of the `modeltime_resample_py` library, including the new advanced visualization capabilities.

## 🚀 Quick Start

### 1. Basic Time Series Cross-Validation

```python
import pandas as pd
import numpy as np
from modeltime_resample_py import time_series_cv, plot_time_series_cv_plan

# Create sample time series
dates = pd.date_range('2022-01-01', periods=365, freq='D')
values = 100 + np.cumsum(np.random.randn(365) * 0.5)
ts_data = pd.Series(values, index=dates)

# Create cross-validation splits
cv_splits = time_series_cv(
    ts_data,
    initial='6 months',    # 6 months for initial training
    assess='1 month',      # 1 month for testing
    skip='2 weeks',        # 2 weeks gap between folds
    slice_limit=5          # Maximum 5 folds
)

# Visualize the CV plan
fig = plot_time_series_cv_plan(ts_data, cv_splits)
fig.show()
```

### 2. Model Fitting and Evaluation

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from modeltime_resample_py import fit_resamples, resample_accuracy

# Define models
models = {
    'Linear': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
}

# Fit models on all CV folds
all_results = []
for name, model in models.items():
    results = fit_resamples(cv_splits, model, ts_data)
    results['model_id'] = name
    all_results.append(results)

combined_results = pd.concat(all_results, ignore_index=True)

# Calculate accuracy metrics
accuracy = resample_accuracy(
    combined_results,
    metrics=['rmse', 'mae', 'mape', 'r2']
)

# View results
print(accuracy.groupby(['model_id', 'metric_name'])['metric_value'].mean())
```

## 🎨 Advanced Visualization Features

### 3. Model Comparison Matrix

```python
from modeltime_resample_py import plot_model_comparison_matrix

# Create heatmap comparison
heatmap_fig = plot_model_comparison_matrix(
    accuracy_df=accuracy,
    plot_type='heatmap',
    title='Model Performance Heatmap',
    show_values=True
)
heatmap_fig.show()

# Create radar chart
radar_fig = plot_model_comparison_matrix(
    accuracy_df=accuracy,
    plot_type='radar',
    title='Model Performance Radar Chart'
)
radar_fig.show()

# Create parallel coordinates plot
parallel_fig = plot_model_comparison_matrix(
    accuracy_df=accuracy,
    plot_type='parallel',
    title='Model Performance Parallel Coordinates'
)
parallel_fig.show()
```

### 4. Comprehensive Comparison Report

```python
from modeltime_resample_py import create_comparison_report

# Generate complete HTML report
report = create_comparison_report(
    accuracy_df=accuracy,
    output_path='model_comparison_report.html',
    include_plots=['heatmap', 'radar', 'parallel'],
    title='Complete Model Comparison Report'
)

# View model rankings
print("Model Rankings:")
print(report['rankings'])

# Access individual components
print("Summary Statistics:")
print(report['summary_stats'])
```

### 5. Interactive Dashboard

```python
from modeltime_resample_py import create_interactive_dashboard

# Prepare data (ensure proper multi-index structure)
viz_data = combined_results.copy()
viz_data['date'] = pd.to_datetime(viz_data['date'])
viz_data['slice_id'] = viz_data.groupby('model_id').cumcount() // 30
viz_data = viz_data.set_index(['date', 'slice_id', 'model_id'])

# Create interactive dashboard
dashboard = create_interactive_dashboard(
    resamples_df=viz_data,
    accuracy_df=accuracy,
    title="Time Series Model Analysis Dashboard"
)

# Run the dashboard
dashboard.run(port=8050)
# Then open http://localhost:8050 in your browser
```

**Dashboard Features:**
- Filter by model, time slice, and date range
- Multiple view types: time series, residuals, metrics
- Interactive plots with zoom and hover
- Statistics and data table views
- Export capabilities

## 🛠️ Common Patterns and Workflows

### Pattern 1: Simple Backtesting

```python
from modeltime_resample_py import time_series_split

# Single train/test split
split = time_series_split(ts_data, initial='18 months', assess='6 months')
train_idx, test_idx = split[0]

# Fit and evaluate
model = LinearRegression()
# ... fit model on train_idx, predict on test_idx
```

### Pattern 2: Quick Model Comparison

```python
from modeltime_resample_py import compare_models

# Compare multiple models automatically
models = {
    'Linear': LinearRegression(),
    'RF_small': RandomForestRegressor(n_estimators=20),
    'RF_large': RandomForestRegressor(n_estimators=100)
}

results = compare_models(
    models=models,
    data=ts_data,
    initial='1 year',
    assess='2 months',
    skip='1 month'
)

# Calculate and compare metrics
accuracy = resample_accuracy(results, metrics=['rmse', 'mae'])
```

### Pattern 3: Feature Engineering

```python
# Create DataFrame with features
df = pd.DataFrame({
    'value': ts_data.values,
    'day_of_week': ts_data.index.dayofweek,
    'month': ts_data.index.month,
    'lag_1': ts_data.shift(1),
    'rolling_mean_7': ts_data.rolling(7).mean()
}, index=ts_data.index).dropna()

# Use features in modeling
cv_splits = time_series_cv(df['value'], initial='8 months', assess='1 month')
# ... fit models using feature columns
```

## 📊 Visualization Options

### Basic Plotting

```python
from modeltime_resample_py import plot_resamples

# Plot model results
fig = plot_resamples(
    combined_results[combined_results['model_id'] == 'Linear'],
    title="Linear Model: Actuals vs Predictions",
    engine='matplotlib'  # or 'plotly' for interactive
)
fig.show()
```

### Model Comparison Visualizations

| Plot Type | Use Case | Code |
|-----------|----------|------|
| **Heatmap** | Overall performance comparison | `plot_model_comparison_matrix(accuracy, plot_type='heatmap')` |
| **Radar Chart** | Multi-metric comparison | `plot_model_comparison_matrix(accuracy, plot_type='radar')` |
| **Parallel Coordinates** | Relationship exploration | `plot_model_comparison_matrix(accuracy, plot_type='parallel')` |

### Customization Options

```python
# Customize heatmap
fig = plot_model_comparison_matrix(
    accuracy_df=accuracy,
    plot_type='heatmap',
    metrics=['rmse', 'mae'],  # Filter specific metrics
    models=['Linear', 'RandomForest'],  # Filter specific models
    show_values=True,
    value_format='.3f',
    colorscale='RdYlGn',
    engine='plotly'  # or 'matplotlib'
)
```

## 🔧 Advanced Features

### Custom Metrics

```python
# Define custom metric function
def custom_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true) * 100

# Use in resample_accuracy
accuracy = resample_accuracy(
    combined_results,
    metrics=['rmse', 'mae', custom_metric],
    metric_names=['rmse', 'mae', 'custom_mape']
)
```

### Filtering and Analysis

```python
# Filter accuracy by period type
test_accuracy = accuracy[accuracy['period_type'] == 'test']

# Filter by specific models
model_subset = accuracy[accuracy['model_id'].isin(['Linear', 'RandomForest'])]

# Statistical analysis
summary_stats = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].agg([
    'mean', 'std', 'min', 'max', 'count'
])
```

## 📁 File Outputs

The library can generate various output files:

- **Static Images**: `.png` files for plots
- **Interactive HTML**: `.html` files for interactive plots
- **Comprehensive Reports**: Complete HTML reports with all visualizations
- **Dashboard**: Live web application for data exploration

## 🎯 Best Practices

1. **Data Preparation**: Ensure your time series has a proper datetime index
2. **CV Strategy**: Choose appropriate `initial`, `assess`, and `skip` periods based on your data frequency
3. **Model Selection**: Start with simple models (Linear) before moving to complex ones
4. **Visualization**: Use heatmaps for quick comparison, radar charts for detailed analysis
5. **Dashboard**: Use for interactive exploration and presentation to stakeholders

## 🚨 Common Issues and Solutions

### Issue: Dashboard not displaying properly
**Solution**: Ensure your data has the correct multi-index structure:
```python
viz_data = viz_data.set_index(['date', 'slice_id', 'model_id'])
```

### Issue: Missing slice_id column
**Solution**: Create slice_id from existing data:
```python
viz_data['slice_id'] = viz_data.groupby('model_id').cumcount() // 30
```

### Issue: Visualization errors
**Solution**: Check that accuracy DataFrame has required columns:
```python
required_cols = ['model_id', 'metric_name', 'metric_value', 'slice_id']
assert all(col in accuracy.columns for col in required_cols)
```

## 📚 Complete Examples

See the example files for complete working examples:
- `examples/complete_usage_guide.py` - Comprehensive guide with all features
- `examples/quick_start_examples.py` - Quick start patterns and recipes
- `examples/demo_visualization_simple.py` - Simple visualization demo

Run any of these files to see the library in action! 