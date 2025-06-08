#!/usr/bin/env python3
"""Quick Interactive Dashboard Demo"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from modeltime_resample_py import (
    time_series_cv,
    fit_resamples,
    resample_accuracy,
    create_interactive_dashboard
)

print('Interactive Dashboard Demo')
print('=' * 50)

# Generate sample data
dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
trend = np.linspace(100, 150, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 5, len(dates))
values = trend + seasonal + noise

df = pd.DataFrame({'date': dates, 'value': values})
df.set_index('date', inplace=True)

print(f'✓ Data shape: {df.shape}')

# Create CV splits
splits = time_series_cv(df['value'], initial='6 months', assess='1 month', skip='2 weeks', slice_limit=3)
print(f'✓ Created {len(splits)} CV splits')

# Set up models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=20, random_state=42)
}

# Fit models
all_results = []
for model_name, model in models.items():
    model_results = fit_resamples(
        cv_splits=splits,
        model_spec=model,
        data=df,
        target_column='value',
        feature_columns=None,  # Will create dummy features
        model_id=model_name
    )
    all_results.append(model_results)

results = pd.concat(all_results, ignore_index=False)

# Results are already properly indexed from fit_resamples
print(f'✓ Model fitting complete: {len(results)} predictions')

# Calculate accuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error
metrics_dict = {
    'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'mae': mean_absolute_error
}
accuracy = resample_accuracy(results, metrics_set=metrics_dict)
print(f'✓ Accuracy metrics calculated: {len(accuracy)} metric values')

# Create dashboard
dashboard = create_interactive_dashboard(
    resamples_df=results,
    accuracy_df=accuracy,
    title='Time Series Analysis Dashboard'
)

print('\n🎉 Dashboard created successfully!')
print('\n📊 Dashboard Features:')
print('  • Interactive time series plots with zoom/pan')
print('  • Model comparison and performance views')
print('  • Residual analysis and diagnostics')
print('  • Real-time filtering by model, date, and split')
print('  • Performance metrics visualization')
print('  • Data table with sorting and filtering')
print('  • Export capabilities for plots and data')

print('\n🚀 To launch the dashboard:')
print('  dashboard.run(port=8050)')
print('  Then open http://localhost:8050 in your browser')

print('\n📈 Sample Data Overview:')
print(f'  Date range: {df.index.min().date()} to {df.index.max().date()}')
print(f'  Value range: {df.value.min():.1f} to {df.value.max():.1f}')
print(f'  Models tested: {", ".join(models.keys())}')

print('\n📋 Accuracy Summary:')
for model in accuracy['model_id'].unique():
    model_acc = accuracy[accuracy['model_id'] == model]
    for metric in model_acc['metric_name'].unique():
        value = model_acc[model_acc['metric_name'] == metric]['metric_value'].mean()
        print(f'  {model:15s} {metric:4s}: {value:.3f}')

print('\n' + '=' * 50)
print('Demo complete! Dashboard object ready to run.') 