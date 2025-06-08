#!/usr/bin/env python3
"""
Interactive Dashboard Demo
=========================

This script demonstrates the create_interactive_dashboard function
and automatically launches the dashboard for you to explore.
"""

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


def main():
    """Run the dashboard demo."""
    print("Interactive Dashboard Demo")
    print("=" * 50)
    
    # 1. Generate sample data
    print("\n1. Generating sample data...")
    dates = pd.date_range(start='2022-01-01', periods=730, freq='D')
    
    # Create trend + seasonality + noise
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    
    values = trend + seasonal + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    df.set_index('date', inplace=True)
    
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # 2. Create CV splits
    print("\n2. Creating time series CV splits...")
    splits = time_series_cv(
        df['value'],
        initial='6 months',
        assess='1 month',
        skip='2 weeks',
        slice_limit=5
    )
    print(f"   Created {len(splits)} splits")
    
    # 3. Set up models
    print("\n3. Setting up models...")
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    # 4. Fit models
    print("\n4. Fitting models on resamples...")
    results = fit_resamples(
        splits,
        models,
        control={'verbose': False}
    )
    
    # Prepare results for dashboard
    if 'slice_id' not in results.columns:
        if 'split_idx' in results.columns:
            results['slice_id'] = results['split_idx']
        else:
            results['slice_id'] = 0
            current_slice = 0
            last_idx = results.index[0]
            for idx in results.index[1:]:
                if idx < last_idx:
                    current_slice += 1
                results.loc[idx, 'slice_id'] = current_slice
                last_idx = idx
    
    results = results.set_index(['date', 'slice_id', 'model_id'])
    print(f"   Total results: {len(results)} rows")
    
    # 5. Calculate accuracy
    print("\n5. Calculating accuracy metrics...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics_dict = {
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error,
        'mape': mape
    }
    
    accuracy = resample_accuracy(results, metrics_set=metrics_dict)
    
    print("\n   Average metrics by model:")
    avg_metrics = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    for (model, metric), value in avg_metrics.items():
        print(f"   {model:20s} {metric:5s}: {value:.3f}")
    
    # 6. Create and launch dashboard
    print("\n6. Creating interactive dashboard...")
    dashboard = create_interactive_dashboard(
        resamples_df=results,
        accuracy_df=accuracy,
        title="Interactive Time Series Analysis Dashboard"
    )
    
    print("\n" + "=" * 50)
    print("Dashboard created successfully!")
    print("\nFeatures available in the dashboard:")
    print("  • Filter by model, time slice, and date range")
    print("  • Multiple view types: time series, residuals, metrics")
    print("  • Interactive plots with zoom and hover")
    print("  • Statistics and data table views")
    print("  • Export capabilities")
    print("\nStarting dashboard on http://localhost:8050")
    print("Press Ctrl+C to stop the server.")
    print("=" * 50)
    
    # Launch the dashboard
    try:
        dashboard.run(debug=False, port=8050)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except Exception as e:
        print(f"\nError running dashboard: {e}")
        print("Make sure you have dash and plotly installed:")
        print("  pip install dash plotly dash-bootstrap-components")


if __name__ == "__main__":
    main() 