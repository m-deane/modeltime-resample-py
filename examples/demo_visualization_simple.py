"""
Simple Visualization Demo
========================

This script demonstrates the visualization features with the existing
modeltime_resample_py modules.
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


def generate_sample_data():
    """Generate sample time series data."""
    # Create 2 years of daily data
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
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    return df


def main():
    """Run the visualization demo."""
    print("Simple Visualization Demo")
    print("=" * 50)
    
    # 1. Generate data
    print("\n1. Generating sample data...")
    data = generate_sample_data()
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    
    # 2. Create time series splits
    print("\n2. Creating time series CV splits...")
    cv_splits = time_series_cv(
        data['value'],
        initial='365 days',
        assess='30 days',
        skip='30 days',
        cumulative=False,
        slice_limit=5
    )
    print(f"   Created {len(cv_splits)} splits")
    
    # 3. Define models
    print("\n3. Setting up models...")
    model1 = LinearRegression()
    model2 = RandomForestRegressor(n_estimators=50, random_state=42)
    
    # 4. Fit models on resamples
    print("\n4. Fitting models on resamples...")
    print("   - Fitting LinearRegression...")
    results1 = fit_resamples(cv_splits, model1, data, 'value')
    
    print("   - Fitting RandomForest...")
    results2 = fit_resamples(cv_splits, model2, data, 'value')
    
    # Add model identifiers
    results1 = results1.copy()
    results2 = results2.copy()
    results1['model_id'] = 'LinearRegression'
    results2['model_id'] = 'RandomForest'
    
    # Combine results
    results = pd.concat([results1, results2])
    
    # Check what columns we have
    print(f"   Results columns: {list(results.columns)}")
    
    # Create proper multi-index for visualization compatibility
    # The visualization expects: date, slice_id, model_id as index
    results['date'] = results.index  # Save date from index
    results.reset_index(drop=True, inplace=True)
    
    # Ensure we have the required columns
    if 'slice_id' not in results.columns:
        # Create slice_id from split_idx if it exists
        if 'split_idx' in results.columns:
            results['slice_id'] = results['split_idx']
        else:
            # Infer from the data - count unique groups
            results['slice_id'] = 0
            current_slice = 0
            last_idx = results.index[0]
            for idx in results.index[1:]:
                if idx < last_idx:  # New slice when index resets
                    current_slice += 1
                results.loc[idx, 'slice_id'] = current_slice
                last_idx = idx
    
    # Set the multi-index
    results = results.set_index(['date', 'slice_id', 'model_id'])
    
    print(f"   Total results: {len(results)} rows")
    
    # 5. Calculate accuracy metrics
    print("\n5. Calculating accuracy metrics...")
    
    # Define metrics as a dictionary
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics_dict = {
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error,
        'mape': mape
    }
    
    accuracy = resample_accuracy(
        results,
        metrics_set=metrics_dict
    )
    
    print("\n   Average metrics by model:")
    avg_metrics = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    for (model, metric), value in avg_metrics.items():
        print(f"   {model:20s} {metric:5s}: {value:.3f}")
    
    # 6. Display model performance summary
    print("\n6. Model performance summary:")
    summary = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].agg(['mean', 'std'])
    print(summary.round(3))
    
    # 7. Create interactive dashboard
    print("\n7. Creating interactive dashboard...")
    dashboard = create_interactive_dashboard(
        resamples_df=results,
        accuracy_df=accuracy,
        title="Simple Time Series Analysis"
    )
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nTo view the interactive dashboard, run:")
    print("  dashboard.run(port=8050)")
    print("Then open http://localhost:8050 in your browser.")
    print("=" * 50)
    
    return dashboard


if __name__ == "__main__":
    dashboard = main() 