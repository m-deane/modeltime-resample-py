"""
Demo of parallel processing capabilities in modeltime_resample_py
"""

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from modeltime_resample_py import evaluate_model
from modeltime_resample_py.parallel import evaluate_model_parallel

def create_large_dataset(n_days=1000):
    """Create a larger dataset to demonstrate parallel speedup."""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # More complex time series with multiple components
    trend = np.linspace(100, 200, n_days)
    seasonal_annual = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    seasonal_weekly = 5 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = np.random.normal(0, 5, n_days)
    values = trend + seasonal_annual + seasonal_weekly + noise
    
    # Create DataFrame with many features
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'day_of_year': dates.dayofyear,
        'month': dates.month,
        'quarter': dates.quarter,
        'year': dates.year,
        'week_of_year': dates.isocalendar().week,
        'day_of_week': dates.dayofweek,
        'is_weekend': dates.weekday.isin([5, 6]).astype(int),
        'lag_1': np.roll(values, 1),
        'lag_7': np.roll(values, 7),
        'lag_30': np.roll(values, 30),
        'rolling_mean_7': pd.Series(values).rolling(7).mean().values,
        'rolling_std_7': pd.Series(values).rolling(7).std().values,
    })
    df.set_index('date', inplace=True)
    
    # Remove NaN values from rolling operations
    df = df.dropna()
    
    return df

def main():
    print("=== Parallel Processing Demo ===\n")
    
    # Create dataset
    print("Creating large dataset...")
    df = create_large_dataset(n_days=1000)
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}\n")
    
    # Define features
    feature_columns = [
        'day_of_year', 'month', 'quarter', 'week_of_year',
        'day_of_week', 'is_weekend', 'lag_1', 'lag_7', 'lag_30',
        'rolling_mean_7', 'rolling_std_7'
    ]
    
    # Use a more complex model that benefits from parallelization
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=1  # Force single-threaded for fair comparison
    )
    
    # Parameters for cross-validation
    cv_params = {
        'data': df,
        'model': model,
        'initial': '365 days',
        'assess': '30 days',
        'skip': '30 days',
        'target_column': 'value',
        'feature_columns': feature_columns,
        'metrics': ['mae', 'rmse'],
        'slice_limit': 10  # More splits to show parallel benefit
    }
    
    # Sequential evaluation
    print("Running sequential evaluation...")
    start_time = time.time()
    results_sequential = evaluate_model(**cv_params)
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f} seconds")
    
    # Parallel evaluation
    print("\nRunning parallel evaluation...")
    start_time = time.time()
    results_parallel = evaluate_model_parallel(
        **cv_params,
        n_jobs=-1,  # Use all cores
        verbose=1   # Show progress bar
    )
    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.2f} seconds")
    
    # Compare results
    print(f"\nSpeedup: {sequential_time / parallel_time:.2f}x")
    
    # Verify results are the same
    seq_mean = results_sequential.groupby('metric_name')['metric_value'].mean()
    par_mean = results_parallel.groupby('metric_name')['metric_value'].mean()
    
    print("\nResults comparison:")
    print("Sequential:", seq_mean.round(3).to_dict())
    print("Parallel:  ", par_mean.round(3).to_dict())
    
    # Show performance by split
    print("\nPerformance by split (MAE):")
    mae_results = results_parallel[results_parallel['metric_name'] == 'mae']
    for _, row in mae_results.iterrows():
        print(f"Split {row['slice_id']}: {row['metric_value']:.2f}")

if __name__ == "__main__":
    main() 