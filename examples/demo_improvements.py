"""
Demo script showcasing the improvements to modeltime_resample_py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# Import the improved API
from modeltime_resample_py import (
    evaluate_model,
    compare_models,
    quick_cv_split,
    plot_time_series_cv_plan,
    plot_resamples,
    ModelTimeError
)

def main():
    print("=== Modeltime Resample Py - Improvements Demo ===\n")
    
    # 1. Generate sample time series data
    print("1. Generating sample time series data...")
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=730, freq='D')  # 2 years
    trend = np.linspace(100, 200, 730)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(730) / 365.25)
    noise = np.random.normal(0, 5, 730)
    values = trend + seasonal + noise
    
    # Create DataFrame with features
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'day_of_year': dates.dayofyear,
        'month': dates.month,
        'quarter': dates.quarter,
        'year': dates.year,
        'week_of_year': dates.isocalendar().week,
        'is_weekend': dates.weekday.isin([5, 6]).astype(int)
    })
    df.set_index('date', inplace=True)
    print(f"Created dataset with {len(df)} observations\n")
    
    # 2. Quick train/test split
    print("2. Using quick_cv_split for simple train/test split...")
    train, test = quick_cv_split(df, test_size='2 months')
    print(f"Train size: {len(train)}, Test size: {len(test)}\n")
    
    # 3. Evaluate a single model
    print("3. Evaluating a single model with cross-validation...")
    model = LinearRegression()
    
    try:
        results = evaluate_model(
            data=df,
            model=model,
            initial='12 months',
            assess='2 months',
            skip='1 month',
            target_column='value',
            feature_columns=['day_of_year', 'month', 'quarter', 'week_of_year', 'is_weekend'],
            metrics=['mae', 'rmse'],
            slice_limit=5
        )
        
        print("Average performance across CV folds:")
        avg_metrics = results.groupby('metric_name')['metric_value'].agg(['mean', 'std'])
        print(avg_metrics)
        print()
        
    except (ModelTimeError, ValueError) as e:
        print(f"Error during model evaluation: {e}\n")
    
    # 4. Compare multiple models
    print("4. Comparing multiple models...")
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
    comparison_results = compare_models(
        data=df,
        models=models,
        initial='12 months',
        assess='2 months',
        skip='1 month',
        target_column='value',
        feature_columns=['day_of_year', 'month', 'quarter', 'week_of_year', 'is_weekend'],
        metrics=['mae', 'rmse'],
        slice_limit=3  # Fewer folds for faster demo
    )
    
    print("\nModel comparison results:")
    summary = comparison_results.groupby(['model_id', 'metric_name'])['metric_value'].agg(['mean', 'std'])
    print(summary.round(2))
    
    # 5. Find best model
    print("\n5. Best model by average MAE:")
    mae_results = comparison_results[comparison_results['metric_name'] == 'mae']
    best_model = mae_results.groupby('model_id')['metric_value'].mean().idxmin()
    best_mae = mae_results.groupby('model_id')['metric_value'].mean().min()
    print(f"Best model: {best_model} with MAE = {best_mae:.2f}")
    
    # 6. Error handling demo
    print("\n6. Demonstrating error handling...")
    try:
        # This should raise an error
        bad_results = evaluate_model(
            data=df,
            model=model,
            initial='25 months',  # Longer than data!
            assess='1 month',
            target_column='value'
        )
    except (ValueError, ModelTimeError) as e:
        print(f"Caught expected error: {type(e).__name__}: {e}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 