"""
Demonstration of Advanced Visualization Features
===============================================

This script demonstrates the interactive dashboard and model comparison
matrix features added to modeltime_resample_py.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Import modeltime_resample_py features
from modeltime_resample_py import (
    TimeSeriesDataPrep,
    fit_resamples,
    resample_accuracy,
    create_interactive_dashboard,
    plot_model_comparison_matrix,
    create_comparison_report
)


def generate_demo_data(n_periods=365*2, start_date='2022-01-01'):
    """Generate synthetic time series data for demonstration."""
    dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
    
    # Create trend, seasonality, and noise
    trend = np.linspace(100, 150, n_periods)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 365.25)
    weekly = 5 * np.sin(2 * np.pi * np.arange(n_periods) / 7)
    noise = np.random.normal(0, 5, n_periods)
    
    # Combine components
    values = trend + seasonal + weekly + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'day_of_year': dates.dayofyear,
        'week_of_year': dates.isocalendar().week,
        'quarter': dates.quarter
    })
    
    # Add lag features
    for lag in [1, 7, 14, 30]:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    # Add rolling features
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df


def main():
    """Run the visualization demonstration."""
    print("Advanced Visualization Features Demo")
    print("=" * 50)
    
    # 1. Generate demo data
    print("\n1. Generating demo data...")
    df = generate_demo_data()
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # 2. Prepare data for modeling
    print("\n2. Preparing data for modeling...")
    feature_cols = [col for col in df.columns if col not in ['date', 'value']]
    
    data_prep = TimeSeriesDataPrep(
        data=df,
        date_column='date',
        target_column='value',
        feature_columns=feature_cols
    )
    
    # 3. Define models
    print("\n3. Setting up models...")
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    
    # 4. Perform time series cross-validation
    print("\n4. Performing time series cross-validation...")
    print("   This may take a minute...")
    
    results = fit_resamples(
        data_prep=data_prep,
        models=models,
        initial_window=180,
        assess_period=30,
        skip_period=30,
        slice_limit=10
    )
    
    print(f"   Generated {len(results)} result records")
    
    # 5. Calculate accuracy metrics
    print("\n5. Calculating accuracy metrics...")
    
    # Define metrics as a dictionary
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics_dict = {
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error,
        'mape': mape,
        'r2': r2_score
    }
    
    accuracy = resample_accuracy(
        resamples_df=results,
        metrics_set=metrics_dict
    )
    
    print("\n   Average metrics by model:")
    avg_metrics = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    print(avg_metrics.round(3))
    
    # 6. Create model comparison visualizations
    print("\n6. Creating model comparison visualizations...")
    
    # Heatmap comparison
    print("   - Creating heatmap comparison...")
    heatmap_fig = plot_model_comparison_matrix(
        accuracy_df=accuracy,
        plot_type='heatmap',
        title='Model Performance Heatmap',
        show_values=True
    )
    heatmap_fig.write_html('model_comparison_heatmap.html')
    print("     Saved: model_comparison_heatmap.html")
    
    # Radar chart comparison
    print("   - Creating radar chart comparison...")
    radar_fig = plot_model_comparison_matrix(
        accuracy_df=accuracy,
        plot_type='radar',
        title='Model Performance Radar Chart'
    )
    radar_fig.write_html('model_comparison_radar.html')
    print("     Saved: model_comparison_radar.html")
    
    # Parallel coordinates comparison
    print("   - Creating parallel coordinates comparison...")
    parallel_fig = plot_model_comparison_matrix(
        accuracy_df=accuracy,
        plot_type='parallel',
        title='Model Performance Parallel Coordinates'
    )
    parallel_fig.write_html('model_comparison_parallel.html')
    print("     Saved: model_comparison_parallel.html")
    
    # 7. Create comprehensive comparison report
    print("\n7. Creating comprehensive comparison report...")
    report = create_comparison_report(
        accuracy_df=accuracy,
        output_path='model_comparison_report.html',
        include_plots=['heatmap', 'radar', 'parallel'],
        title='Time Series Model Comparison Report'
    )
    print("   Report saved: model_comparison_report.html")
    
    # Display rankings
    print("\n   Model Rankings (lower is better):")
    print(report['rankings'].round(2))
    
    # 8. Create interactive dashboard
    print("\n8. Creating interactive dashboard...")
    dashboard = create_interactive_dashboard(
        resamples_df=results,
        accuracy_df=accuracy,
        title="Time Series Model Analysis Dashboard"
    )
    
    print("\n" + "=" * 50)
    print("Demo complete! Generated files:")
    print("  - model_comparison_heatmap.html")
    print("  - model_comparison_radar.html") 
    print("  - model_comparison_parallel.html")
    print("  - model_comparison_report.html")
    print("\nTo view the interactive dashboard, run:")
    print("  dashboard.run(port=8050)")
    print("\nThen open http://localhost:8050 in your browser.")
    print("=" * 50)
    
    return dashboard, report


if __name__ == "__main__":
    # Run the demo
    dashboard, report = main()
    
    # Optionally run the dashboard
    user_input = input("\nWould you like to run the interactive dashboard? (y/n): ")
    if user_input.lower() == 'y':
        print("\nStarting dashboard on http://localhost:8050")
        print("Press Ctrl+C to stop the server.")
        dashboard.run(debug=False, port=8050) 