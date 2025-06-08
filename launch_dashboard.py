#!/usr/bin/env python3
"""
Launch Interactive Dashboard
===========================

This script creates and launches the interactive dashboard for exploring
time series model results.
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
    """Create and launch the interactive dashboard."""
    print("🚀 Creating Interactive Dashboard")
    print("=" * 50)
    
    # Generate sample data
    print("📊 Generating sample time series data...")
    dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + noise
    
    df = pd.DataFrame({'date': dates, 'value': values})
    df.set_index('date', inplace=True)
    
    # Create CV splits
    print("🔄 Creating cross-validation splits...")
    splits = time_series_cv(df['value'], initial='6 months', assess='1 month', skip='2 weeks', slice_limit=3)
    
    # Set up models
    print("🤖 Setting up models...")
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=20, random_state=42)
    }
    
    # Fit models
    print("⚙️  Fitting models on resamples...")
    all_results = []
    for model_name, model in models.items():
        model_results = fit_resamples(
            cv_splits=splits,
            model_spec=model,
            data=df,
            target_column='value',
            feature_columns=None,
            model_id=model_name
        )
        all_results.append(model_results)
    
    results = pd.concat(all_results, ignore_index=False)
    
    # Calculate accuracy
    print("📈 Calculating accuracy metrics...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    metrics_dict = {
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error
    }
    accuracy = resample_accuracy(results, metrics_set=metrics_dict)
    
    # Create dashboard
    print("🎛️  Creating interactive dashboard...")
    dashboard = create_interactive_dashboard(
        resamples_df=results,
        accuracy_df=accuracy,
        title='Time Series Model Analysis Dashboard'
    )
    
    print("\n" + "🎉" * 20)
    print("DASHBOARD READY!")
    print("🎉" * 20)
    
    print("\n📋 Dashboard Overview:")
    print(f"  • Data points: {len(results):,}")
    print(f"  • Models: {len(models)}")
    print(f"  • CV splits: {len(splits)}")
    print(f"  • Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    print("\n🎛️  Dashboard Features:")
    print("  • 📊 Interactive time series plots")
    print("  • 🔍 Filter by model, split, and date range")
    print("  • 📈 Multiple view types (time series, residuals, metrics)")
    print("  • 📋 Data table with sorting and filtering")
    print("  • 📊 Real-time statistics and summaries")
    print("  • 💾 Export capabilities")
    
    print("\n🌐 Starting dashboard server...")
    print("  URL: http://localhost:8050")
    print("  Press Ctrl+C to stop the server")
    print("\n" + "=" * 50)
    
    try:
        dashboard.run(debug=False, port=8090)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")
        print("Make sure you have the required dependencies:")
        print("  pip install dash plotly dash-bootstrap-components")

if __name__ == "__main__":
    main() 