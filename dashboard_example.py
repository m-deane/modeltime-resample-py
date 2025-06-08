#!/usr/bin/env python3
"""
Interactive Dashboard Example
============================

This example shows how to use the create_interactive_dashboard function
and demonstrates its key features.
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

def create_sample_data():
    """Create sample time series data for demonstration."""
    # Generate 1 year of daily data with trend and seasonality
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # Create realistic time series with trend, seasonality, and noise
    trend = np.linspace(100, 120, len(dates))  # Upward trend
    seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)  # Annual seasonality
    weekly = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly pattern
    noise = np.random.normal(0, 3, len(dates))  # Random noise
    
    values = trend + seasonal + weekly + noise
    
    df = pd.DataFrame({'date': dates, 'value': values})
    df.set_index('date', inplace=True)
    
    return df

def main():
    """Demonstrate the interactive dashboard functionality."""
    print("Interactive Dashboard Example")
    print("=" * 50)
    
    # 1. Create sample data
    print("\n1. Creating sample time series data...")
    data = create_sample_data()
    print(f"   ✓ Generated {len(data)} daily observations")
    print(f"   ✓ Date range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"   ✓ Value range: {data.value.min():.1f} to {data.value.max():.1f}")
    
    # 2. Create cross-validation splits
    print("\n2. Creating time series cross-validation splits...")
    cv_splits = time_series_cv(
        data['value'],
        initial='8 months',    # 8 months for initial training
        assess='1 month',      # 1 month for testing
        skip='2 weeks',        # 2 weeks gap between folds
        slice_limit=4          # Maximum 4 folds
    )
    print(f"   ✓ Created {len(cv_splits)} CV splits")
    
    # 3. Set up multiple models for comparison
    print("\n3. Setting up models for comparison...")
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'SimpleAverage': LinearRegression()  # Will use dummy features
    }
    print(f"   ✓ Configured {len(models)} models")
    
    # 4. Fit models on all CV splits
    print("\n4. Fitting models on cross-validation splits...")
    all_results = []
    
    for model_name, model in models.items():
        print(f"   • Fitting {model_name}...")
        model_results = fit_resamples(
            cv_splits=cv_splits,
            model_spec=model,
            data=data,
            target_column='value',
            feature_columns=None,  # Use dummy features for univariate models
            model_id=model_name
        )
        all_results.append(model_results)
    
    # Combine all results
    results = pd.concat(all_results, ignore_index=False)
    print(f"   ✓ Generated {len(results):,} predictions across all models and splits")
    
    # 5. Calculate accuracy metrics
    print("\n5. Calculating accuracy metrics...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics_dict = {
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error,
        'mape': mape
    }
    
    accuracy = resample_accuracy(results, metrics_set=metrics_dict)
    print(f"   ✓ Calculated {len(accuracy)} accuracy measurements")
    
    # Show accuracy summary
    print("\n   📊 Model Performance Summary:")
    avg_metrics = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    for (model, metric), value in avg_metrics.items():
        print(f"      {model:15s} {metric:4s}: {value:6.2f}")
    
    # 6. Create interactive dashboard
    print("\n6. Creating interactive dashboard...")
    dashboard = create_interactive_dashboard(
        resamples_df=results,
        accuracy_df=accuracy,
        title="Time Series Model Comparison Dashboard"
    )
    
    print("\n" + "🎉" * 25)
    print("INTERACTIVE DASHBOARD CREATED!")
    print("🎉" * 25)
    
    print("\n📊 Dashboard Features:")
    print("   🔍 FILTERING:")
    print("      • Select specific models to compare")
    print("      • Choose CV splits to analyze")
    print("      • Filter by date range")
    
    print("\n   📈 VISUALIZATIONS:")
    print("      • Time Series View: Actual vs Predicted values")
    print("      • Residuals View: Model error analysis")
    print("      • Metrics View: Performance comparison")
    
    print("\n   📋 DATA EXPLORATION:")
    print("      • Interactive data table with sorting")
    print("      • Real-time statistics")
    print("      • Summary metrics by model and split")
    
    print("\n   💾 EXPORT:")
    print("      • Download plots as images")
    print("      • Export data as CSV")
    print("      • Save filtered results")
    
    print("\n🚀 To launch the dashboard:")
    print("   dashboard.run(port=8050)")
    print("   Then open: http://localhost:8050")
    
    print("\n💡 Dashboard Usage Tips:")
    print("   • Use the model selector to compare different algorithms")
    print("   • Switch between view types to analyze different aspects")
    print("   • Hover over plots for detailed information")
    print("   • Use date picker to focus on specific time periods")
    print("   • Check the statistics panel for numerical summaries")
    
    print("\n" + "=" * 50)
    print("Example complete! Dashboard object ready to run.")
    print("=" * 50)
    
    return dashboard

if __name__ == "__main__":
    dashboard = main()
    
    # Optionally launch the dashboard
    launch = input("\nWould you like to launch the dashboard now? (y/n): ")
    if launch.lower() in ['y', 'yes']:
        print("\n🌐 Launching dashboard at http://localhost:8050")
        print("Press Ctrl+C to stop...")
        try:
            dashboard.run(debug=False, port=8050)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("\n👍 Dashboard object created. Run dashboard.run(port=8050) when ready!") 