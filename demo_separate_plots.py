#!/usr/bin/env python3
"""
Demo: Separate Plots Feature
============================

This script demonstrates the new "All Splits - Separate Plots" feature
in the enhanced dashboard.
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
    """Demonstrate the separate plots feature."""
    print("🎯 Separate Plots Feature Demo")
    print("=" * 50)
    
    # Generate sample data with clear patterns
    print("\n📊 Generating sample data with distinct patterns...")
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    
    # Create data with trend, seasonality, and some noise
    trend = np.linspace(100, 130, len(dates))
    seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # Monthly pattern
    noise = np.random.normal(0, 3, len(dates))
    values = trend + seasonal + noise
    
    df = pd.DataFrame({'date': dates, 'value': values})
    df.set_index('date', inplace=True)
    
    print(f"   ✓ Generated {len(df)} daily observations")
    print(f"   ✓ Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Create multiple CV splits for better demonstration
    print("\n🔄 Creating multiple CV splits...")
    splits = time_series_cv(
        df['value'],
        initial='2 months',    # 2 months for initial training
        assess='2 weeks',      # 2 weeks for testing
        skip='1 week',         # 1 week gap between folds
        slice_limit=6          # 6 splits for good visualization
    )
    print(f"   ✓ Created {len(splits)} CV splits")
    
    # Set up multiple models for comparison
    print("\n🤖 Setting up multiple models...")
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=30, random_state=42),
        'SimpleModel': LinearRegression()  # Another simple model for comparison
    }
    print(f"   ✓ Configured {len(models)} models")
    
    # Fit all models
    print("\n⚙️  Fitting models on all CV splits...")
    all_results = []
    for model_name, model in models.items():
        print(f"   • Fitting {model_name}...")
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
    print(f"   ✓ Generated {len(results):,} predictions across all models and splits")
    
    # Calculate accuracy metrics
    print("\n📈 Calculating accuracy metrics...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics_dict = {
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error,
        'mape': mape
    }
    
    accuracy = resample_accuracy(results, metrics_set=metrics_dict)
    print(f"   ✓ Calculated {len(accuracy)} accuracy measurements")
    
    # Create enhanced dashboard
    print("\n🎛️  Creating enhanced dashboard with separate plots feature...")
    dashboard = create_interactive_dashboard(
        resamples_df=results,
        accuracy_df=accuracy,
        title="Enhanced Dashboard: Separate Plots Demo"
    )
    
    print("\n" + "🎉" * 30)
    print("ENHANCED DASHBOARD WITH SEPARATE PLOTS!")
    print("🎉" * 30)
    
    print("\n📊 New Features Available:")
    print("   🔍 VIEWING OPTIONS:")
    print("      • All Splits - Aggregated View (overlaid plots)")
    print("      • All Splits - Separate Plots (individual subplots)")
    print("      • Individual Slice/Model combinations")
    
    print("\n   📈 SEPARATE PLOTS BENEFITS:")
    print("      • Clear visualization of each model's performance per split")
    print("      • Easy comparison of train vs test periods")
    print("      • Individual subplot for each slice/model combination")
    print("      • Consistent color coding across models")
    print("      • Train/test split boundaries clearly marked")
    
    print("\n   🎛️  INTERACTIVE FEATURES:")
    print("      • Toggle between aggregated and separate views")
    print("      • Show/hide train and test periods")
    print("      • Comprehensive residual analysis")
    print("      • Model comparison dashboard")
    print("      • Export capabilities")
    
    print("\n🚀 To explore the separate plots feature:")
    print("   1. dashboard.run(port=8090)")
    print("   2. Open http://localhost:8090")
    print("   3. In the 'Select Split/Model' dropdown, choose:")
    print("      • 'All Splits - Separate Plots' for individual subplots")
    print("      • 'All Splits - Aggregated View' for overlaid plots")
    print("   4. Use display options to show/hide train/test periods")
    
    print("\n📋 Data Summary:")
    print(f"   • Models: {len(models)}")
    print(f"   • CV Splits: {len(splits)}")
    print(f"   • Total Predictions: {len(results):,}")
    print(f"   • Date Range: {df.index.min().date()} to {df.index.max().date()}")
    
    print("\n💡 Pro Tips:")
    print("   • Use separate plots to identify which splits are most challenging")
    print("   • Compare model performance across different time periods")
    print("   • Look for patterns in train vs test performance")
    print("   • Use the residual analysis tab for detailed diagnostics")
    
    print("\n" + "=" * 50)
    print("Demo ready! Launch dashboard to explore separate plots.")
    print("=" * 50)
    
    return dashboard

if __name__ == "__main__":
    dashboard = main()
    
    # Optionally launch the dashboard
    launch = input("\nWould you like to launch the dashboard to see separate plots? (y/n): ")
    if launch.lower() in ['y', 'yes']:
        print("\n🌐 Launching enhanced dashboard at http://localhost:8090")
        print("📊 Try both 'All Splits - Aggregated View' and 'All Splits - Separate Plots'")
        print("Press Ctrl+C to stop...")
        try:
            dashboard.run(debug=False, port=8090)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("\n👍 Dashboard object created. Run dashboard.run(port=8090) when ready!")
        print("🎯 Don't forget to try the 'All Splits - Separate Plots' option!") 