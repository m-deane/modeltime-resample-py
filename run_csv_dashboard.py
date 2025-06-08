#!/usr/bin/env python3
"""
CSV Dashboard Launcher
=====================

This script loads a CSV file with resamples data and launches the interactive dashboard.
Usage: python run_csv_dashboard.py [csv_file_path]
"""

import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from modeltime_resample_py import (
    resample_accuracy,
    create_interactive_dashboard
)


def load_csv_data(csv_path):
    """Load and prepare CSV data for the dashboard."""
    print(f"📊 Loading data from: {csv_path}")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set the multi-index as expected by the dashboard
    df = df.set_index(['date', 'slice_id', 'model_id'])
    
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
    print(f"   Unique slices: {df.index.get_level_values('slice_id').nunique()}")
    print(f"   Unique models: {df.index.get_level_values('model_id').nunique()}")
    
    return df


def calculate_accuracy_from_csv(results_df):
    """Calculate accuracy metrics from the loaded CSV data."""
    print("📈 Calculating accuracy metrics...")
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def mape(y_true, y_pred):
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not mask.any():
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def rmse(y_true, y_pred):
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics_dict = {
        'rmse': rmse,
        'mae': mean_absolute_error,
        'mape': mape
    }
    
    # Calculate accuracy using the package function
    accuracy = resample_accuracy(results_df, metrics_set=metrics_dict)
    
    print("\n   Average metrics by model:")
    if not accuracy.empty:
        avg_metrics = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].mean()
        for (model, metric), value in avg_metrics.items():
            print(f"   {model:30s} {metric:5s}: {value:.3f}")
    
    return accuracy


def main():
    """Main function to run the CSV dashboard."""
    print("CSV Dashboard Launcher")
    print("=" * 50)
    
    # Get CSV file path from command line or use default
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "resamples_outputs.csv"
    
    try:
        # 1. Load CSV data
        results_df = load_csv_data(csv_path)
        
        # 2. Calculate accuracy metrics
        accuracy_df = calculate_accuracy_from_csv(results_df)
        
        # 3. Create and launch dashboard
        print("\n🎛️  Creating interactive dashboard...")
        dashboard = create_interactive_dashboard(
            resamples_df=results_df,
            accuracy_df=accuracy_df,
            title=f"Interactive Analysis Dashboard - {csv_path}"
        )
        
        print("\n" + "=" * 50)
        print("Dashboard created successfully!")
        print("\nData Summary:")
        print(f"  • Total data points: {len(results_df):,}")
        print(f"  • Models: {results_df.index.get_level_values('model_id').nunique()}")
        print(f"  • CV slices: {results_df.index.get_level_values('slice_id').nunique()}")
        print(f"  • Date range: {results_df.index.get_level_values('date').min().date()} to {results_df.index.get_level_values('date').max().date()}")
        
        print("\n🎛️  Dashboard Features:")
        print("  • 📊 Interactive time series plots")
        print("  • 🔍 Filter by model, slice, and date range")
        print("  • 📈 Multiple view types (time series, residuals, metrics)")
        print("  • 📋 Data table with sorting and filtering")
        print("  • 📊 Real-time statistics and summaries")
        print("  • 💾 Export capabilities")
        
        print("\n🌐 Starting dashboard server...")
        print("  URL: http://localhost:8050")
        print("  Press Ctrl+C to stop the server")
        print("\n" + "=" * 50)
        
        # Launch the dashboard
        try:
            dashboard.run(debug=False, port=8050)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user.")
        except Exception as e:
            print(f"\n❌ Error running dashboard: {e}")
            print("Make sure you have the required dependencies:")
            print("  pip install dash plotly dash-bootstrap-components")
    
    except FileNotFoundError:
        print(f"❌ Error: File '{csv_path}' not found.")
        print("Usage: python run_csv_dashboard.py [csv_file_path]")
        print("Example: python run_csv_dashboard.py resamples_outputs.csv")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Make sure your CSV file has the required columns:")
        print("  date, slice_id, model_id, actuals, fitted_values, predictions, residuals, period_type")


if __name__ == "__main__":
    main() 