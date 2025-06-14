#!/usr/bin/env python3
"""
Simple test script to launch the enhanced dashboard with sample data.
This script creates mock data and launches the dashboard for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modeltime_resample_py.visualization.dashboard import create_interactive_dashboard

def create_sample_data():
    """Create sample resamples data for testing the dashboard."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define models and slices - more models for better comparison analysis
    models = ['ARIMA', 'Prophet', 'XGBoost', 'RandomForest', 'LinearRegression', 'LSTM', 'ETS', 'SARIMA', 'SVM', 'GBM']
    slices = [1, 2, 3, 4, 5, 6]
    
    # Add some realistic model-specific behavior patterns
    model_characteristics = {
        'ARIMA': {'bias_factor': 0.02, 'volatility': 0.8},
        'Prophet': {'bias_factor': -0.01, 'volatility': 0.9},
        'XGBoost': {'bias_factor': 0.00, 'volatility': 0.7},
        'RandomForest': {'bias_factor': 0.03, 'volatility': 0.85},
        'LinearRegression': {'bias_factor': 0.05, 'volatility': 1.2},
        'LSTM': {'bias_factor': -0.02, 'volatility': 0.75},
        'ETS': {'bias_factor': 0.01, 'volatility': 0.95},
        'SARIMA': {'bias_factor': -0.03, 'volatility': 0.82},
        'SVM': {'bias_factor': 0.04, 'volatility': 1.1},
        'GBM': {'bias_factor': -0.01, 'volatility': 0.65}
    }
    
    # Create base time series with trend and seasonality
    base_trend = np.linspace(100, 200, len(dates))
    seasonal_component = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    base_actuals = base_trend + seasonal_component + noise
    
    data_rows = []
    
    for slice_id in slices:
        # Split data into train/test (80/20 split)
        split_idx = int(0.8 * len(dates))
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        # Add slice-specific variation
        slice_variation = np.random.normal(0, 10, len(dates))
        slice_actuals = base_actuals + slice_variation
        
        for model_id in models:
            # Generate model-specific predictions with realistic characteristics
            characteristics = model_characteristics[model_id]
            model_accuracy = np.random.uniform(0.7, 0.95)  # Different models have different accuracy
            
            # Train period - fitted values with model-specific bias and volatility
            train_actuals = slice_actuals[:split_idx]
            train_bias = characteristics['bias_factor'] * np.mean(train_actuals)
            train_noise = np.random.normal(train_bias, 5 * characteristics['volatility'] * (1 - model_accuracy), len(train_dates))
            fitted_values = train_actuals + train_noise
            
            for i, date in enumerate(train_dates):
                data_rows.append({
                    'slice_id': slice_id,
                    'model_id': model_id,
                    'date': date,
                    'actuals': train_actuals[i],
                    'fitted_values': fitted_values[i],
                    'predictions': np.nan,
                    'period_type': 'train'
                })
            
            # Test period - predictions with model-specific characteristics
            test_actuals = slice_actuals[split_idx:]
            test_bias = characteristics['bias_factor'] * np.mean(test_actuals)
            test_noise = np.random.normal(test_bias, 8 * characteristics['volatility'] * (1 - model_accuracy), len(test_dates))
            predictions = test_actuals + test_noise
            
            for i, date in enumerate(test_dates):
                data_rows.append({
                    'slice_id': slice_id,
                    'model_id': model_id,
                    'date': date,
                    'actuals': test_actuals[i],
                    'fitted_values': np.nan,
                    'predictions': predictions[i],
                    'period_type': 'test'
                })
    
    # Create DataFrame with proper MultiIndex
    df = pd.DataFrame(data_rows)
    df = df.set_index(['slice_id', 'model_id', 'date'])
    df = df.sort_index()
    
    return df

def main():
    """Main function to create sample data and launch dashboard."""
    print("🚀 Creating sample data for dashboard testing...")
    
    # Create sample data
    resamples_df = create_sample_data()
    
    print(f"📊 Created sample data with:")
    print(f"   • {len(resamples_df.index.get_level_values('model_id').unique())} models")
    print(f"   • {len(resamples_df.index.get_level_values('slice_id').unique())} slices")
    print(f"   • {len(resamples_df)} total observations")
    print(f"   • Date range: {resamples_df.index.get_level_values('date').min()} to {resamples_df.index.get_level_values('date').max()}")
    
    # Create and launch dashboard
    print("\n🎯 Launching Enhanced Dashboard...")
    dashboard = create_interactive_dashboard(
        resamples_df=resamples_df,
        title="Enhanced Model Analysis Dashboard - Test Data",
        port=8050,
        debug=True
    )
    
    print("\n🌟 Enhanced Dashboard Features:")
    print("   • Navigate to the 'Performance Metrics' tab")
    print("   • NEW: Temporal groups now appear as COLUMNS")
    print("   • NEW: Hierarchical rows show Model | Period Type structure")
    print("   • NEW: Single metric selection (dropdown instead of multi-select)")
    print("   • NEW: Month Groups granularity (Jan, Feb, Mar...) - properly ordered!")
    print("   • NEW: 'Total' column shows average performance across all temporal groups")
    print("\n📊 ADVANCED MODEL COMPARISON TAB:")
    print("   • 🎯 Performance Summary: Enhanced tables with confidence intervals")
    print("   • 🔄 Pairwise Comparison: Statistical significance testing (Wilcoxon)")
    print("   • 🏆 Dominance Analysis: Win/loss heatmaps between models")
    print("   • ⚖️ Multi-Metric Ranking: Custom weighted composite scores")
    print("   • 🌈 Color coding: Green=best, Yellow=middle, Red=worst")
    print("   • ★ Statistical indicators: *, **, *** for significance levels")
    print("   • Export data and filter by different criteria")
    
    print(f"\n🔗 Open your browser to: http://localhost:8050")
    print("⚡ Press Ctrl+C to stop the dashboard")
    
    # Run the dashboard
    dashboard.run(debug=True, port=8050)

if __name__ == "__main__":
    main()