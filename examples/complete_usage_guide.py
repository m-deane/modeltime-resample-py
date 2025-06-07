"""
Complete Usage Guide for modeltime_resample_py
==============================================

This guide demonstrates all the key features of the library including:
1. Basic time series cross-validation
2. Model fitting and evaluation
3. Advanced visualization features
4. Interactive dashboards
5. Model comparison tools
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core library imports
from modeltime_resample_py import (
    # Core functionality
    time_series_cv,
    time_series_split,
    fit_resamples,
    resample_accuracy,
    evaluate_model,
    compare_models,
    
    # Plotting
    plot_resamples,
    plot_time_series_cv_plan,
    
    # Advanced visualization
    create_interactive_dashboard,
    plot_model_comparison_matrix,
    create_comparison_report
)

# Machine learning models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def create_sample_time_series(n_periods=730, start_date='2022-01-01'):
    """Create realistic sample time series data."""
    print("Creating sample time series data...")
    
    dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
    
    # Create realistic components
    trend = np.linspace(100, 200, n_periods)  # Growing trend
    
    # Annual seasonality
    annual_season = 20 * np.sin(2 * np.pi * np.arange(n_periods) / 365.25)
    
    # Weekly seasonality (weekends lower)
    weekly_season = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 7)
    
    # Random noise
    noise = np.random.normal(0, 8, n_periods)
    
    # Combine all components
    values = trend + annual_season + weekly_season + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'quarter': dates.quarter,
        'is_weekend': dates.dayofweek >= 5
    })
    
    # Add lag features
    df['lag_1'] = df['value'].shift(1)
    df['lag_7'] = df['value'].shift(7)
    df['lag_30'] = df['value'].shift(30)
    
    # Add rolling features
    df['rolling_mean_7'] = df['value'].rolling(7).mean()
    df['rolling_std_7'] = df['value'].rolling(7).std()
    df['rolling_mean_30'] = df['value'].rolling(30).mean()
    
    # Drop NaN values
    df = df.dropna()
    
    # Set date as index for time series operations
    df_indexed = df.set_index('date')
    
    print(f"   Created {len(df)} observations from {df['date'].min()} to {df['date'].max()}")
    
    return df, df_indexed


def example_1_basic_time_series_cv():
    """Example 1: Basic Time Series Cross-Validation"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Time Series Cross-Validation")
    print("="*60)
    
    # Create data
    df, df_indexed = create_sample_time_series()
    
    # Method 1: Single train/test split
    print("\n1.1 Single Train/Test Split:")
    single_split = time_series_split(
        df_indexed['value'],
        initial='18 months',  # 18 months for training
        assess='3 months'     # 3 months for testing
    )
    
    print(f"   Train period: {single_split[0][0].min()} to {single_split[0][0].max()}")
    print(f"   Test period:  {single_split[0][1].min()} to {single_split[0][1].max()}")
    
    # Method 2: Time series cross-validation
    print("\n1.2 Time Series Cross-Validation:")
    cv_splits = time_series_cv(
        df_indexed['value'],
        initial='12 months',   # Initial training window
        assess='2 months',     # Assessment period
        skip='1 month',        # Skip period between folds
        cumulative=False,      # Rolling window (not expanding)
        slice_limit=6          # Maximum 6 folds
    )
    
    print(f"   Created {len(cv_splits)} CV folds")
    for i, (train_idx, test_idx) in enumerate(cv_splits):
        print(f"   Fold {i+1}: Train {train_idx.min()} to {train_idx.max()}, "
              f"Test {test_idx.min()} to {test_idx.max()}")
    
    # Visualize the CV plan
    print("\n1.3 Visualizing CV Plan:")
    fig = plot_time_series_cv_plan(
        df_indexed['value'],
        cv_splits,
        title="Time Series Cross-Validation Plan"
    )
    fig.savefig('cv_plan.png', dpi=150, bbox_inches='tight')
    print("   Saved CV plan visualization: cv_plan.png")
    
    return cv_splits, df, df_indexed


def example_2_model_fitting_and_evaluation():
    """Example 2: Model Fitting and Evaluation"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Model Fitting and Evaluation")
    print("="*60)
    
    # Get CV splits and data from previous example
    cv_splits, df, df_indexed = example_1_basic_time_series_cv()
    
    # Define models
    print("\n2.1 Setting up models:")
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    
    print(f"   Using {len(models)} models: {list(models.keys())}")
    
    # Method 1: Fit individual models
    print("\n2.2 Fitting individual models:")
    all_results = []
    
    for model_name, model in models.items():
        print(f"   Fitting {model_name}...")
        
        # Fit model on resamples
        results = fit_resamples(cv_splits, model, df_indexed, 'value')
        results['model_id'] = model_name
        all_results.append(results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    print(f"   Total results: {len(combined_results)} rows")
    
    # Method 2: Use convenience function to compare models
    print("\n2.3 Using compare_models convenience function:")
    comparison_results = compare_models(
        models=models,
        cv_splits=cv_splits,
        data=df_indexed,
        target_column='value'
    )
    
    print("   Model comparison completed!")
    
    return combined_results, comparison_results, df_indexed


def example_3_accuracy_metrics():
    """Example 3: Calculate and Analyze Accuracy Metrics"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Accuracy Metrics and Analysis")
    print("="*60)
    
    # Get results from previous example
    combined_results, comparison_results, df_indexed = example_2_model_fitting_and_evaluation()
    
    # Calculate accuracy metrics
    print("\n3.1 Calculating accuracy metrics:")
    
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
        combined_results,
        metrics_set=metrics_dict,
        period_types_to_evaluate=['test']  # Only evaluate on test data
    )
    
    print(f"   Calculated {len(accuracy)} metric records")
    
    # Display summary statistics
    print("\n3.2 Summary by model:")
    summary = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].agg(['mean', 'std'])
    print(summary.round(3))
    
    # Find best model for each metric
    print("\n3.3 Best model by metric:")
    best_models = accuracy.groupby('metric_name').apply(
        lambda x: x.loc[x['metric_value'].idxmin() if x.name != 'r2' 
                       else x['metric_value'].idxmax(), 'model_id']
    )
    
    for metric, model in best_models.items():
        print(f"   {metric:6s}: {model}")
    
    return accuracy, combined_results


def example_4_basic_plotting():
    """Example 4: Basic Plotting and Visualization"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Basic Plotting and Visualization")
    print("="*60)
    
    # Get data from previous examples
    accuracy, combined_results = example_3_accuracy_metrics()
    
    # Plot resamples results
    print("\n4.1 Plotting resample results:")
    
    # Filter to show just one model for clarity
    linear_results = combined_results[combined_results['model_id'] == 'LinearRegression']
    
    # Create matplotlib plot
    fig = plot_resamples(
        linear_results,
        title="Linear Regression: Actuals vs Predictions",
        engine='matplotlib'
    )
    fig.savefig('resamples_plot.png', dpi=150, bbox_inches='tight')
    print("   Saved resamples plot: resamples_plot.png")
    
    # Create interactive plotly version
    try:
        fig_interactive = plot_resamples(
            linear_results,
            title="Linear Regression: Interactive Plot",
            engine='plotly'
        )
        fig_interactive.write_html('resamples_interactive.html')
        print("   Saved interactive plot: resamples_interactive.html")
    except Exception as e:
        print(f"   Could not create interactive plot: {e}")
    
    return accuracy, combined_results


def example_5_advanced_visualization():
    """Example 5: Advanced Visualization Features"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Advanced Visualization Features")
    print("="*60)
    
    # Get data from previous examples
    accuracy, combined_results = example_4_basic_plotting()
    
    # Prepare data for visualization (needs proper index structure)
    print("\n5.1 Preparing data for advanced visualization:")
    
    # Create proper multi-index structure
    viz_results = combined_results.copy()
    viz_results['date'] = pd.to_datetime(viz_results['date']) if 'date' in viz_results.columns else viz_results.index
    
    # Ensure we have slice_id
    if 'slice_id' not in viz_results.columns:
        if 'split_idx' in viz_results.columns:
            viz_results['slice_id'] = viz_results['split_idx']
        else:
            # Create slice_id based on groups
            viz_results = viz_results.sort_values(['model_id', 'date'])
            viz_results['slice_id'] = viz_results.groupby('model_id').cumcount() // 60  # Approximate
    
    # Set multi-index
    viz_results = viz_results.set_index(['date', 'slice_id', 'model_id'])
    
    print(f"   Prepared {len(viz_results)} records for visualization")
    
    # 5.2 Model Comparison Matrix
    print("\n5.2 Creating model comparison visualizations:")
    
    # Heatmap comparison
    print("   - Creating heatmap comparison...")
    heatmap_fig = plot_model_comparison_matrix(
        accuracy_df=accuracy,
        plot_type='heatmap',
        title='Model Performance Heatmap',
        show_values=True
    )
    heatmap_fig.write_html('model_heatmap.html')
    print("     Saved: model_heatmap.html")
    
    # Radar chart comparison
    print("   - Creating radar chart...")
    radar_fig = plot_model_comparison_matrix(
        accuracy_df=accuracy,
        plot_type='radar',
        title='Model Performance Radar Chart'
    )
    radar_fig.write_html('model_radar.html')
    print("     Saved: model_radar.html")
    
    # Parallel coordinates
    print("   - Creating parallel coordinates plot...")
    parallel_fig = plot_model_comparison_matrix(
        accuracy_df=accuracy,
        plot_type='parallel',
        title='Model Performance Parallel Coordinates'
    )
    parallel_fig.write_html('model_parallel.html')
    print("     Saved: model_parallel.html")
    
    # 5.3 Comprehensive Comparison Report
    print("\n5.3 Creating comprehensive comparison report:")
    report = create_comparison_report(
        accuracy_df=accuracy,
        output_path='comprehensive_model_report.html',
        include_plots=['heatmap', 'radar', 'parallel'],
        title='Complete Model Comparison Report'
    )
    
    print("   Report components:")
    print(f"   - Figures: {list(report['figures'].keys())}")
    print(f"   - Rankings shape: {report['rankings'].shape}")
    print(f"   - Summary stats shape: {report['summary_stats'].shape}")
    print("   Saved: comprehensive_model_report.html")
    
    # Display model rankings
    print("\n   Model Rankings (lower rank = better performance):")
    print(report['rankings'].round(2))
    
    return viz_results, accuracy, report


def example_6_interactive_dashboard():
    """Example 6: Interactive Dashboard"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Interactive Dashboard")
    print("="*60)
    
    # Get data from previous examples
    viz_results, accuracy, report = example_5_advanced_visualization()
    
    print("\n6.1 Creating interactive dashboard:")
    
    # Create dashboard
    dashboard = create_interactive_dashboard(
        resamples_df=viz_results,
        accuracy_df=accuracy,
        title="Time Series Model Analysis Dashboard"
    )
    
    print("   Dashboard created successfully!")
    print("   Features available:")
    print("   - Filter by model, time slice, and date range")
    print("   - Multiple view types: time series, residuals, metrics")
    print("   - Interactive plots with zoom and hover")
    print("   - Statistics and data table views")
    print("   - Export capabilities")
    
    print("\n6.2 To run the dashboard:")
    print("   dashboard.run(port=8050)")
    print("   Then open http://localhost:8050 in your browser")
    
    return dashboard


def example_7_convenience_functions():
    """Example 7: Using Convenience Functions"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Convenience Functions")
    print("="*60)
    
    # Create fresh data
    df, df_indexed = create_sample_time_series(n_periods=500)
    
    # Single model evaluation
    print("\n7.1 Single model evaluation:")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    results = evaluate_model(
        data=df_indexed,
        model=model,
        target_column='value',
        initial='8 months',
        assess='1 month',
        skip='2 weeks'
    )
    
    print("   Evaluation completed!")
    print(f"   Results shape: {results.shape}")
    print(f"   Columns: {list(results.columns)}")
    
    # Quick model comparison
    print("\n7.2 Quick model comparison:")
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=0.5),
        'RF': RandomForestRegressor(n_estimators=30, random_state=42)
    }
    
    comparison = compare_models(
        models=models,
        data=df_indexed,
        target_column='value',
        initial='6 months',
        assess='1 month'
    )
    
    print("   Comparison completed!")
    print(f"   Results shape: {comparison.shape}")
    
    return results, comparison


def main():
    """Run all examples."""
    print("COMPLETE USAGE GUIDE FOR MODELTIME_RESAMPLE_PY")
    print("=" * 80)
    print("This guide demonstrates all key features of the library.")
    print("Generated files will be saved in the current directory.")
    print("=" * 80)
    
    try:
        # Run all examples
        example_1_basic_time_series_cv()
        example_2_model_fitting_and_evaluation()
        example_3_accuracy_metrics()
        example_4_basic_plotting()
        example_5_advanced_visualization()
        dashboard = example_6_interactive_dashboard()
        example_7_convenience_functions()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY - FILES GENERATED:")
        print("="*80)
        print("Static Visualizations:")
        print("  - cv_plan.png                    : Cross-validation plan")
        print("  - resamples_plot.png             : Model predictions plot")
        print("  - resamples_interactive.html     : Interactive predictions plot")
        print()
        print("Model Comparison:")
        print("  - model_heatmap.html             : Performance heatmap")
        print("  - model_radar.html               : Radar chart comparison")
        print("  - model_parallel.html            : Parallel coordinates plot")
        print("  - comprehensive_model_report.html: Complete comparison report")
        print()
        print("Interactive Dashboard:")
        print("  - Run: dashboard.run(port=8050)")
        print("  - URL: http://localhost:8050")
        print("="*80)
        
        return dashboard
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    dashboard = main()
    
    # Optionally run the dashboard
    run_dashboard = input("\nWould you like to run the interactive dashboard? (y/n): ")
    if run_dashboard.lower() == 'y':
        print("\nStarting dashboard on http://localhost:8050")
        print("Press Ctrl+C to stop the server.")
        try:
            dashboard.run(debug=False, port=8050)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
        except Exception as e:
            print(f"Error running dashboard: {e}") 