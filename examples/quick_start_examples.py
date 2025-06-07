"""
Quick Start Examples for modeltime_resample_py
==============================================

This file shows the most common use cases and patterns for using the library.
Perfect for getting started quickly!
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from modeltime_resample_py import (
    time_series_cv,
    fit_resamples,
    resample_accuracy,
    plot_resamples,
    plot_time_series_cv_plan,
    create_interactive_dashboard,
    plot_model_comparison_matrix,
    create_comparison_report
)


# ============================================================================
# 1. BASIC TIME SERIES CROSS-VALIDATION
# ============================================================================

def quick_example_1_basic_cv():
    """Quick Example 1: Basic time series cross-validation"""
    print("Example 1: Basic Time Series Cross-Validation")
    print("-" * 50)
    
    # Create sample data
    dates = pd.date_range('2022-01-01', periods=365, freq='D')
    values = 100 + np.cumsum(np.random.randn(365) * 0.5)  # Random walk
    ts_data = pd.Series(values, index=dates)
    
    # Create cross-validation splits
    cv_splits = time_series_cv(
        ts_data,
        initial='6 months',    # 6 months for initial training
        assess='1 month',      # 1 month for testing
        skip='2 weeks',        # 2 weeks gap between folds
        slice_limit=5          # Maximum 5 folds
    )
    
    print(f"Created {len(cv_splits)} CV folds")
    
    # Visualize the plan
    ax = plot_time_series_cv_plan(ts_data, cv_splits)
    ax.get_figure().savefig('quick_cv_plan.png', dpi=150, bbox_inches='tight')
    print("Saved: quick_cv_plan.png")
    
    return cv_splits, ts_data


# ============================================================================
# 2. MODEL FITTING AND EVALUATION
# ============================================================================

def quick_example_2_model_evaluation():
    """Quick Example 2: Fit and evaluate models"""
    print("\nExample 2: Model Fitting and Evaluation")
    print("-" * 50)
    
    # Get data from previous example
    cv_splits, ts_data = quick_example_1_basic_cv()
    
    # Define models
    models = {
        'Linear': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    # Fit models and combine results
    all_results = []
    for name, model in models.items():
        print(f"Fitting {name}...")
        results = fit_resamples(
            cv_splits=cv_splits, 
            model_spec=model, 
            data=ts_data, 
            target_column=ts_data.name if ts_data.name else 'value',
            model_id=name
        )
        all_results.append(results)
    
    combined_results = pd.concat(all_results)
    
    # Calculate accuracy metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics_dict = {
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error,
        'mape': mape
    }
    
    accuracy = resample_accuracy(
        combined_results,
        metrics_set=metrics_dict
    )
    
    print("\nModel Performance Summary:")
    summary = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    print(summary.round(3))
    
    return combined_results, accuracy


# ============================================================================
# 3. VISUALIZATION
# ============================================================================

def quick_example_3_visualization():
    """Quick Example 3: Create visualizations"""
    print("\nExample 3: Visualization")
    print("-" * 50)
    
    # Get results from previous example
    combined_results, accuracy = quick_example_2_model_evaluation()
    
    # Plot model results
    linear_results = combined_results.loc[combined_results.index.get_level_values('model_id') == 'Linear']
    fig = plot_resamples(linear_results, title="Linear Model Results")
    fig.savefig('quick_model_results.png', dpi=150, bbox_inches='tight')
    print("Saved: quick_model_results.png")
    
    # Model comparison heatmap
    heatmap_fig = plot_model_comparison_matrix(
        accuracy_df=accuracy,
        plot_type='heatmap',
        title='Model Performance Comparison'
    )
    heatmap_fig.write_html('quick_comparison.html')
    print("Saved: quick_comparison.html")
    
    return combined_results, accuracy


# ============================================================================
# 4. INTERACTIVE DASHBOARD
# ============================================================================

def quick_example_4_dashboard():
    """Quick Example 4: Interactive dashboard"""
    print("\nExample 4: Interactive Dashboard")
    print("-" * 50)
    
    # Get results from previous example
    combined_results, accuracy = quick_example_3_visualization()
    
    # Prepare data for dashboard (already has proper index structure)
    viz_data = combined_results.copy()
    
    # Create dashboard
    dashboard = create_interactive_dashboard(
        resamples_df=viz_data,
        accuracy_df=accuracy,
        title="Quick Start Dashboard"
    )
    
    print("Dashboard created! To run:")
    print("  dashboard.run(port=8050)")
    print("  Then open: http://localhost:8050")
    
    return dashboard


# ============================================================================
# 5. COMPREHENSIVE REPORT
# ============================================================================

def quick_example_5_report():
    """Quick Example 5: Generate comprehensive report"""
    print("\nExample 5: Comprehensive Report")
    print("-" * 50)
    
    # Get accuracy data
    _, accuracy = quick_example_2_model_evaluation()
    
    # Create comprehensive report
    report = create_comparison_report(
        accuracy_df=accuracy,
        output_path='quick_model_report.html',
        include_plots=['heatmap', 'radar'],
        title='Quick Start Model Report'
    )
    
    print("Report saved: quick_model_report.html")
    print("\nModel Rankings:")
    print(report['rankings'].round(2))
    
    return report


# ============================================================================
# COMMON PATTERNS AND RECIPES
# ============================================================================

def pattern_1_simple_backtest():
    """Pattern 1: Simple backtesting workflow"""
    print("\n" + "="*60)
    print("PATTERN 1: Simple Backtesting Workflow")
    print("="*60)
    
    # 1. Create time series data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    values = 100 + np.cumsum(np.random.randn(1000) * 0.3)
    ts = pd.Series(values, index=dates)
    
    # 2. Single train/test split
    from modeltime_resample_py import time_series_split
    train_data, test_data = time_series_split(ts, initial='2 years', assess='6 months')
    
    # 3. Fit model
    model = LinearRegression()
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
    
    model.fit(X_train, train_data.values)
    predictions = model.predict(X_test)
    
    # 4. Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(test_data.values, predictions))
    mae = mean_absolute_error(test_data.values, predictions)
    
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    
    return ts, predictions


def pattern_2_multiple_models():
    """Pattern 2: Compare multiple models quickly"""
    print("\n" + "="*60)
    print("PATTERN 2: Multiple Model Comparison")
    print("="*60)
    
    # Use the convenience function for quick comparison
    from modeltime_resample_py import compare_models
    
    # Create data
    dates = pd.date_range('2021-01-01', periods=500, freq='D')
    values = 100 + np.cumsum(np.random.randn(500) * 0.4)
    ts_data = pd.Series(values, index=dates)
    
    # Define models
    models = {
        'Linear': LinearRegression(),
        'RF_small': RandomForestRegressor(n_estimators=20, random_state=42),
        'RF_large': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Compare models (this does CV + fitting + metrics automatically)
    accuracy = compare_models(
        models=models,
        data=ts_data,
        initial='1 year',
        assess='2 months',
        skip='1 month'
    )
    
    # Show results
    summary = accuracy.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    print("Model Performance:")
    print(summary.round(3))
    
    return accuracy, accuracy


def pattern_3_feature_engineering():
    """Pattern 3: Time series with features"""
    print("\n" + "="*60)
    print("PATTERN 3: Time Series with Feature Engineering")
    print("="*60)
    
    # Create DataFrame with features
    dates = pd.date_range('2021-01-01', periods=400, freq='D')
    df = pd.DataFrame({
        'value': 100 + np.cumsum(np.random.randn(400) * 0.3),
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    }, index=dates)
    
    # Add lag features
    df['lag_1'] = df['value'].shift(1)
    df['lag_7'] = df['value'].shift(7)
    df['rolling_mean_7'] = df['value'].rolling(7).mean()
    
    # Drop NaN
    df = df.dropna()
    
    # Create CV splits
    cv_splits = time_series_cv(
        df['value'],
        initial='8 months',
        assess='1 month',
        slice_limit=4
    )
    
    # Fit model with features
    feature_cols = ['day_of_week', 'month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7']
    
    # Custom fitting function for features
    def fit_with_features(cv_splits, model, data, target_col, feature_cols):
        results = []
        
        for i, (train_idx, test_idx) in enumerate(cv_splits):
            # Get train/test data
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Fit model
            model.fit(train_data[feature_cols], train_data[target_col])
            
            # Predictions
            train_pred = model.predict(train_data[feature_cols])
            test_pred = model.predict(test_data[feature_cols])
            
            # Store results
            for idx, (actual, pred, period) in zip(
                train_idx, 
                zip(train_data[target_col], train_pred, ['train'] * len(train_idx))
            ):
                results.append({
                    'date': data.index[idx],
                    'split_idx': i,
                    'actuals': actual,
                    'predictions': pred,
                    'period_type': period
                })
            
            for idx, (actual, pred, period) in zip(
                test_idx,
                zip(test_data[target_col], test_pred, ['test'] * len(test_idx))
            ):
                results.append({
                    'date': data.index[idx],
                    'split_idx': i,
                    'actuals': actual,
                    'predictions': pred,
                    'period_type': period
                })
        
        return pd.DataFrame(results)
    
    # Fit Random Forest with features
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    results = fit_with_features(cv_splits, rf_model, df, 'value', feature_cols)
    
    # Calculate accuracy
    results['residuals'] = results['actuals'] - results['predictions']
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    metrics_dict = {
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error
    }
    accuracy = resample_accuracy(results, metrics_set=metrics_dict)
    
    print("Feature-based model performance:")
    print(accuracy.groupby('metric_name')['metric_value'].agg(['mean', 'std']).round(3))
    
    return results, df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run all quick start examples"""
    print("QUICK START EXAMPLES FOR MODELTIME_RESAMPLE_PY")
    print("=" * 80)
    
    # Basic examples
    quick_example_1_basic_cv()
    quick_example_2_model_evaluation()
    quick_example_3_visualization()
    dashboard = quick_example_4_dashboard()
    quick_example_5_report()
    
    # Advanced patterns
    pattern_1_simple_backtest()
    pattern_2_multiple_models()
    # pattern_3_feature_engineering()  # TODO: Fix index structure for custom fitting function
    
    print("\n" + "=" * 80)
    print("QUICK START COMPLETE!")
    print("=" * 80)
    print("Files generated:")
    print("  - quick_cv_plan.png")
    print("  - quick_model_results.png")
    print("  - quick_comparison.html")
    print("  - quick_model_report.html")
    print("\nTo run interactive dashboard:")
    print("  dashboard.run(port=8050)")
    print("=" * 80)
    
    return dashboard


if __name__ == "__main__":
    dashboard = main()
    
    # Option to run dashboard
    run_dash = input("\nRun interactive dashboard? (y/n): ")
    if run_dash.lower() == 'y':
        print("Starting dashboard at http://localhost:8050")
        dashboard.run(port=8050) 