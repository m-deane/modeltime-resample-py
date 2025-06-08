#!/usr/bin/env python3
"""
Enhanced Metrics Calculator for Time Series Resamples
====================================================

This module provides advanced statistical metrics that can be calculated
from resamples output data to enhance dashboard analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class EnhancedMetricsCalculator:
    """Calculate advanced statistical metrics from resamples data."""
    
    def __init__(self, resamples_df):
        """
        Initialize with resamples DataFrame.
        
        Args:
            resamples_df: DataFrame with columns [actuals, fitted_values, predictions, residuals, period_type]
        """
        self.df = resamples_df.copy()
        self.metrics_cache = {}
    
    def calculate_all_metrics(self, rolling_windows=[7, 14, 30]):
        """Calculate comprehensive set of metrics."""
        metrics = {}
        
        # Basic metrics
        metrics.update(self.calculate_basic_metrics())
        
        # Rolling metrics
        metrics.update(self.calculate_rolling_metrics(rolling_windows))
        
        # Statistical tests
        metrics.update(self.calculate_statistical_tests())
        
        # Advanced performance metrics
        metrics.update(self.calculate_advanced_performance())
        
        # Cross-validation metrics
        metrics.update(self.calculate_cv_metrics())
        
        # Directional accuracy
        metrics.update(self.calculate_directional_metrics())
        
        return metrics
    
    def calculate_basic_metrics(self):
        """Calculate enhanced basic performance metrics."""
        metrics = {}
        
        for period in ['train', 'test', 'all']:
            if period == 'all':
                data = self.df
            else:
                data = self.df[self.df['period_type'] == period]
            
            if data.empty:
                continue
                
            actuals = data['actuals']
            predictions = data['predictions'] if period == 'test' else data['fitted_values']
            residuals = data['residuals']
            
            # Skip if predictions are NaN (common for train period predictions)
            valid_mask = ~pd.isna(predictions)
            if not valid_mask.any():
                continue
                
            actuals_valid = actuals[valid_mask]
            predictions_valid = predictions[valid_mask]
            residuals_valid = residuals[valid_mask]
            
            prefix = f"{period}_"
            
            # Basic error metrics
            metrics[f"{prefix}mae"] = mean_absolute_error(actuals_valid, predictions_valid)
            metrics[f"{prefix}rmse"] = np.sqrt(mean_squared_error(actuals_valid, predictions_valid))
            metrics[f"{prefix}mape"] = np.mean(np.abs((actuals_valid - predictions_valid) / actuals_valid)) * 100
            
            # Additional metrics
            metrics[f"{prefix}r2"] = self._calculate_r2(actuals_valid, predictions_valid)
            metrics[f"{prefix}mbe"] = np.mean(residuals_valid)  # Mean Bias Error
            metrics[f"{prefix}smape"] = self._calculate_smape(actuals_valid, predictions_valid)
            metrics[f"{prefix}mase"] = self._calculate_mase(actuals_valid, predictions_valid)
            metrics[f"{prefix}theil_u"] = self._calculate_theil_u(actuals_valid, predictions_valid)
            
            # Residual statistics
            metrics[f"{prefix}residual_std"] = np.std(residuals_valid)
            metrics[f"{prefix}residual_skew"] = stats.skew(residuals_valid)
            metrics[f"{prefix}residual_kurtosis"] = stats.kurtosis(residuals_valid)
            
        return metrics
    
    def calculate_rolling_metrics(self, windows=[7, 14, 30]):
        """Calculate rolling performance metrics."""
        metrics = {}
        
        # Sort by date for rolling calculations
        df_sorted = self.df.sort_index()
        
        for window in windows:
            for period in ['test']:  # Focus on test period for rolling metrics
                period_data = df_sorted[df_sorted['period_type'] == period]
                
                if len(period_data) < window:
                    continue
                
                actuals = period_data['actuals']
                predictions = period_data['predictions']
                
                # Skip if predictions are NaN
                valid_mask = ~pd.isna(predictions)
                if not valid_mask.any():
                    continue
                
                actuals_valid = actuals[valid_mask]
                predictions_valid = predictions[valid_mask]
                
                # Rolling MAE
                rolling_mae = []
                rolling_rmse = []
                rolling_r2 = []
                
                for i in range(window, len(actuals_valid) + 1):
                    window_actuals = actuals_valid.iloc[i-window:i]
                    window_preds = predictions_valid.iloc[i-window:i]
                    
                    rolling_mae.append(mean_absolute_error(window_actuals, window_preds))
                    rolling_rmse.append(np.sqrt(mean_squared_error(window_actuals, window_preds)))
                    rolling_r2.append(self._calculate_r2(window_actuals, window_preds))
                
                if rolling_mae:
                    prefix = f"rolling_{window}d_"
                    metrics[f"{prefix}mae_mean"] = np.mean(rolling_mae)
                    metrics[f"{prefix}mae_std"] = np.std(rolling_mae)
                    metrics[f"{prefix}rmse_mean"] = np.mean(rolling_rmse)
                    metrics[f"{prefix}rmse_std"] = np.std(rolling_rmse)
                    metrics[f"{prefix}r2_mean"] = np.mean(rolling_r2)
                    metrics[f"{prefix}r2_std"] = np.std(rolling_r2)
        
        return metrics
    
    def calculate_statistical_tests(self):
        """Calculate statistical tests on residuals."""
        metrics = {}
        
        for period in ['train', 'test']:
            data = self.df[self.df['period_type'] == period]
            
            if data.empty:
                continue
                
            residuals = data['residuals'].dropna()
            
            if len(residuals) < 3:
                continue
                
            prefix = f"{period}_"
            
            # Normality tests
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                metrics[f"{prefix}shapiro_stat"] = shapiro_stat
                metrics[f"{prefix}shapiro_pvalue"] = shapiro_p
                metrics[f"{prefix}residuals_normal"] = shapiro_p > 0.05
            except:
                pass
            
            try:
                jb_stat, jb_p = stats.jarque_bera(residuals)
                metrics[f"{prefix}jarque_bera_stat"] = jb_stat
                metrics[f"{prefix}jarque_bera_pvalue"] = jb_p
            except:
                pass
            
            # Autocorrelation test (Durbin-Watson)
            try:
                dw_stat = self._durbin_watson(residuals)
                metrics[f"{prefix}durbin_watson"] = dw_stat
                metrics[f"{prefix}no_autocorr"] = 1.5 < dw_stat < 2.5
            except:
                pass
        
        return metrics
    
    def calculate_advanced_performance(self):
        """Calculate advanced performance metrics."""
        metrics = {}
        
        # Model consistency across slices
        slice_metrics = []
        for slice_id in self.df.index.get_level_values('slice_id').unique():
            slice_data = self.df.xs(slice_id, level='slice_id')
            test_data = slice_data[slice_data['period_type'] == 'test']
            
            if not test_data.empty and not pd.isna(test_data['predictions']).all():
                valid_mask = ~pd.isna(test_data['predictions'])
                actuals = test_data['actuals'][valid_mask]
                predictions = test_data['predictions'][valid_mask]
                
                if len(actuals) > 0:
                    slice_mae = mean_absolute_error(actuals, predictions)
                    slice_metrics.append(slice_mae)
        
        if slice_metrics:
            metrics['slice_consistency'] = 1 / (1 + np.std(slice_metrics))  # Higher = more consistent
            metrics['slice_mae_variance'] = np.var(slice_metrics)
            metrics['best_slice_mae'] = np.min(slice_metrics)
            metrics['worst_slice_mae'] = np.max(slice_metrics)
        
        return metrics
    
    def calculate_cv_metrics(self):
        """Calculate cross-validation specific metrics."""
        metrics = {}
        
        # Performance by slice
        slice_performance = {}
        for slice_id in self.df.index.get_level_values('slice_id').unique():
            slice_data = self.df.xs(slice_id, level='slice_id')
            test_data = slice_data[slice_data['period_type'] == 'test']
            
            if not test_data.empty and not pd.isna(test_data['predictions']).all():
                valid_mask = ~pd.isna(test_data['predictions'])
                actuals = test_data['actuals'][valid_mask]
                predictions = test_data['predictions'][valid_mask]
                
                if len(actuals) > 0:
                    slice_performance[slice_id] = {
                        'mae': mean_absolute_error(actuals, predictions),
                        'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
                        'r2': self._calculate_r2(actuals, predictions)
                    }
        
        if slice_performance:
            # Calculate CV statistics
            mae_values = [perf['mae'] for perf in slice_performance.values()]
            rmse_values = [perf['rmse'] for perf in slice_performance.values()]
            r2_values = [perf['r2'] for perf in slice_performance.values()]
            
            metrics['cv_mae_mean'] = np.mean(mae_values)
            metrics['cv_mae_std'] = np.std(mae_values)
            metrics['cv_rmse_mean'] = np.mean(rmse_values)
            metrics['cv_rmse_std'] = np.std(rmse_values)
            metrics['cv_r2_mean'] = np.mean(r2_values)
            metrics['cv_r2_std'] = np.std(r2_values)
            
            # Stability metrics
            metrics['cv_stability_mae'] = 1 - (np.std(mae_values) / np.mean(mae_values))
            metrics['cv_stability_rmse'] = 1 - (np.std(rmse_values) / np.mean(rmse_values))
        
        return metrics
    
    def calculate_directional_metrics(self):
        """Calculate directional accuracy metrics."""
        metrics = {}
        
        test_data = self.df[self.df['period_type'] == 'test']
        
        if test_data.empty or pd.isna(test_data['predictions']).all():
            return metrics
        
        # Sort by date for directional analysis
        test_data = test_data.sort_index()
        valid_mask = ~pd.isna(test_data['predictions'])
        test_data = test_data[valid_mask]
        
        if len(test_data) < 2:
            return metrics
        
        # Calculate directional changes
        actual_direction = np.diff(test_data['actuals']) > 0
        pred_direction = np.diff(test_data['predictions']) > 0
        
        if len(actual_direction) > 0:
            # Hit rate (directional accuracy)
            hit_rate = np.mean(actual_direction == pred_direction)
            metrics['directional_accuracy'] = hit_rate
            
            # Precision and recall for up movements
            true_ups = actual_direction
            pred_ups = pred_direction
            
            if np.any(pred_ups):
                precision_up = np.sum(true_ups & pred_ups) / np.sum(pred_ups)
                metrics['precision_up'] = precision_up
            
            if np.any(true_ups):
                recall_up = np.sum(true_ups & pred_ups) / np.sum(true_ups)
                metrics['recall_up'] = recall_up
        
        return metrics
    
    # Helper methods
    def _calculate_r2(self, actual, predicted):
        """Calculate R-squared."""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _calculate_smape(self, actual, predicted):
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))
    
    def _calculate_mase(self, actual, predicted):
        """Calculate Mean Absolute Scaled Error."""
        naive_forecast = actual.shift(1).dropna()
        actual_aligned = actual[1:]
        
        if len(naive_forecast) == 0:
            return np.nan
            
        mae_naive = np.mean(np.abs(actual_aligned - naive_forecast))
        mae_model = np.mean(np.abs(actual - predicted))
        
        return mae_model / mae_naive if mae_naive != 0 else np.nan
    
    def _calculate_theil_u(self, actual, predicted):
        """Calculate Theil's U statistic."""
        mse = np.mean((actual - predicted) ** 2)
        mse_naive = np.mean((actual[1:] - actual[:-1]) ** 2)
        
        return np.sqrt(mse) / np.sqrt(mse_naive) if mse_naive != 0 else np.nan
    
    def _durbin_watson(self, residuals):
        """Calculate Durbin-Watson statistic."""
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)


def demonstrate_enhanced_metrics(csv_path="resamples_outputs.csv"):
    """Demonstrate the enhanced metrics on your data."""
    
    # Load the data
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'slice_id', 'model_id'])
    
    # Calculate enhanced metrics
    calculator = EnhancedMetricsCalculator(df)
    metrics = calculator.calculate_all_metrics()
    
    # Display results
    print("🔢 Enhanced Statistical Metrics")
    print("=" * 50)
    
    # Group metrics by category
    categories = {
        'Basic Performance': [k for k in metrics.keys() if any(x in k for x in ['mae', 'rmse', 'mape', 'r2']) and 'rolling' not in k],
        'Rolling Metrics': [k for k in metrics.keys() if 'rolling' in k],
        'Statistical Tests': [k for k in metrics.keys() if any(x in k for x in ['shapiro', 'jarque', 'durbin'])],
        'CV Analysis': [k for k in metrics.keys() if 'cv_' in k or 'slice_' in k],
        'Directional Accuracy': [k for k in metrics.keys() if any(x in k for x in ['directional', 'precision', 'recall'])],
        'Advanced Metrics': [k for k in metrics.keys() if any(x in k for x in ['mbe', 'smape', 'mase', 'theil', 'residual'])]
    }
    
    for category, metric_keys in categories.items():
        if metric_keys:
            print(f"\n📊 {category}:")
            for key in sorted(metric_keys):
                value = metrics[key]
                if isinstance(value, bool):
                    print(f"   {key:25s}: {'✓' if value else '✗'}")
                elif isinstance(value, (int, float)):
                    print(f"   {key:25s}: {value:.4f}")
    
    return metrics


if __name__ == "__main__":
    # Run demonstration
    metrics = demonstrate_enhanced_metrics()
    
    print(f"\n📈 Total metrics calculated: {len(metrics)}")
    print("\n💡 These metrics could be added to your dashboard for:")
    print("   • Rolling performance analysis")
    print("   • Statistical validation of model assumptions")
    print("   • Cross-validation stability assessment")
    print("   • Directional prediction accuracy")
    print("   • Advanced model diagnostics") 