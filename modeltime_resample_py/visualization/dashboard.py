"""Enhanced interactive dashboard for exploring model resampling results."""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro, kstest
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import warnings


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    non_zero_mask = denominator != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    return np.mean(np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / denominator[non_zero_mask]) * 100


class ModelComparisonAnalyzer:
    """Advanced statistical analysis for model comparison."""
    
    @staticmethod
    def wilcoxon_test(values1, values2):
        """Perform Wilcoxon signed-rank test for paired samples."""
        try:
            if len(values1) != len(values2) or len(values1) < 3:
                return np.nan, np.nan
            
            # Remove pairs where either value is NaN
            mask = ~(pd.isna(values1) | pd.isna(values2))
            clean_vals1 = np.array(values1)[mask]
            clean_vals2 = np.array(values2)[mask]
            
            if len(clean_vals1) < 3:
                return np.nan, np.nan
                
            statistic, p_value = stats.wilcoxon(clean_vals1, clean_vals2, alternative='two-sided')
            return statistic, p_value
        except Exception:
            return np.nan, np.nan
    
    @staticmethod
    def effect_size(values1, values2):
        """Calculate Cohen's d effect size."""
        try:
            arr1, arr2 = np.array(values1), np.array(values2)
            # Remove NaN values
            arr1, arr2 = arr1[~pd.isna(arr1)], arr2[~pd.isna(arr2)]
            
            if len(arr1) < 2 or len(arr2) < 2:
                return np.nan
                
            pooled_std = np.sqrt(((len(arr1) - 1) * np.var(arr1, ddof=1) + 
                                 (len(arr2) - 1) * np.var(arr2, ddof=1)) / 
                                (len(arr1) + len(arr2) - 2))
            
            if pooled_std == 0:
                return np.nan
                
            return (np.mean(arr1) - np.mean(arr2)) / pooled_std
        except Exception:
            return np.nan
    
    @staticmethod
    def significance_stars(p_value):
        """Convert p-value to significance stars."""
        if pd.isna(p_value):
            return ''
        elif p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    
    @staticmethod
    def dominance_score(values1, values2):
        """Calculate how often model 1 beats model 2 (for error metrics, lower is better)."""
        try:
            arr1, arr2 = np.array(values1), np.array(values2)
            mask = ~(pd.isna(arr1) | pd.isna(arr2))
            clean_vals1, clean_vals2 = arr1[mask], arr2[mask]
            
            if len(clean_vals1) == 0:
                return np.nan
                
            return np.mean(clean_vals1 < clean_vals2) * 100  # Percentage of times model1 < model2
        except Exception:
            return np.nan
    
    @staticmethod
    def skill_score(values, baseline_values):
        """Calculate skill score relative to baseline."""
        try:
            vals = np.array(values)
            baseline = np.array(baseline_values)
            
            # Remove NaN values
            mask = ~(pd.isna(vals) | pd.isna(baseline))
            clean_vals = vals[mask]
            clean_baseline = baseline[mask]
            
            if len(clean_vals) == 0 or np.mean(clean_baseline) == 0:
                return np.nan
                
            # Skill score = 1 - (model_error / baseline_error)
            # For error metrics, lower is better, so higher skill score is better
            return 1 - (np.mean(clean_vals) / np.mean(clean_baseline))
        except Exception:
            return np.nan


class ModelScorer:
    """Multi-criteria ranking system with user-defined weights."""
    
    def __init__(self, metrics_weights=None):
        """Initialize with optional metric weights."""
        self.default_weights = {
            'mae': 0.25,
            'rmse': 0.25, 
            'mape': 0.25,
            'smape': 0.25,
            'r2': 0.0,  # R2 is inverse (higher is better)
            'mean_error': 0.0,
            'std_error': 0.0
        }
        self.weights = metrics_weights or self.default_weights
    
    def calculate_composite_score(self, metrics_df, model_col='model_id'):
        """Calculate composite score for each model based on weighted metrics."""
        scores = {}
        
        # Check if metrics_df has model_col as a column or in index
        if model_col in metrics_df.columns:
            # Regular DataFrame with model_col as column
            available_metrics = [m for m in self.weights.keys() if m in metrics_df.columns and self.weights[m] > 0]
            
            if not available_metrics:
                return scores
            
            # Normalize metrics to 0-1 scale (lower is better for error metrics)
            normalized_df = metrics_df.copy()
            
            for metric in available_metrics:
                values = metrics_df[metric].dropna()
                if len(values) > 0:
                    if metric == 'r2':  # Higher is better for R2
                        min_val, max_val = values.min(), values.max()
                        if max_val != min_val:
                            normalized_df[f'{metric}_norm'] = (values - min_val) / (max_val - min_val)
                        else:
                            normalized_df[f'{metric}_norm'] = 0.5
                    else:  # Lower is better for error metrics
                        min_val, max_val = values.min(), values.max()
                        if max_val != min_val:
                            normalized_df[f'{metric}_norm'] = 1 - ((values - min_val) / (max_val - min_val))
                        else:
                            normalized_df[f'{metric}_norm'] = 0.5
            
            # Calculate weighted composite scores
            for model in metrics_df[model_col].unique():
                model_data = normalized_df[normalized_df[model_col] == model]
                if not model_data.empty:
                    weighted_sum = 0
                    total_weight = 0
                    
                    for metric in available_metrics:
                        norm_col = f'{metric}_norm'
                        if norm_col in model_data.columns:
                            metric_value = model_data[norm_col].mean()
                            if not pd.isna(metric_value):
                                weighted_sum += metric_value * self.weights[metric]
                                total_weight += self.weights[metric]
                    
                    if total_weight > 0:
                        scores[model] = weighted_sum / total_weight
                    else:
                        scores[model] = np.nan
        
        return scores
    
    def calculate_dominance_matrix(self, df_with_multiindex, metric_col):
        """Calculate pairwise dominance matrix for MultiIndex DataFrame."""
        # Get unique models from the MultiIndex
        models = df_with_multiindex.index.get_level_values('model_id').unique()
        dominance_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
        
        # Calculate metric values for each model first
        model_values = {}
        for model in models:
            model_data = df_with_multiindex[df_with_multiindex.index.get_level_values('model_id') == model]
            if not model_data.empty:
                actuals = model_data['actuals'].dropna()
                predictions = model_data['predictions'].dropna()
                common_idx = actuals.index.intersection(predictions.index)
                
                if len(common_idx) > 0:
                    if metric_col == 'mae':
                        values = np.abs(actuals.loc[common_idx] - predictions.loc[common_idx])
                    elif metric_col == 'rmse':
                        values = (actuals.loc[common_idx] - predictions.loc[common_idx]) ** 2
                        values = np.sqrt(values)
                    elif metric_col == 'mape':
                        mask = actuals.loc[common_idx] != 0
                        clean_actuals = actuals.loc[common_idx][mask]
                        clean_preds = predictions.loc[common_idx][mask]
                        if len(clean_actuals) > 0:
                            values = np.abs((clean_actuals - clean_preds) / clean_actuals) * 100
                        else:
                            values = np.array([])
                    elif metric_col == 'smape':
                        denom = (np.abs(actuals.loc[common_idx]) + np.abs(predictions.loc[common_idx])) / 2
                        mask = denom != 0
                        clean_actuals = actuals.loc[common_idx][mask]
                        clean_preds = predictions.loc[common_idx][mask]
                        clean_denom = denom[mask]
                        if len(clean_actuals) > 0:
                            values = np.abs(clean_actuals - clean_preds) / clean_denom * 100
                        else:
                            values = np.array([])
                    else:
                        values = np.array([])
                    
                    model_values[model] = values
        
        # Now calculate pairwise dominance
        for model1 in models:
            for model2 in models:
                if model1 == model2:
                    dominance_matrix.loc[model1, model2] = 50.0  # Tie
                elif model1 in model_values and model2 in model_values:
                    dominance_score = ModelComparisonAnalyzer.dominance_score(
                        model_values[model1], model_values[model2]
                    )
                    dominance_matrix.loc[model1, model2] = dominance_score
                else:
                    dominance_matrix.loc[model1, model2] = np.nan
        
        return dominance_matrix


class RollingResidualAnalyzer:
    """Advanced rolling residual analysis for time series models."""
    
    @staticmethod
    def calculate_rolling_metrics(residuals, window=30):
        """Calculate rolling statistical metrics for residuals."""
        if len(residuals) < window:
            window = max(5, len(residuals) // 2)
        
        metrics = {
            'rolling_mean': residuals.rolling(window=window, min_periods=5).mean(),
            'rolling_std': residuals.rolling(window=window, min_periods=5).std(),
            'rolling_skew': residuals.rolling(window=window, min_periods=5).skew(),
            'rolling_kurt': residuals.rolling(window=window, min_periods=5).kurt(),
            'rolling_var': residuals.rolling(window=window, min_periods=5).var()
        }
        
        return metrics
    
    @staticmethod
    def autocorrelation_test(residuals, lags=10):
        """Perform Ljung-Box test for autocorrelation."""
        try:
            if len(residuals) > lags + 5:
                result = acorr_ljungbox(residuals, lags=lags, return_df=True)
                return {
                    'ljung_box_stat': result['lb_stat'].iloc[-1],
                    'ljung_box_pvalue': result['lb_pvalue'].iloc[-1],
                    'is_white_noise': result['lb_pvalue'].iloc[-1] > 0.05
                }
        except Exception:
            pass
        return {'ljung_box_stat': np.nan, 'ljung_box_pvalue': np.nan, 'is_white_noise': None}
    
    @staticmethod
    def normality_tests(residuals):
        """Perform multiple normality tests on residuals."""
        tests = {}
        
        try:
            if len(residuals) >= 8:
                # Shapiro-Wilk test (best for small samples)
                stat, pval = shapiro(residuals)
                tests['shapiro_stat'] = stat
                tests['shapiro_pvalue'] = pval
        except Exception:
            tests['shapiro_stat'] = np.nan
            tests['shapiro_pvalue'] = np.nan
        
        try:
            if len(residuals) >= 20:
                # Jarque-Bera test
                stat, pval = jarque_bera(residuals)
                tests['jarque_bera_stat'] = stat
                tests['jarque_bera_pvalue'] = pval
        except Exception:
            tests['jarque_bera_stat'] = np.nan
            tests['jarque_bera_pvalue'] = np.nan
        
        try:
            if len(residuals) >= 8:
                # D'Agostino's normality test
                stat, pval = normaltest(residuals)
                tests['dagostino_stat'] = stat
                tests['dagostino_pvalue'] = pval
        except Exception:
            tests['dagostino_stat'] = np.nan
            tests['dagostino_pvalue'] = np.nan
        
        return tests
    
    @staticmethod
    def heteroscedasticity_test(residuals, fitted_values):
        """Test for heteroscedasticity using Breusch-Pagan test."""
        try:
            if len(residuals) >= 10 and len(fitted_values) >= 10:
                # Create design matrix (constant + fitted values)
                X = np.column_stack([np.ones(len(fitted_values)), fitted_values])
                lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X)
                
                return {
                    'breusch_pagan_lm': lm,
                    'breusch_pagan_pvalue': lm_pvalue,
                    'is_homoscedastic': lm_pvalue > 0.05
                }
        except Exception:
            pass
        
        return {
            'breusch_pagan_lm': np.nan,
            'breusch_pagan_pvalue': np.nan,
            'is_homoscedastic': None
        }
    
    @staticmethod
    def durbin_watson_test(residuals):
        """Calculate Durbin-Watson statistic for serial correlation."""
        try:
            if len(residuals) >= 10:
                dw_stat = durbin_watson(residuals)
                # DW interpretation: ~2 = no autocorr, <2 = positive, >2 = negative
                interpretation = 'No autocorrelation'
                if dw_stat < 1.5:
                    interpretation = 'Positive autocorrelation'
                elif dw_stat > 2.5:
                    interpretation = 'Negative autocorrelation'
                
                return {
                    'durbin_watson_stat': dw_stat,
                    'autocorr_interpretation': interpretation
                }
        except Exception:
            pass
        
        return {
            'durbin_watson_stat': np.nan,
            'autocorr_interpretation': 'Unknown'
        }


class ModelDiagnostics:
    """Comprehensive model diagnostic analysis."""
    
    @staticmethod
    def calculate_forecast_bias_metrics(residuals):
        """Calculate various forecast bias metrics."""
        residuals_clean = residuals.dropna()
        
        if len(residuals_clean) == 0:
            return {}
        
        metrics = {
            'mean_bias': np.mean(residuals_clean),
            'median_bias': np.median(residuals_clean),
            'bias_variance': np.var(residuals_clean),
            'systematic_bias_test': abs(np.mean(residuals_clean)) > 2 * (np.std(residuals_clean) / np.sqrt(len(residuals_clean))),
            'positive_bias_pct': np.mean(residuals_clean > 0) * 100,
            'negative_bias_pct': np.mean(residuals_clean < 0) * 100
        }
        
        return metrics
    
    @staticmethod
    def outlier_detection_metrics(residuals, threshold=3):
        """Detect outliers in residuals using multiple methods."""
        residuals_clean = residuals.dropna()
        
        if len(residuals_clean) == 0:
            return {}
        
        # Z-score method
        z_scores = np.abs((residuals_clean - np.mean(residuals_clean)) / np.std(residuals_clean))
        z_outliers = z_scores > threshold
        
        # IQR method
        Q1 = np.percentile(residuals_clean, 25)
        Q3 = np.percentile(residuals_clean, 75)
        IQR = Q3 - Q1
        iqr_outliers = (residuals_clean < (Q1 - 1.5 * IQR)) | (residuals_clean > (Q3 + 1.5 * IQR))
        
        return {
            'z_score_outliers_count': int(np.sum(z_outliers)),
            'z_score_outliers_pct': np.mean(z_outliers) * 100,
            'iqr_outliers_count': int(np.sum(iqr_outliers)),
            'iqr_outliers_pct': np.mean(iqr_outliers) * 100,
            'max_absolute_residual': np.max(np.abs(residuals_clean)),
            'residual_range': np.max(residuals_clean) - np.min(residuals_clean)
        }
    
    @staticmethod
    def model_stability_metrics(residuals, window=30):
        """Calculate model stability metrics over time."""
        if len(residuals) < window * 2:
            return {}
        
        # Calculate rolling standard deviation
        rolling_std = residuals.rolling(window=window, min_periods=10).std()
        
        # Stability metrics
        std_stability = np.std(rolling_std.dropna()) / np.mean(rolling_std.dropna()) if len(rolling_std.dropna()) > 0 else np.nan
        
        # Trend in residual volatility
        time_idx = np.arange(len(rolling_std.dropna()))
        if len(time_idx) > 5:
            try:
                slope, _, _, p_value, _ = stats.linregress(time_idx, rolling_std.dropna())
                volatility_trend = 'Increasing' if slope > 0 and p_value < 0.05 else 'Stable' if p_value >= 0.05 else 'Decreasing'
            except:
                volatility_trend = 'Unknown'
                p_value = np.nan
        else:
            volatility_trend = 'Unknown'
            p_value = np.nan
        
        return {
            'volatility_stability_coeff': std_stability,
            'volatility_trend': volatility_trend,
            'volatility_trend_pvalue': p_value,
            'max_volatility_period': np.argmax(rolling_std) if len(rolling_std) > 0 else np.nan,
            'min_volatility_period': np.argmin(rolling_std) if len(rolling_std) > 0 else np.nan
        }


class CoefficientAnalyzer:
    """Analysis of model coefficients and statistical significance."""
    
    def __init__(self):
        self.coefficient_data = {}
        self.p_value_data = {}
        self.significance_data = {}
    
    def add_model_results(self, model_id, slice_id, coefficients=None, p_values=None, std_errors=None):
        """Add model coefficient results for analysis."""
        # This would be called when we have access to model objects
        # For now, we'll simulate coefficient analysis from residuals
        key = f"{model_id}_{slice_id}"
        
        if coefficients is not None:
            self.coefficient_data[key] = coefficients
        
        if p_values is not None:
            self.p_value_data[key] = p_values
    
    def simulate_coefficient_analysis(self, model_id, slice_id, actuals, predictions):
        """Simulate coefficient analysis from predictions (for demonstration)."""
        # This is a simplified simulation - in real implementation,
        # you would extract actual coefficients from fitted models
        
        try:
            # Simple linear relationship analysis
            if len(actuals) > 5 and len(predictions) > 5:
                slope, intercept, r_value, p_value, std_err = stats.linregress(actuals, predictions)
                
                key = f"{model_id}_{slice_id}"
                
                # Simulate coefficients (in practice, these would come from the actual model)
                self.coefficient_data[key] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2
                }
                
                self.p_value_data[key] = {
                    'slope_pvalue': p_value,
                    'intercept_pvalue': 0.001 if abs(intercept) > std_err else 0.5  # Simplified
                }
                
                return True
        except Exception:
            pass
        
        return False
    
    def get_max_p_values_across_slices(self, exclude_intercept=True):
        """Get maximum p-values for each coefficient across all slices."""
        if not self.p_value_data:
            return {}
        
        max_p_values = {}
        
        # Get all coefficient names
        all_coeffs = set()
        for p_vals in self.p_value_data.values():
            all_coeffs.update(p_vals.keys())
        
        if exclude_intercept:
            all_coeffs = {coeff for coeff in all_coeffs if 'intercept' not in coeff.lower()}
        
        # Find maximum p-value for each coefficient
        for coeff in all_coeffs:
            p_values = []
            for p_vals in self.p_value_data.values():
                if coeff in p_vals and not pd.isna(p_vals[coeff]):
                    p_values.append(p_vals[coeff])
            
            if p_values:
                max_p_values[coeff] = {
                    'max_pvalue': max(p_values),
                    'min_pvalue': min(p_values),
                    'mean_pvalue': np.mean(p_values),
                    'significant_slices': sum(1 for p in p_values if p < 0.05),
                    'total_slices': len(p_values)
                }
        
        return max_p_values
    
    def get_coefficient_stability(self):
        """Analyze coefficient stability across slices."""
        if not self.coefficient_data:
            return {}
        
        stability_metrics = {}
        
        # Get all coefficient names
        all_coeffs = set()
        for coeffs in self.coefficient_data.values():
            all_coeffs.update(coeffs.keys())
        
        # Calculate stability for each coefficient
        for coeff in all_coeffs:
            values = []
            for coeffs in self.coefficient_data.values():
                if coeff in coeffs and not pd.isna(coeffs[coeff]):
                    values.append(coeffs[coeff])
            
            if len(values) > 1:
                stability_metrics[coeff] = {
                    'mean_value': np.mean(values),
                    'std_value': np.std(values),
                    'coefficient_of_variation': np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else np.inf,
                    'min_value': min(values),
                    'max_value': max(values),
                    'range': max(values) - min(values),
                    'slices_count': len(values)
                }
        
        return stability_metrics


class TemporalAggregator:
    """Simple temporal aggregation utilities for dashboard metrics."""
    
    @staticmethod
    def aggregate_by_temporal_granularity(df, granularity):
        """
        Aggregate DataFrame by temporal granularity.
        
        Args:
            df: DataFrame with date index
            granularity: One of 'year', 'quarter', 'month', 'week', 'day'
            
        Returns:
            DataFrame grouped by temporal granularity
        """
        if granularity not in ['year', 'quarter', 'month', 'month_groups', 'week', 'day']:
            return df
            
        # Get date index level
        dates = df.index.get_level_values('date')
        
        if granularity == 'year':
            period_key = dates.year
        elif granularity == 'quarter':
            period_key = dates.to_period('Q')
        elif granularity == 'month':
            period_key = dates.to_period('M')
        elif granularity == 'month_groups':
            period_key = dates.strftime('%B')  # Full month names (January, February, etc.)
        elif granularity == 'week':
            period_key = dates.to_period('W')
        else:  # day
            period_key = dates.date
            
        # Add temporal grouping to DataFrame
        df_with_temporal = df.copy()
        df_with_temporal['temporal_group'] = period_key
        
        return df_with_temporal
    
    @staticmethod
    def get_available_granularities(df):
        """Get available temporal granularities based on data timespan."""
        dates = df.index.get_level_values('date')
        min_date, max_date = dates.min(), dates.max()
        timespan = (max_date - min_date).days
        
        granularities = [('day', 'Daily')]
        
        if timespan > 7:
            granularities.append(('week', 'Weekly'))
        if timespan > 30:
            granularities.append(('month', 'Monthly'))
        if timespan > 90:
            granularities.append(('quarter', 'Quarterly'))
        if timespan > 365:
            granularities.append(('year', 'Yearly'))
            
        return granularities


class PerformanceColorizer:
    """Simple color coding utilities for model performance ranking."""
    
    @staticmethod
    def get_color_for_rank(rank, total_models, metric_name):
        """
        Get color based on model rank for a specific metric.
        
        Args:
            rank: Model rank (1 = best, higher = worse)
            total_models: Total number of models
            metric_name: Name of the metric (for direction logic)
            
        Returns:
            RGB color string
        """
        if total_models <= 1:
            return 'rgb(248, 249, 250)'  # Light gray for single model
            
        # Calculate color intensity (0 = best, 1 = worst)
        intensity = (rank - 1) / (total_models - 1)
        
        # Green (best) to Red (worst) gradient
        if intensity <= 0.5:
            # Green to Yellow
            red = int(255 * (intensity * 2))
            green = 255
            blue = 0
        else:
            # Yellow to Red
            red = 255
            green = int(255 * (2 - intensity * 2))
            blue = 0
            
        return f'rgb({red}, {green}, {blue})'
    
    @staticmethod
    def rank_models_by_metric(df, metric_column):
        """
        Rank models by metric performance.
        
        Args:
            df: DataFrame with model performance metrics
            metric_column: Column name to rank by
            
        Returns:
            DataFrame with added 'rank' column
        """
        if metric_column not in df.columns or df.empty:
            return df
            
        df_ranked = df.copy()
        
        # For most metrics, lower is better (except R²)
        ascending = metric_column.lower() != 'r2'
        
        # Rank models by metric value
        df_ranked['rank'] = df_ranked[metric_column].rank(ascending=ascending)
            
        return df_ranked


class EnhancedResamplesDashboard:
    """Enhanced interactive dashboard for exploring time series model results."""
    
    def __init__(
        self,
        resamples_df: pd.DataFrame,
        accuracy_df: Optional[pd.DataFrame] = None,
        title: str = "Time Series Model Analysis Dashboard"
    ):
        """
        Initialize the enhanced dashboard.
        
        Args:
            resamples_df: Output from fit_resamples
            accuracy_df: Output from resample_accuracy (optional)
            title: Dashboard title
        """
        self.resamples_df = resamples_df.copy()
        self.accuracy_df = accuracy_df.copy() if accuracy_df is not None else None
        self.title = title
        
        # Ensure proper index and data structure
        self._prepare_data()
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._setup_layout()
        self._setup_callbacks()
        
        # Add client-side callback to trigger initial load
        self.app.clientside_callback(
            """
            function(id) {
                if (id) {
                    // Trigger the update button after a short delay
                    setTimeout(function() {
                        var updateBtn = document.getElementById('update-button');
                        if (updateBtn) {
                            updateBtn.click();
                        }
                    }, 1000);
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('update-button', 'n_clicks'),
            Input('main-tabs', 'id')
        )
    
    def _prepare_data(self):
        """Prepare and validate data for dashboard use."""
        # Ensure the index is sorted for proper slicing and plotting
        self.resamples_df = self.resamples_df.sort_index()
        
        # Add residuals if not present
        if 'residuals' not in self.resamples_df.columns:
            train_mask = self.resamples_df['period_type'] == 'train'
            test_mask = self.resamples_df['period_type'] == 'test'
            
            residuals = pd.Series(index=self.resamples_df.index, dtype=float)
            residuals[train_mask] = (self.resamples_df.loc[train_mask, 'actuals'] - 
                                   self.resamples_df.loc[train_mask, 'fitted_values'])
            residuals[test_mask] = (self.resamples_df.loc[test_mask, 'actuals'] - 
                                  self.resamples_df.loc[test_mask, 'predictions'])
            
            self.resamples_df['residuals'] = residuals
        
        # Get unique values for filters
        self.unique_models = self.resamples_df.index.get_level_values('model_id').unique().tolist()
        self.unique_slices = self.resamples_df.index.get_level_values('slice_id').unique().tolist()
        
        # Date range
        dates = self.resamples_df.index.get_level_values('date')
        self.min_date = dates.min()
        self.max_date = dates.max()
        
        # Create model options for filtering
        self.model_options = [
            {'label': model, 'value': model} for model in self.unique_models
        ]
        
        # Create split/model combinations
        unique_groups = self.resamples_df.index.droplevel('date').unique().tolist()
        self.split_model_options = [
            {'label': f"Slice {s}, Model {m}", 'value': f"{s}_{m}"} 
            for s, m in unique_groups
        ]
        self.split_model_options.insert(0, {'label': 'All Splits - Aggregated View', 'value': 'all_aggregated'})
        self.split_model_options.insert(1, {'label': 'All Splits - Separate Plots', 'value': 'all_separate'})
        
        # Create split options for multi-select
        self.split_options = [
            {'label': f"Slice {s}", 'value': s} for s in self.unique_slices
        ]
    
    def _setup_layout(self):
        """Create the enhanced dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(self.title, className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Summary Statistics Bar (moved to top)
            dbc.Row([
                dbc.Col([
                    dbc.Alert(
                        id='summary-stats',
                        color="info",
                        className="text-center mb-4"
                    )
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Controls & Filters"),
                        dbc.CardBody([
                            # Date range picker
                            dbc.Label("📅 Select Date Range:"),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                min_date_allowed=self.min_date.date(),
                                max_date_allowed=self.max_date.date(),
                                initial_visible_month=self.min_date.date(),
                                start_date=self.min_date.date(),
                                end_date=self.max_date.date(),
                                className="mb-3"
                            ),
                            
                            # Model selector
                            dbc.Label("🤖 Select Models:"),
                            dcc.Dropdown(
                                id='model-selector',
                                options=self.model_options,
                                value=self.unique_models,  # Select all by default
                                multi=True,
                                clearable=False,
                                className="mb-3"
                            ),
                            
                            # Split selector
                            dbc.Label("🎯 Select Splits:"),
                            dcc.Dropdown(
                                id='split-selector',
                                options=self.split_options,
                                value=self.unique_slices,  # Select all by default
                                multi=True,
                                clearable=False,
                                className="mb-3"
                            ),
                            
                            # View mode selector
                            dbc.Label("📊 View Mode:"),
                            dcc.Dropdown(
                                id='view-mode-selector',
                                options=[
                                    {'label': 'All Splits - Aggregated View', 'value': 'all_aggregated'},
                                    {'label': 'All Splits - Separate Plots', 'value': 'all_separate'}
                                ],
                                value='all_aggregated',
                                clearable=False,
                                className="mb-3"
                            ),
                            
                            # Performance metrics selector
                            dbc.Label("📈 Select Performance Metrics:"),
                            dcc.Dropdown(
                                id='metric-selector',
                                options=[
                                    {'label': 'MAE (Mean Absolute Error)', 'value': 'mae'},
                                    {'label': 'RMSE (Root Mean Squared Error)', 'value': 'rmse'},
                                    {'label': 'MAPE (Mean Absolute Percentage Error)', 'value': 'mape'}
                                ],
                                value=['mae', 'rmse'],
                                multi=True,
                                className="mb-3"
                            ),
                            
                            # View options
                            dbc.Label("👁️ Display Options:"),
                            dbc.Checklist(
                                id='display-options',
                                options=[
                                    {'label': 'Show Train Period', 'value': 'show_train'},
                                    {'label': 'Show Test Period', 'value': 'show_test'},
                                    {'label': 'Show Residuals', 'value': 'show_residuals'},
                                    {'label': 'Show Confidence Bands', 'value': 'show_confidence'}
                                ],
                                value=['show_train', 'show_test'],
                                className="mb-3"
                            ),
                            
                            # Update button
                            dbc.Button(
                                '🔄 Update View / Calculate Metrics',
                                id='update-button',
                                n_clicks=0,
                                color="primary",
                                className="w-100 mb-3"
                            ),
                            
                            # Export button
                            dbc.Button(
                                '💾 Export Data',
                                id='export-button',
                                color="secondary",
                                className="w-100"
                            ),
                            dcc.Download(id="download-data")
                        ])
                    ])
                ], md=3),
                
                # Main content area
                dbc.Col([
                    # Tabs for different views
                    dcc.Tabs(id='main-tabs', value='plot-tab', children=[
                        dcc.Tab(label='📊 Time Series Plot', value='plot-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Time Series Visualization"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-plot",
                                        type="default",
                                        children=dcc.Graph(
                                            id='resample-plot',
                                            figure=go.Figure(),
                                            style={'height': '600px'}
                                        )
                                    )
                                ])
                            ])
                        ]),
                        
                        dcc.Tab(label='📈 Performance Metrics', value='metrics-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Model Performance Analysis"),
                                dbc.CardBody([
                                    # Performance controls
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("📊 Metrics Granularity:"),
                                            dcc.Dropdown(
                                                id='metrics-granularity-dropdown',
                                                options=[
                                                    {'label': 'Overall', 'value': 'overall'},
                                                    {'label': 'By Model', 'value': 'by_model'},
                                                    {'label': 'By Split', 'value': 'by_split'},
                                                    {'label': 'By Model & Split', 'value': 'by_model_split'}
                                                ],
                                                value='by_model',
                                                clearable=False,
                                                className="mb-3"
                                            )
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("🎯 Split Type Filter:"),
                                            dcc.Dropdown(
                                                id='metrics-split-type-filter-dropdown',
                                                options=[
                                                    {'label': 'All', 'value': 'all'},
                                                    {'label': 'Train Only', 'value': 'train'},
                                                    {'label': 'Test Only', 'value': 'test'}
                                                ],
                                                value='all',
                                                clearable=False,
                                                className="mb-3"
                                            )
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("📋 Comparison Mode:"),
                                            dbc.Switch(
                                                id='comparison-mode-switch',
                                                label="Enable Comparison",
                                                value=False,
                                                className="mb-3"
                                            )
                                        ], md=4)
                                    ]),
                                    
                                    # Temporal granularity controls
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("📅 Temporal Granularity:"),
                                            dcc.Dropdown(
                                                id='temporal-granularity-dropdown',
                                                options=[
                                                    {'label': 'None (All Data)', 'value': 'none'},
                                                    {'label': 'Daily', 'value': 'day'},
                                                    {'label': 'Weekly', 'value': 'week'},
                                                    {'label': 'Monthly', 'value': 'month'},
                                                    {'label': 'Month Groups (Jan, Feb, Mar...)', 'value': 'month_groups'},
                                                    {'label': 'Quarterly', 'value': 'quarter'},
                                                    {'label': 'Yearly', 'value': 'year'}
                                                ],
                                                value='none',
                                                clearable=False,
                                                className="mb-3"
                                            )
                                        ], md=6),
                                        dbc.Col([
                                            html.Div(id='temporal-granularity-info', 
                                                className="mt-4",
                                                children=[
                                                    html.Small("Select temporal grouping for metric aggregation", 
                                                             className="text-muted")
                                                ])
                                        ], md=6)
                                    ]),
                                    
                                    # Baseline model selector (shown when comparison mode is on)
                                    html.Div(id='performance-baseline-container', children=[
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("🎯 Baseline Model:"),
                                                dcc.Dropdown(
                                                    id='performance-baseline-dropdown',
                                                    options=[],
                                                    value=None,
                                                    clearable=False,
                                                    disabled=True,
                                                    className="mb-3"
                                                )
                                            ], md=6),
                                            dbc.Col([
                                                html.Div(id='comparison-mode-info', className="mt-4")
                                            ], md=6)
                                        ])
                                    ]),
                                    
                                    # Metrics column selector
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("📊 Select Metric Columns:"),
                                            dcc.Dropdown(
                                                id='metrics-column-toggle-dropdown',
                                                options=[
                                                    {'label': 'MAE', 'value': 'mae'},
                                                    {'label': 'RMSE', 'value': 'rmse'},
                                                    {'label': 'MAPE', 'value': 'mape'},
                                                    {'label': 'sMAPE', 'value': 'smape'},
                                                    {'label': 'R²', 'value': 'r2'},
                                                    {'label': 'Mean Error', 'value': 'mean_error'},
                                                    {'label': 'Std Error', 'value': 'std_error'}
                                                ],
                                                value='mae',
                                                multi=False,
                                                clearable=False,
                                                className="mb-3"
                                            )
                                        ])
                                    ]),
                                    
                                    # Performance metrics table container
                                    dcc.Loading(
                                        id="loading-metrics",
                                        type="default",
                                        children=[
                                            html.Div(id='metrics-summary', className="mb-4"),
                                            html.Div(id='performance-metrics-table-container')
                                        ]
                                    )
                                ])
                            ])
                        ]),
                        
                        dcc.Tab(label='🔍 Residual Analysis', value='residuals-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Residual Analysis"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-residuals",
                                        type="default",
                                        children=dcc.Graph(
                                            id='residuals-plot',
                                            style={'height': '600px'}
                                        )
                                    )
                                ])
                            ])
                        ]),
                        
                        dcc.Tab(label='📋 Data Table', value='data-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Raw Data Explorer"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-table",
                                        type="default",
                                        children=html.Div(id='data-table-content')
                                    )
                                ])
                            ])
                        ]),
                        
                        dcc.Tab(label='📊 Model Comparison', value='comparison-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Advanced Model Comparison Dashboard"),
                                dbc.CardBody([
                                    # Enhanced comparison controls
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("🎯 Comparison Type:"),
                                            dcc.Dropdown(
                                                id='comparison-type-dropdown',
                                                options=[
                                                    {'label': 'Performance Summary', 'value': 'summary'},
                                                    {'label': 'Pairwise Comparison Matrix', 'value': 'pairwise'},
                                                    {'label': 'Dominance Analysis', 'value': 'dominance'},
                                                    {'label': 'Multi-Metric Ranking', 'value': 'ranking'}
                                                ],
                                                value='summary',
                                                clearable=False,
                                                className="mb-3"
                                            )
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("📏 Analysis Metric:"),
                                            dcc.Dropdown(
                                                id='comparison-metric-dropdown',
                                                options=[
                                                    {'label': 'MAE', 'value': 'mae'},
                                                    {'label': 'RMSE', 'value': 'rmse'},
                                                    {'label': 'MAPE', 'value': 'mape'},
                                                    {'label': 'sMAPE', 'value': 'smape'}
                                                ],
                                                value='mae',
                                                clearable=False,
                                                className="mb-3"
                                            )
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("⚙️ Baseline Model:"),
                                            dcc.Dropdown(
                                                id='comparison-baseline-dropdown',
                                                options=[],
                                                value=None,
                                                clearable=True,
                                                className="mb-3"
                                            )
                                        ], md=4)
                                    ]),
                                    
                                    # Metric weights section (for ranking)
                                    html.Div(id='metric-weights-section', children=[
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("⚖️ Metric Weights (for Multi-Metric Ranking):"),
                                                html.Div(id='metric-weights-sliders', children=[
                                                    dbc.Row([
                                                        dbc.Col([
                                                            html.Label('MAE Weight:'),
                                                            dcc.Slider(id='mae-weight', min=0, max=1, step=0.1, value=0.25, marks={0: '0', 0.5: '0.5', 1: '1'})
                                                        ], md=3),
                                                        dbc.Col([
                                                            html.Label('RMSE Weight:'),
                                                            dcc.Slider(id='rmse-weight', min=0, max=1, step=0.1, value=0.25, marks={0: '0', 0.5: '0.5', 1: '1'})
                                                        ], md=3),
                                                        dbc.Col([
                                                            html.Label('MAPE Weight:'),
                                                            dcc.Slider(id='mape-weight', min=0, max=1, step=0.1, value=0.25, marks={0: '0', 0.5: '0.5', 1: '1'})
                                                        ], md=3),
                                                        dbc.Col([
                                                            html.Label('sMAPE Weight:'),
                                                            dcc.Slider(id='smape-weight', min=0, max=1, step=0.1, value=0.25, marks={0: '0', 0.5: '0.5', 1: '1'})
                                                        ], md=3)
                                                    ])
                                                ])
                                            ])
                                        ])
                                    ], style={'display': 'none'}),
                                    
                                    html.Hr(),
                                    
                                    dcc.Loading(
                                        id="loading-comparison",
                                        type="default",
                                        children=html.Div(id='comparison-content')
                                    )
                                ])
                            ])
                        ])
                    ])
                ], md=9)
            ])
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Set up enhanced dashboard callbacks."""
        
        @self.app.callback(
            [Output('resample-plot', 'figure'),
             Output('metrics-summary', 'children'),
             Output('residuals-plot', 'figure'),
             Output('data-table-content', 'children'),
             Output('summary-stats', 'children')],
            [Input('update-button', 'n_clicks'),
             Input('display-options', 'value'),
             Input('view-mode-selector', 'value')],
            [State('date-picker-range', 'start_date'),
             State('date-picker-range', 'end_date'),
             State('model-selector', 'value'),
             State('split-selector', 'value'),
             State('metric-selector', 'value')]
        )
        def update_dashboard(n_clicks, display_options, view_mode, start_date_str, end_date_str, selected_models, 
                           selected_splits, selected_metrics):
            
            if n_clicks == 0 or not selected_metrics or not selected_models or not selected_splits:
                empty_fig = go.Figure()
                empty_fig.update_layout(title_text="Select options and click 'Update' to view analysis.")
                return (empty_fig, "Select metrics to view summary", empty_fig, 
                       "No data selected", "Click Update to load data")
            
            # Filter data
            filtered_df = self._filter_data(start_date_str, end_date_str, selected_models, selected_splits)
            
            if filtered_df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(title_text="No data available for the selected filters.")
                return (empty_fig, "No data available", empty_fig,
                       "No data available", "No data for selected filters")
                        
            # Create visualizations
            plot_fig = self._create_enhanced_timeseries_plot(filtered_df, display_options, view_mode)
            residuals_fig = self._create_enhanced_residuals_plot(filtered_df)
            
            # Calculate metrics
            metrics_data, metrics_summary = self._calculate_enhanced_metrics(
                filtered_df, selected_metrics, view_mode
            )
            
            # Create data table
            data_table = self._create_enhanced_data_table(filtered_df)
            
            # Create summary
            summary = self._create_enhanced_summary(filtered_df, view_mode)
            
            return (plot_fig, metrics_summary, residuals_fig, data_table, summary)
        
        @self.app.callback(
            Output("download-data", "data"),
            Input("export-button", "n_clicks"),
            [State('date-picker-range', 'start_date'),
             State('date-picker-range', 'end_date'),
             State('model-selector', 'value'),
             State('split-selector', 'value')],
            prevent_initial_call=True
        )
        def export_data(n_clicks, start_date_str, end_date_str, selected_models, selected_splits):
            filtered_df = self._filter_data(start_date_str, end_date_str, selected_models, selected_splits)
            
            # Reset index to make it exportable
            export_df = filtered_df.reset_index()
            
            return dcc.send_data_frame(
                export_df.to_csv,
                f"resample_data_{start_date_str}_{end_date_str}.csv",
                index=False
            )
        
        # Enhanced Performance Metrics Callbacks
        @self.app.callback(
            [Output('performance-baseline-dropdown', 'options'),
             Output('performance-baseline-dropdown', 'disabled'),
             Output('comparison-mode-info', 'children')],
            [Input('comparison-mode-switch', 'value'),
             Input('model-selector', 'value')]
        )
        def manage_performance_baseline_dropdown(comparison_mode, selected_models):
            if not comparison_mode or not selected_models or len(selected_models) < 2:
                return [], True, dbc.Alert("Select at least 2 models to enable comparison mode.", color="warning")
            
            options = [{'label': model, 'value': model} for model in selected_models]
            info = dbc.Alert(f"Comparing {len(selected_models)} models. Select baseline for relative metrics.", color="info")
            return options, False, info
        
        @self.app.callback(
            Output('performance-metrics-table-container', 'children'),
            [Input('update-button', 'n_clicks')],
            [State('model-selector', 'value'),
             State('split-selector', 'value'),
             State('date-picker-range', 'start_date'),
             State('date-picker-range', 'end_date'),
             State('metrics-granularity-dropdown', 'value'),
             State('temporal-granularity-dropdown', 'value'),
             State('metrics-split-type-filter-dropdown', 'value'),
             State('metrics-column-toggle-dropdown', 'value'),
             State('comparison-mode-switch', 'value'),
             State('performance-baseline-dropdown', 'value')]
        )
        def generate_performance_metrics_table(n_clicks, selected_models, selected_splits, 
                                             start_date_str, end_date_str, granularity, temporal_granularity,
                                             split_filter, selected_metric_column, 
                                             comparison_mode, baseline_model):
            if n_clicks == 0 or not selected_models or not selected_splits:
                return html.Div("Click 'Update View' to generate performance metrics table.")
            
            # Filter data
            filtered_df = self._filter_data(start_date_str, end_date_str, selected_models, selected_splits)
            
            if filtered_df.empty:
                return html.Div("No data available for selected filters.")
            
            return self._create_enhanced_performance_table(
                filtered_df, granularity, split_filter, selected_metric_column, 
                comparison_mode, baseline_model, temporal_granularity
            )
        
        # Enhanced Model Comparison Callbacks
        @self.app.callback(
            [Output('comparison-baseline-dropdown', 'options'),
             Output('metric-weights-section', 'style')],
            [Input('model-selector', 'value'),
             Input('comparison-type-dropdown', 'value')]
        )
        def update_comparison_controls(selected_models, comparison_type):
            baseline_options = [{'label': model, 'value': model} for model in (selected_models or [])]
            
            # Show metric weights only for ranking comparison
            weights_style = {'display': 'block'} if comparison_type == 'ranking' else {'display': 'none'}
            
            return baseline_options, weights_style
        
        @self.app.callback(
            Output('comparison-content', 'children'),
            [Input('update-button', 'n_clicks')],
            [State('model-selector', 'value'),
             State('split-selector', 'value'),
             State('date-picker-range', 'start_date'),
             State('date-picker-range', 'end_date'),
             State('comparison-type-dropdown', 'value'),
             State('comparison-metric-dropdown', 'value'),
             State('comparison-baseline-dropdown', 'value'),
             State('mae-weight', 'value'),
             State('rmse-weight', 'value'),
             State('mape-weight', 'value'),
             State('smape-weight', 'value')]
        )
        def update_model_comparison(n_clicks, selected_models, selected_splits, 
                                  start_date_str, end_date_str, comparison_type, 
                                  comparison_metric, baseline_model,
                                  mae_weight, rmse_weight, mape_weight, smape_weight):
            if n_clicks == 0 or not selected_models or len(selected_models) < 2:
                return html.Div("Select at least 2 models and click 'Update' to see comparison.")
            
            # Filter data
            filtered_df = self._filter_data(start_date_str, end_date_str, selected_models, selected_splits)
            
            if filtered_df.empty:
                return html.Div("No data available for selected filters.")
            
            # Create metric weights dictionary
            metric_weights = {
                'mae': mae_weight or 0,
                'rmse': rmse_weight or 0,
                'mape': mape_weight or 0,
                'smape': smape_weight or 0
            }
            
            return self._create_advanced_model_comparison(
                filtered_df, comparison_type, comparison_metric, baseline_model, metric_weights
            )
        
    
    def _filter_data(self, start_date_str, end_date_str, selected_models, selected_splits):
        """Filter data based on selections."""
        # Convert dates
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        # Filter by date range
        df_filtered = self.resamples_df[
            (self.resamples_df.index.get_level_values('date') >= start_date) &
            (self.resamples_df.index.get_level_values('date') <= end_date)
        ]
        
        # Filter by model selection
        if selected_models:
            df_filtered = df_filtered[
                df_filtered.index.get_level_values('model_id').isin(selected_models)
            ]
        
        # Filter by split selection
        if selected_splits:
            df_filtered = df_filtered[
                df_filtered.index.get_level_values('slice_id').isin(selected_splits)
            ]
        
        return df_filtered
    
    def _create_enhanced_timeseries_plot(self, df, display_options, view_mode=None):
        """Create enhanced time series plot with multiple options."""
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title_text="No data to display")
            return fig
        
        # Check if we need separate plots for each slice/model combination
        if view_mode == 'all_separate':
            return self._create_separate_plots(df, display_options)
        
        fig = go.Figure()
        
        # Color palette for different models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Group by model and slice
        for i, ((slice_id, model_id), group) in enumerate(df.groupby(level=['slice_id', 'model_id'])):
            color = colors[i % len(colors)]
            dates = group.index.get_level_values('date')
            
            # Plot actuals - heavy dashed dark blue line
            fig.add_trace(go.Scatter(
                x=dates,
                y=group['actuals'],
                mode='lines',
                name=f'Actuals - {model_id} (Slice {slice_id})',
                line=dict(color='#1e3a8a', width=4, dash='dash'),  # Heavy dashed dark blue
                opacity=1.0
            ))
            
            # Plot fitted values (train period) - solid orange line
            if 'show_train' in display_options:
                train_data = group[group['period_type'] == 'train']
                if not train_data.empty:
                    fig.add_trace(go.Scatter(
                        x=train_data.index.get_level_values('date'),
                        y=train_data['fitted_values'],
                        mode='lines',
                        name=f'Train - {model_id} (Slice {slice_id})',
                        line=dict(color='#f97316', width=2, dash='solid'),  # Solid orange
                        opacity=1.0
                    ))
            
            # Plot predictions (test period) - solid red line
            if 'show_test' in display_options:
                test_data = group[group['period_type'] == 'test']
                if not test_data.empty:
                    fig.add_trace(go.Scatter(
                        x=test_data.index.get_level_values('date'),
                        y=test_data['predictions'],
                        mode='lines',
                        name=f'Test - {model_id} (Slice {slice_id})',
                        line=dict(color='#dc2626', width=2, dash='solid'),  # Solid red
                        opacity=1.0
                    ))
        
        fig.update_layout(
            title="Time Series Analysis: Actuals vs Fitted/Predicted Values",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600
        )
        
        return fig
    
    def _create_separate_plots(self, df, display_options):
        """Create separate subplots for each slice/model combination."""
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title_text="No data to display")
            return fig
        
        # Get unique combinations
        unique_combinations = df.index.droplevel('date').unique().tolist()
        n_plots = len(unique_combinations)
        
        if n_plots == 0:
            fig = go.Figure()
            fig.update_layout(title_text="No data to display")
            return fig
        
        # Calculate subplot layout (prefer more rows than columns for better readability)
        n_cols = min(2, n_plots)  # Maximum 2 columns
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
        
        # Calculate appropriate spacing based on number of rows
        # Maximum vertical spacing allowed is (1 / (rows - 1))
        if n_rows > 1:
            max_vertical_spacing = 1.0 / (n_rows - 1)
            vertical_spacing = min(0.08, max_vertical_spacing * 0.8)  # Use 80% of max allowed
        else:
            vertical_spacing = 0.08
        
        # For many plots, reduce spacing further
        if n_rows > 10:
            vertical_spacing = min(vertical_spacing, 0.02)
        elif n_rows > 5:
            vertical_spacing = min(vertical_spacing, 0.04)
        
        # Create subplot titles
        subplot_titles = [f"Slice {slice_id}, Model {model_id}" 
                         for slice_id, model_id in unique_combinations]
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.05
        )
        
        # Color palette for different models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        model_colors = {}
        
        # Assign consistent colors to models
        for i, model_id in enumerate(df.index.get_level_values('model_id').unique()):
            model_colors[model_id] = colors[i % len(colors)]
        
        # Plot each combination in its own subplot
        for idx, (slice_id, model_id) in enumerate(unique_combinations):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            # Get data for this combination
            try:
                group = df.xs((slice_id, model_id), level=('slice_id', 'model_id'))
                dates = group.index
                color = model_colors[model_id]
                
                # Plot actuals - heavy dashed dark blue line
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=group['actuals'],
                    mode='lines',
                    name=f'Actuals',
                    line=dict(color='#1e3a8a', width=4, dash='dash'),  # Heavy dashed dark blue
                    opacity=1.0,
                    showlegend=(idx == 0),  # Only show legend for first plot
                    legendgroup='actuals'
                ), row=row, col=col)
                
                # Plot fitted values (train period) - solid orange line
                if 'show_train' in display_options:
                    train_data = group[group['period_type'] == 'train']
                    if not train_data.empty:
                        fig.add_trace(go.Scatter(
                            x=train_data.index,
                            y=train_data['fitted_values'],
                            mode='lines',
                            name=f'Train',
                            line=dict(color='#f97316', width=2, dash='solid'),  # Solid orange
                            opacity=1.0,
                            showlegend=(idx == 0),
                            legendgroup='train'
                        ), row=row, col=col)
                
                # Plot predictions (test period) - solid red line
                if 'show_test' in display_options:
                    test_data = group[group['period_type'] == 'test']
                    if not test_data.empty:
                        fig.add_trace(go.Scatter(
                            x=test_data.index,
                            y=test_data['predictions'],
                            mode='lines',
                            name=f'Test',
                            line=dict(color='#dc2626', width=2, dash='solid'),  # Solid red
                            opacity=1.0,
                            showlegend=(idx == 0),
                            legendgroup='test'
                        ), row=row, col=col)
                
                # Add vertical line to separate train/test periods if both are shown
                if 'show_train' in display_options and 'show_test' in display_options:
                    train_data = group[group['period_type'] == 'train']
                    test_data = group[group['period_type'] == 'test']
                    
                    if not train_data.empty and not test_data.empty:
                        split_date = test_data.index.min()
                        fig.add_vline(
                            x=split_date,
                            line_dash="dash",
                            line_color="gray",
                            opacity=0.5,
                            row=row, col=col
                        )
                
            except KeyError:
                # Handle case where combination doesn't exist
                continue
        
        # Update layout - extend plots down the page with consistent dimensions
        fig.update_layout(
            title="Time Series Analysis: Individual Plots by Slice and Model",
            height=max(600, 400 * n_rows),  # Dynamic height based on number of rows
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes labels
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(title_text="Date" if i == n_rows else "", row=i, col=j)
                fig.update_yaxes(title_text="Value" if j == 1 else "", row=i, col=j)
        
        return fig
    
    def _create_enhanced_performance_table(self, df, granularity, split_filter, selected_metric_column, 
                                         comparison_mode, baseline_model, temporal_granularity=None):
        """Create enhanced performance metrics table with temporal groups as columns and hierarchical rows."""
        if df.empty:
            return html.Div("No data available.")
        
        # Filter by split type if specified
        if split_filter != 'all':
            df = df[df['period_type'] == split_filter]
        
        # Calculate metrics based on granularity
        temporal_param = temporal_granularity if temporal_granularity != 'none' else None
        metrics_df = self._calculate_performance_metrics(df, granularity, temporal_param)
        
        if metrics_df.empty:
            return html.Div("No metrics calculated for current selection.")
        
        # Check if selected metric is available
        if selected_metric_column not in metrics_df.columns:
            return html.Div(f"Selected metric '{selected_metric_column}' not available in data.")
        
        # Create pivot table with temporal groups as columns
        if temporal_granularity and temporal_granularity != 'none' and 'temporal_group' in metrics_df.columns:
            # Create hierarchical table with temporal groups as columns
            pivot_df = self._create_temporal_pivot_table(metrics_df, selected_metric_column, granularity)
        else:
            # Create simple table without temporal grouping
            pivot_df = self._create_simple_performance_table(metrics_df, selected_metric_column, granularity)
        
        if pivot_df.empty:
            return html.Div("No data available for table generation.")
        
        # Create styled data table
        table_columns = [{"name": col, "id": col} for col in pivot_df.columns]
        
        # Add conditional styling
        style_data_conditional = [
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
        
        # Add hierarchical row styling
        style_data_conditional.extend(self._get_hierarchical_row_styles(pivot_df))
        
        # Add Total column styling
        if 'Total' in pivot_df.columns:
            style_data_conditional.append({
                'if': {'column_id': 'Total'},
                'backgroundColor': 'rgba(255, 193, 7, 0.2)',
                'fontWeight': 'bold',
                'border': '2px solid rgba(255, 193, 7, 0.5)'
            })
        
        data_table = dash_table.DataTable(
            data=pivot_df.to_dict('records'),
            columns=table_columns,
            style_table={
                'overflowX': 'auto',
                'height': '600px',
                'overflowY': 'auto'
            },
            style_cell={
                'textAlign': 'center', 
                'padding': '8px',
                'fontSize': '11px',
                'fontFamily': 'Arial, sans-serif',
                'whiteSpace': 'normal',
                'height': 'auto',
                'maxWidth': '120px',
                'minWidth': '80px'
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Hierarchy'},
                    'textAlign': 'left',
                    'minWidth': '200px',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'column_id': 'Total'},
                    'minWidth': '100px'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'fontSize': '12px',
                'padding': '10px'
            },
            style_data_conditional=style_data_conditional,
            sort_action="native",
            filter_action="native",
            export_format="csv",
            fixed_rows={'headers': True},
            merge_duplicate_headers=True
        )
        
        # Add summary statistics
        summary_info = html.Div([
            html.H6(f"Performance Metrics: {selected_metric_column.upper()}", className="mb-2"),
            html.P(f"Granularity: {granularity.replace('_', ' ').title()}", className="text-muted mb-2"),
            html.P(f"Temporal Grouping: {temporal_granularity if temporal_granularity != 'none' else 'None'}", className="text-muted mb-3")
        ])
        
        return html.Div([
            summary_info,
            data_table
        ])
    
    def _create_temporal_pivot_table(self, metrics_df, selected_metric, granularity):
        """Create pivot table with temporal groups as columns and hierarchical rows."""
        # Create hierarchical index based on granularity
        if granularity == 'by_model':
            index_cols = ['model_id', 'period_type']
        elif granularity == 'by_split':
            index_cols = ['slice_id', 'period_type']
        elif granularity == 'by_model_split':
            index_cols = ['model_id', 'slice_id', 'period_type']
        else:  # overall
            index_cols = ['period_type']
        
        # Create pivot table
        pivot_df = metrics_df.pivot_table(
            values=selected_metric,
            index=index_cols,
            columns='temporal_group',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        # Check if we need to reorder month columns
        if 'temporal_group' in metrics_df.columns:
            temporal_values = metrics_df['temporal_group'].unique()
            # Check if these are month names
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            if all(val in month_names for val in temporal_values):
                # Reorder columns by month order
                ordered_months = [month for month in month_names if month in pivot_df.columns]
                other_cols = [col for col in pivot_df.columns if col not in month_names]
                pivot_df = pivot_df[other_cols + ordered_months]
        
        # Add Total column (average across all temporal columns)
        numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            pivot_df['Total'] = pivot_df[numeric_cols].mean(axis=1)
        
        # Reset index to create hierarchical row structure
        pivot_df = pivot_df.reset_index()
        
        # Create hierarchical row labels
        if len(index_cols) > 1:
            # Create multi-level row labels
            row_labels = []
            for _, row in pivot_df.iterrows():
                label_parts = []
                for col in index_cols:
                    if col == 'model_id':
                        label_parts.append(f"Model: {row[col]}")
                    elif col == 'slice_id':
                        label_parts.append(f"Slice: {row[col]}")
                    elif col == 'period_type':
                        label_parts.append(f"{row[col].title()}")
                row_labels.append(" | ".join(label_parts))
            
            # Replace the index columns with the hierarchical label
            pivot_df.insert(0, 'Hierarchy', row_labels)
            pivot_df = pivot_df.drop(columns=index_cols)
        
        # Round numeric values
        numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
        pivot_df[numeric_cols] = pivot_df[numeric_cols].round(4)
        
        return pivot_df
    
    def _create_simple_performance_table(self, metrics_df, selected_metric, granularity):
        """Create simple performance table without temporal grouping."""
        # Select relevant columns based on granularity
        if granularity == 'by_model':
            display_cols = ['model_id', 'period_type', selected_metric]
        elif granularity == 'by_split':
            display_cols = ['slice_id', 'period_type', selected_metric]
        elif granularity == 'by_model_split':
            display_cols = ['model_id', 'slice_id', 'period_type', selected_metric]
        else:  # overall
            display_cols = ['period_type', selected_metric]
        
        # Filter available columns
        available_cols = [col for col in display_cols if col in metrics_df.columns]
        table_df = metrics_df[available_cols].copy()
        
        # Round numeric values
        if selected_metric in table_df.columns:
            table_df[selected_metric] = table_df[selected_metric].round(4)
        
        return table_df
    
    def _get_hierarchical_row_styles(self, df):
        """Create styling for hierarchical rows."""
        styles = []
        
        # Check if we have a hierarchy column
        if 'Hierarchy' in df.columns:
            for i, row in df.iterrows():
                hierarchy_text = str(row['Hierarchy'])
                
                # Style based on hierarchy level
                if 'Model:' in hierarchy_text and 'Train' in hierarchy_text:
                    styles.append({
                        'if': {'row_index': i},
                        'backgroundColor': 'rgba(0, 123, 255, 0.1)',
                        'fontWeight': 'bold'
                    })
                elif 'Model:' in hierarchy_text and 'Test' in hierarchy_text:
                    styles.append({
                        'if': {'row_index': i},
                        'backgroundColor': 'rgba(255, 193, 7, 0.1)',
                        'fontWeight': 'bold'
                    })
        
        return styles
    
    def _calculate_performance_metrics(self, df, granularity, temporal_granularity=None):
        """Calculate performance metrics based on granularity and optional temporal grouping."""
        from sklearn.metrics import r2_score
        
        metrics_list = []
        
        # Apply temporal aggregation if specified
        if temporal_granularity and temporal_granularity in ['year', 'quarter', 'month', 'month_groups', 'week', 'day']:
            df = TemporalAggregator.aggregate_by_temporal_granularity(df, temporal_granularity)
        
        if granularity == 'overall':
            # Overall metrics across all models and splits
            for period in df['period_type'].unique():
                period_data = df[df['period_type'] == period]
                if not period_data.empty:
                    metrics = self._compute_metrics_for_group(period_data)
                    metrics['period_type'] = period
                    metrics_list.append(metrics)
        
        elif granularity == 'by_model':
            # Metrics by model (and temporal group if specified)
            for model_id in df.index.get_level_values('model_id').unique():
                model_data = df[df.index.get_level_values('model_id') == model_id]
                
                # Group by temporal_group if it exists
                if 'temporal_group' in model_data.columns:
                    for temporal_group in model_data['temporal_group'].unique():
                        temporal_data = model_data[model_data['temporal_group'] == temporal_group]
                        for period in temporal_data['period_type'].unique():
                            period_data = temporal_data[temporal_data['period_type'] == period]
                            if not period_data.empty:
                                metrics = self._compute_metrics_for_group(period_data)
                                metrics['model_id'] = model_id
                                metrics['period_type'] = period
                                metrics['temporal_group'] = str(temporal_group)
                                metrics_list.append(metrics)
                else:
                    for period in model_data['period_type'].unique():
                        period_data = model_data[model_data['period_type'] == period]
                        if not period_data.empty:
                            metrics = self._compute_metrics_for_group(period_data)
                            metrics['model_id'] = model_id
                            metrics['period_type'] = period
                            metrics_list.append(metrics)
        
        elif granularity == 'by_split':
            # Metrics by split
            for slice_id in df.index.get_level_values('slice_id').unique():
                split_data = df[df.index.get_level_values('slice_id') == slice_id]
                for period in split_data['period_type'].unique():
                    period_data = split_data[split_data['period_type'] == period]
                    if not period_data.empty:
                        metrics = self._compute_metrics_for_group(period_data)
                        metrics['slice_id'] = slice_id
                        metrics['period_type'] = period
                        metrics_list.append(metrics)
        
        elif granularity == 'by_model_split':
            # Metrics by model and split
            for (slice_id, model_id) in df.index.droplevel('date').unique():
                group_data = df.xs((slice_id, model_id), level=('slice_id', 'model_id'))
                for period in group_data['period_type'].unique():
                    period_data = group_data[group_data['period_type'] == period]
                    if not period_data.empty:
                        metrics = self._compute_metrics_for_group(period_data)
                        metrics['model_id'] = model_id
                        metrics['slice_id'] = slice_id
                        metrics['period_type'] = period
                        metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
    
    def _compute_metrics_for_group(self, data):
        """Compute metrics for a group of data."""
        from sklearn.metrics import r2_score
        
        # Determine which values to use based on period type
        if 'train' in data['period_type'].values:
            y_true = data['actuals']
            y_pred = data['fitted_values']
        else:
            y_true = data['actuals']
            y_pred = data['predictions']
        
        # Remove any NaN values
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {}
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = calculate_mape(y_true, y_pred)
        smape = calculate_smape(y_true, y_pred)
        
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan
        
        residuals = y_true - y_pred
        mean_error = np.mean(residuals)
        std_error = np.std(residuals)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'r2': r2,
            'mean_error': mean_error,
            'std_error': std_error,
            'n_observations': len(y_true)
        }
    
    def _apply_comparison_mode(self, table_df, baseline_model, metric_columns):
        """Apply comparison mode to show relative performance."""
        if 'model_id' not in table_df.columns:
            return table_df
        
        baseline_data = table_df[table_df['model_id'] == baseline_model]
        if baseline_data.empty:
            return table_df
        
        result_df = table_df.copy()
        
        for col in metric_columns:
            if col in table_df.columns:
                baseline_value = baseline_data[col].iloc[0] if len(baseline_data) > 0 else 0
                if baseline_value != 0:
                    # Calculate percentage difference
                    result_df[col] = ((table_df[col] - baseline_value) / baseline_value * 100).round(2)
                else:
                    result_df[col] = 0
        
        return result_df
    
    def _create_performance_summary(self, table_df, metric_columns, comparison_mode):
        """Create dynamic performance summary with best/worst performers."""
        if table_df.empty:
            return html.Div("No data for summary.")
        
        summary_content = []
        
        # Create aggregated summary for selected metrics
        if metric_columns and any(col in table_df.columns for col in metric_columns):
            # Summary statistics panel
            summary_stats = []
            best_performers = []
            
            for col in metric_columns:
                if col in table_df.columns:
                    values = table_df[col].dropna()
                    if len(values) > 0:
                        # Calculate summary stats
                        avg_val = values.mean()
                        std_val = values.std()
                        min_val = values.min()
                        max_val = values.max()
                        
                        if comparison_mode:
                            stat_text = f"Avg: {avg_val:.2f}% (±{std_val:.2f}%)"
                        else:
                            stat_text = f"Avg: {avg_val:.4f} (±{std_val:.4f})"
                        
                        summary_stats.append(
                            html.Div([
                                html.Strong(f"{col.upper()}: "),
                                html.Span(stat_text)
                            ], className="mb-1")
                        )
                        
                        # Find best performer for this metric
                        if 'model_id' in table_df.columns:
                            if col.lower() == 'r2':
                                best_idx = table_df[col].idxmax()
                            else:
                                best_idx = table_df[col].idxmin()
                            
                            best_model = table_df.loc[best_idx, 'model_id']
                            best_value = table_df.loc[best_idx, col]
                            
                            best_performers.append(
                                html.Div([
                                    html.Strong(f"{col.upper()}: "),
                                    html.Span(f"{best_model} ({best_value:.4f})", 
                                             style={'color': '#28a745'})
                                ], className="mb-1")
                            )
            
            # Create summary cards layout
            summary_card = dbc.Card([
                dbc.CardBody([
                    html.H6("📊 Aggregate Statistics", className="card-title mb-3"),
                    html.Div(summary_stats)
                ])
            ], color="light", outline=True, className="mb-2")
            
            summary_content.append(dbc.Col(summary_card, md=6))
            
            if best_performers and not comparison_mode:
                best_card = dbc.Card([
                    dbc.CardBody([
                        html.H6("🏆 Best Performers", className="card-title mb-3"),
                        html.Div(best_performers)
                    ])
                ], color="success", outline=True, className="mb-2")
                
                summary_content.append(dbc.Col(best_card, md=6))
        
        if summary_content:
            return dbc.Row(summary_content, className="mb-3")
        else:
            return html.Div("No summary data available.")

    def _create_enhanced_residuals_plot(self, df):
        """Create enhanced residuals analysis plot."""
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title_text="No data for residuals analysis")
            return fig
        
        # Create subplots for different residual analyses
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals Over Time', 'Residuals Distribution', 
                          'Q-Q Plot', 'Residuals vs Fitted'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, ((slice_id, model_id), group) in enumerate(df.groupby(level=['slice_id', 'model_id'])):
            color = colors[i % len(colors)]
            dates = group.index.get_level_values('date')
            residuals = group['residuals'].dropna()
            
            if len(residuals) == 0:
                continue
            
            # Residuals over time
            fig.add_trace(
                go.Scatter(x=dates, y=group['residuals'], mode='markers',
                          name=f'{model_id} (Slice {slice_id})', marker_color=color),
                row=1, col=1
            )
            
            # Residuals distribution
            fig.add_trace(
                go.Histogram(x=residuals, name=f'{model_id} (Slice {slice_id})',
                           marker_color=color, opacity=0.7, nbinsx=20),
                row=1, col=2
            )
            
            # Q-Q plot (simplified)
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = np.linspace(-3, 3, len(sorted_residuals))
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                          mode='markers', name=f'{model_id} (Slice {slice_id})',
                          marker_color=color),
                row=2, col=1
            )
            
            # Residuals vs Fitted
            fitted_values = group['fitted_values'].fillna(group['predictions'])
            fig.add_trace(
                go.Scatter(x=fitted_values, y=group['residuals'],
                          mode='markers', name=f'{model_id} (Slice {slice_id})',
                          marker_color=color),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, title_text="Residual Analysis Dashboard")
        return fig
    
    def _calculate_enhanced_metrics(self, df, selected_metrics, view_mode):
        """Calculate enhanced performance metrics."""
        metrics_data = []
        
        # Define metric functions
        metric_functions = {
            'mae': mean_absolute_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': calculate_mape,
            'smape': calculate_smape
        }
        
        # Calculate metrics for each group
        for (slice_id, model_id), group in df.groupby(level=['slice_id', 'model_id']):
            # Train metrics
            train_data = group[group['period_type'] == 'train']
            if not train_data.empty and 'fitted_values' in train_data.columns:
                actuals_train = train_data['actuals'].dropna()
                fitted_vals = train_data['fitted_values'].dropna()
                common_idx = actuals_train.index.intersection(fitted_vals.index)
                
                if len(common_idx) > 0:
                    for metric_name in selected_metrics:
                        try:
                            value = metric_functions[metric_name](
                                actuals_train.loc[common_idx], 
                                fitted_vals.loc[common_idx]
                            )
                            metrics_data.append({
                                'Slice ID': slice_id,
                                'Model ID': model_id,
                                'Period': 'Train',
                                'Metric': metric_name.upper(),
                                'Value': f"{value:.4f}",
                                'Count': len(common_idx)
                            })
                        except Exception as e:
                            metrics_data.append({
                                'Slice ID': slice_id,
                                'Model ID': model_id,
                                'Period': 'Train',
                                'Metric': metric_name.upper(),
                                'Value': "Error",
                                'Count': 0
                            })
            
            # Test metrics
            test_data = group[group['period_type'] == 'test']
            if not test_data.empty and 'predictions' in test_data.columns:
                actuals_test = test_data['actuals'].dropna()
                predictions = test_data['predictions'].dropna()
                common_idx = actuals_test.index.intersection(predictions.index)
                
                if len(common_idx) > 0:
                    for metric_name in selected_metrics:
                        try:
                            value = metric_functions[metric_name](
                                actuals_test.loc[common_idx], 
                                predictions.loc[common_idx]
                            )
                            metrics_data.append({
                                'Slice ID': slice_id,
                                'Model ID': model_id,
                                'Period': 'Test',
                                'Metric': metric_name.upper(),
                                'Value': f"{value:.4f}",
                                'Count': len(common_idx)
                            })
                        except Exception as e:
                            metrics_data.append({
                                'Slice ID': slice_id,
                                'Model ID': model_id,
                                'Period': 'Test',
                                'Metric': metric_name.upper(),
                                'Value': "Error",
                                'Count': 0
                            })
        
        # Create metrics summary
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df['Value_Numeric'] = pd.to_numeric(metrics_df['Value'], errors='coerce')
            
            summary_stats = metrics_df.groupby(['Metric', 'Period'])['Value_Numeric'].agg(['mean', 'std', 'count'])
            
            summary_cards = []
            for (metric, period), stats in summary_stats.iterrows():
                if not pd.isna(stats['mean']):
                    summary_cards.append(
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(f"{metric} ({period})", className="card-title"),
                                    html.H2(f"{stats['mean']:.4f}", className="text-primary"),
                                    html.P(f"±{stats['std']:.4f} (n={int(stats['count'])})", 
                                          className="text-muted")
                                ])
                            ])
                        ], md=3)
                    )
            
            metrics_summary = dbc.Row(summary_cards) if summary_cards else "No valid metrics calculated"
        else:
            metrics_summary = "No metrics data available"
        
        return metrics_data, metrics_summary
    
    def _create_enhanced_data_table(self, df):
        """Create enhanced data table with filtering and sorting."""
        if df.empty:
            return "No data available"
        
        # Reset index for table display
        table_df = df.reset_index()
        
        # Round numeric columns
        numeric_cols = table_df.select_dtypes(include=[np.number]).columns
        table_df[numeric_cols] = table_df[numeric_cols].round(4)
        
        return dash_table.DataTable(
            data=table_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in table_df.columns],
            style_table={'overflowX': 'auto', 'height': '500px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '12px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{period_type} = train'},
                    'backgroundColor': 'rgba(0, 123, 255, 0.1)'
                },
                {
                    'if': {'filter_query': '{period_type} = test'},
                    'backgroundColor': 'rgba(255, 193, 7, 0.1)'
                }
            ],
            sort_action="native",
            filter_action="native",
            page_action="native",
            page_size=20
        )
    
    def _create_model_comparison(self, df, selected_metrics):
        """Create model comparison dashboard."""
        if df.empty or len(self.unique_models) < 2:
            return "Need multiple models for comparison"
        
        # Calculate average metrics by model
        comparison_data = []
        metric_functions = {
            'mae': mean_absolute_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': calculate_mape,
            'smape': calculate_smape
        }
        
        for model_id in self.unique_models:
            model_data = df[df.index.get_level_values('model_id') == model_id]
            if model_data.empty:
                continue
            
            # Test period metrics
            test_data = model_data[model_data['period_type'] == 'test']
            if not test_data.empty:
                actuals = test_data['actuals'].dropna()
                predictions = test_data['predictions'].dropna()
                common_idx = actuals.index.intersection(predictions.index)
                
                if len(common_idx) > 0:
                    model_metrics = {'Model': model_id}
                    for metric_name in selected_metrics:
                        try:
                            value = metric_functions[metric_name](
                                actuals.loc[common_idx], 
                                predictions.loc[common_idx]
                            )
                            model_metrics[metric_name.upper()] = value
                        except:
                            model_metrics[metric_name.upper()] = np.nan
                    
                    comparison_data.append(model_metrics)
        
        if not comparison_data:
            return "No comparison data available"
        
        # Create comparison visualization
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create radar chart for model comparison
        fig = go.Figure()
        
        for _, row in comparison_df.iterrows():
            metrics_values = [row[col] for col in comparison_df.columns if col != 'Model']
            metrics_names = [col for col in comparison_df.columns if col != 'Model']
            
            # Normalize values for radar chart (invert for error metrics)
            normalized_values = []
            for val in metrics_values:
                if pd.isna(val):
                    normalized_values.append(0)
                else:
                    # For error metrics, lower is better, so invert
                    normalized_values.append(1 / (1 + val))
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=metrics_names,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison (Higher = Better)"
        )
        
        # Create comparison table
        comparison_table = dash_table.DataTable(
            data=comparison_df.round(4).to_dict('records'),
            columns=[{"name": i, "id": i} for i in comparison_df.columns],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
        
        return html.Div([
            dcc.Graph(figure=fig, style={'height': '400px'}),
            html.Hr(),
            html.H5("Detailed Comparison Table"),
            comparison_table
        ])
    
    def _create_advanced_model_comparison(self, df, comparison_type, comparison_metric, baseline_model, metric_weights):
        """Create advanced model comparison with multiple analysis types."""
        if df.empty:
            return html.Div("No data available for comparison.")
        
        if comparison_type == 'summary':
            return self._create_performance_summary_comparison(df, comparison_metric)
        elif comparison_type == 'pairwise':
            return self._create_pairwise_comparison_matrix(df, comparison_metric)
        elif comparison_type == 'dominance':
            return self._create_dominance_analysis(df, comparison_metric)
        elif comparison_type == 'ranking':
            return self._create_multi_metric_ranking(df, metric_weights, baseline_model)
        else:
            return html.Div("Unknown comparison type selected.")
    
    def _create_performance_summary_comparison(self, df, metric):
        """Create enhanced performance summary with statistical indicators."""
        # Calculate metrics by model
        comparison_data = []
        test_data = df[df['period_type'] == 'test']
        
        for model_id in test_data.index.get_level_values('model_id').unique():
            model_data = test_data[test_data.index.get_level_values('model_id') == model_id]
            if not model_data.empty:
                actuals = model_data['actuals'].dropna()
                predictions = model_data['predictions'].dropna()
                common_idx = actuals.index.intersection(predictions.index)
                
                if len(common_idx) > 0:
                    if metric == 'mae':
                        value = mean_absolute_error(actuals.loc[common_idx], predictions.loc[common_idx])
                    elif metric == 'rmse':
                        value = np.sqrt(mean_squared_error(actuals.loc[common_idx], predictions.loc[common_idx]))
                    elif metric == 'mape':
                        value = calculate_mape(actuals.loc[common_idx], predictions.loc[common_idx])
                    elif metric == 'smape':
                        value = calculate_smape(actuals.loc[common_idx], predictions.loc[common_idx])
                    else:
                        value = np.nan
                    
                    # Calculate confidence interval
                    residuals = actuals.loc[common_idx] - predictions.loc[common_idx]
                    std_error = np.std(residuals) / np.sqrt(len(residuals))
                    ci_95 = 1.96 * std_error
                    
                    comparison_data.append({
                        'Model': model_id,
                        f'{metric.upper()}': round(value, 4),
                        'Std Error': round(std_error, 4),
                        '95% CI Lower': round(value - ci_95, 4),
                        '95% CI Upper': round(value + ci_95, 4),
                        'Sample Size': len(common_idx)
                    })
        
        if not comparison_data:
            return html.Div("No comparison data available.")
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models and add performance indicators
        comparison_df = comparison_df.sort_values(f'{metric.upper()}')
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        comparison_df['Relative Performance (%)'] = ((comparison_df[f'{metric.upper()}'] / comparison_df[f'{metric.upper()}'].iloc[0]) - 1) * 100
        
        # Create enhanced table with color coding
        style_data_conditional = []
        for i, row in comparison_df.iterrows():
            rank = row['Rank']
            if rank == 1:
                color = 'rgba(40, 167, 69, 0.3)'  # Green for best
            elif rank == len(comparison_df):
                color = 'rgba(220, 53, 69, 0.3)'  # Red for worst
            else:
                color = 'rgba(255, 193, 7, 0.2)'  # Yellow for middle
            
            style_data_conditional.append({
                'if': {'row_index': i},
                'backgroundColor': color
            })
        
        table = dash_table.DataTable(
            data=comparison_df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in comparison_df.columns],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=style_data_conditional,
            sort_action="native"
        )
        
        return html.Div([
            html.H5(f"Performance Summary - {metric.upper()}"),
            html.P("🥇 Green: Best performer | 🥈 Yellow: Middle performers | 🥉 Red: Worst performer"),
            table
        ])
    
    def _create_pairwise_comparison_matrix(self, df, metric):
        """Create pairwise comparison matrix with significance testing."""
        test_data = df[df['period_type'] == 'test']
        models = test_data.index.get_level_values('model_id').unique().tolist()
        
        # Calculate metric values for each model
        model_values = {}
        for model_id in models:
            model_data = test_data[test_data.index.get_level_values('model_id') == model_id]
            if not model_data.empty:
                actuals = model_data['actuals'].dropna()
                predictions = model_data['predictions'].dropna()
                common_idx = actuals.index.intersection(predictions.index)
                
                if len(common_idx) > 0:
                    if metric == 'mae':
                        values = np.abs(actuals.loc[common_idx] - predictions.loc[common_idx])
                    elif metric == 'rmse':
                        values = (actuals.loc[common_idx] - predictions.loc[common_idx]) ** 2
                    elif metric == 'mape':
                        values = np.abs((actuals.loc[common_idx] - predictions.loc[common_idx]) / actuals.loc[common_idx]) * 100
                    elif metric == 'smape':
                        denom = (np.abs(actuals.loc[common_idx]) + np.abs(predictions.loc[common_idx])) / 2
                        values = np.abs(actuals.loc[common_idx] - predictions.loc[common_idx]) / denom * 100
                    else:
                        values = np.array([])
                    
                    model_values[model_id] = values
        
        # Create pairwise comparison matrix
        matrix_data = []
        for model1 in models:
            row_data = {'Model': model1}
            for model2 in models:
                if model1 == model2:
                    row_data[model2] = '—'
                elif model1 in model_values and model2 in model_values:
                    _, p_value = ModelComparisonAnalyzer.wilcoxon_test(
                        model_values[model1], model_values[model2]
                    )
                    effect = ModelComparisonAnalyzer.effect_size(
                        model_values[model1], model_values[model2]
                    )
                    stars = ModelComparisonAnalyzer.significance_stars(p_value)
                    
                    if not pd.isna(effect):
                        row_data[model2] = f"{effect:.2f}{stars}"
                    else:
                        row_data[model2] = 'N/A'
                else:
                    row_data[model2] = 'N/A'
            
            matrix_data.append(row_data)
        
        matrix_df = pd.DataFrame(matrix_data)
        
        table = dash_table.DataTable(
            data=matrix_df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in matrix_df.columns],
            style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '11px'},
            style_header={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'fontWeight': 'bold'},
            style_table={'overflowX': 'auto'}
        )
        
        return html.Div([
            html.H5(f"Pairwise Comparison Matrix - {metric.upper()}"),
            html.P("Effect sizes with significance: *** p<0.001, ** p<0.01, * p<0.05"),
            html.P("Positive values: Row model performs worse than column model"),
            table
        ])
    
    def _create_dominance_analysis(self, df, metric):
        """Create dominance analysis showing win/loss ratios."""
        test_data = df[df['period_type'] == 'test']
        models = test_data.index.get_level_values('model_id').unique().tolist()
        
        # Calculate dominance matrix
        model_scorer = ModelScorer()
        dominance_matrix = model_scorer.calculate_dominance_matrix(test_data, metric)
        
        # Create visualization
        fig = go.Figure(data=go.Heatmap(
            z=dominance_matrix.values,
            x=dominance_matrix.columns,
            y=dominance_matrix.index,
            colorscale='RdYlGn',
            zmid=50,
            text=dominance_matrix.round(1),
            texttemplate="%{text}%",
            textfont={"size": 10},
            colorbar=dict(title="Win Rate %")
        ))
        
        fig.update_layout(
            title=f"Dominance Matrix - {metric.upper()} (% of times row beats column)",
            xaxis_title="Model (Column)",
            yaxis_title="Model (Row)",
            height=500
        )
        
        # Calculate overall dominance scores
        dominance_scores = []
        for model in models:
            if model in dominance_matrix.index:
                avg_dominance = dominance_matrix.loc[model].drop(model).mean()
                dominance_scores.append({
                    'Model': model,
                    'Average Dominance (%)': round(avg_dominance, 1),
                    'Wins': int((dominance_matrix.loc[model] > 50).sum() - 1),  # Exclude self
                    'Losses': int((dominance_matrix.loc[model] < 50).sum())
                })
        
        dominance_df = pd.DataFrame(dominance_scores).sort_values('Average Dominance (%)', ascending=False)
        dominance_df['Rank'] = range(1, len(dominance_df) + 1)
        
        table = dash_table.DataTable(
            data=dominance_df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in dominance_df.columns],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'fontWeight': 'bold'}
        )
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.H5("Dominance Summary"),
            table
        ])
    
    def _create_multi_metric_ranking(self, df, metric_weights, baseline_model):
        """Create multi-metric ranking with composite scores."""
        test_data = df[df['period_type'] == 'test']
        
        # Calculate all metrics for each model
        ranking_data = []
        models = test_data.index.get_level_values('model_id').unique().tolist()
        
        for model_id in models:
            model_data = test_data[test_data.index.get_level_values('model_id') == model_id]
            if not model_data.empty:
                actuals = model_data['actuals'].dropna()
                predictions = model_data['predictions'].dropna()
                common_idx = actuals.index.intersection(predictions.index)
                
                if len(common_idx) > 0:
                    mae = mean_absolute_error(actuals.loc[common_idx], predictions.loc[common_idx])
                    rmse = np.sqrt(mean_squared_error(actuals.loc[common_idx], predictions.loc[common_idx]))
                    mape = calculate_mape(actuals.loc[common_idx], predictions.loc[common_idx])
                    smape = calculate_smape(actuals.loc[common_idx], predictions.loc[common_idx])
                    
                    ranking_data.append({
                        'model_id': model_id,
                        'mae': mae,
                        'rmse': rmse,
                        'mape': mape,
                        'smape': smape
                    })
        
        if not ranking_data:
            return html.Div("No data available for ranking.")
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Calculate composite scores
        scorer = ModelScorer(metric_weights)
        composite_scores = scorer.calculate_composite_score(ranking_df)
        
        # Add composite scores to dataframe
        ranking_df['Composite Score'] = ranking_df['model_id'].map(composite_scores)
        ranking_df = ranking_df.sort_values('Composite Score', ascending=False)
        ranking_df['Overall Rank'] = range(1, len(ranking_df) + 1)
        
        # Calculate skill scores if baseline provided
        if baseline_model and baseline_model in ranking_df['model_id'].values:
            baseline_metrics = ranking_df[ranking_df['model_id'] == baseline_model].iloc[0]
            for metric in ['mae', 'rmse', 'mape', 'smape']:
                skill_col = f'{metric.upper()} Skill Score'
                ranking_df[skill_col] = ranking_df[metric].apply(
                    lambda x: ModelComparisonAnalyzer.skill_score([x], [baseline_metrics[metric]])
                )
        
        # Format for display
        display_df = ranking_df.copy()
        for col in ['mae', 'rmse', 'mape', 'smape', 'Composite Score']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        # Add color coding based on rank
        style_data_conditional = []
        for i, row in display_df.iterrows():
            rank = row['Overall Rank']
            if rank <= 3:
                color = f'rgba(40, 167, 69, {0.4 - (rank-1)*0.1})'  # Green gradient for top 3
            elif rank >= len(display_df) - 2:
                color = f'rgba(220, 53, 69, {0.4 - (len(display_df)-rank)*0.1})'  # Red gradient for bottom 3
            else:
                color = 'rgba(255, 255, 255, 0)'  # No color for middle
            
            style_data_conditional.append({
                'if': {'row_index': i},
                'backgroundColor': color
            })
        
        table = dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[{"name": col.replace('_', ' ').title(), "id": col} for col in display_df.columns],
            style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '11px'},
            style_header={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=style_data_conditional,
            sort_action="native"
        )
        
        # Create weights summary
        weights_summary = html.Div([
            html.H6("Current Metric Weights:"),
            html.P(f"MAE: {metric_weights.get('mae', 0):.1f}, RMSE: {metric_weights.get('rmse', 0):.1f}, MAPE: {metric_weights.get('mape', 0):.1f}, sMAPE: {metric_weights.get('smape', 0):.1f}")
        ])
        
        return html.Div([
            weights_summary,
            html.H5("Multi-Metric Ranking with Composite Scores"),
            html.P("🥇 Top 3: Green | 🥉 Bottom 3: Red | Composite Score: Weighted combination of all metrics"),
            table
        ])
    
    
    def _create_enhanced_summary(self, df, view_mode):
        """Create enhanced summary statistics."""
        if df.empty:
            return "No data selected"
        
        n_models = len(df.index.get_level_values('model_id').unique())
        n_splits = len(df.index.get_level_values('slice_id').unique())
        n_obs = len(df)
        date_range = f"{df.index.get_level_values('date').min():%Y-%m-%d} to {df.index.get_level_values('date').max():%Y-%m-%d}"
        
        train_count = len(df[df['period_type'] == 'train'])
        test_count = len(df[df['period_type'] == 'test'])
        
        return f"📊 Models: {n_models} | 🔄 Splits: {n_splits} | 📈 Observations: {n_obs:,} | 🗓️ Range: {date_range} | 🏋️ Train: {train_count:,} | 🧪 Test: {test_count:,}"
    
    def run(self, debug=False, port=8050):
        """Run the enhanced dashboard."""
        print(f"🚀 Starting Enhanced Time Series Dashboard on http://localhost:{port}")
        print("📊 Features available:")
        print("   • Interactive time series visualization")
        print("   • Comprehensive performance metrics")
        print("   • Residual analysis dashboard")
        print("   • Model comparison tools")
        print("   • Data export capabilities")
        print("   • Advanced filtering and exploration")
        self.app.run_server(debug=debug, port=port)


def create_interactive_dashboard(
    resamples_df: pd.DataFrame,
    accuracy_df: Optional[pd.DataFrame] = None,
    title: str = "Time Series Model Analysis Dashboard",
    port: int = 8050,
    debug: bool = False
) -> EnhancedResamplesDashboard:
    """
    Create and optionally run an enhanced interactive dashboard for exploring results.
    
    Args:
        resamples_df: Output from fit_resamples
        accuracy_df: Output from resample_accuracy (optional)
        title: Dashboard title
        port: Port to run the dashboard on
        debug: Whether to run in debug mode
        
    Returns:
        EnhancedResamplesDashboard instance
        
    Example:
        >>> dashboard = create_interactive_dashboard(
        ...     resamples_df=results,
        ...     accuracy_df=accuracy,
        ...     title="My Enhanced Model Analysis"
        ... )
        >>> dashboard.run(port=8050)
    """
    dashboard = EnhancedResamplesDashboard(resamples_df, accuracy_df, title)
    return dashboard 