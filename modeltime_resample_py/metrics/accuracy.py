"""Accuracy metrics for time series resampling."""

import pandas as pd
import numpy as np
from typing import Optional, List
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error

DEFAULT_METRICS = {
    "mae": mean_absolute_error,
    "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
}

def resample_accuracy(
    resamples_df: pd.DataFrame, # Output from fit_resamples (new long format)
    metrics_set: Optional[dict] = None, # Dict of name: callable(y_true, y_pred)
    period_types_to_evaluate: Optional[List[str]] = None # e.g. ['train', 'test']
) -> pd.DataFrame:
    """
    Calculates accuracy metrics for each split from the long-format output of fit_resamples.
    Metrics can be calculated for specified period types (e.g., 'train', 'test').

    Args:
        resamples_df: DataFrame output from fit_resamples (new long format).
                      Expected to be indexed by ['date', 'slice_id', 'model_id'] and contain
                      columns: 'actuals', 'fitted_values', 'predictions', 'period_type'.
        metrics_set: A dictionary where keys are metric names (str) and values are
                     callable functions that take (y_true, y_pred) and return a scalar metric.
                     If None, uses DEFAULT_METRICS (mae, rmse).
        period_types_to_evaluate: A list of period types (strings, e.g., ['train', 'test'])
                                  for which to calculate accuracy. If None, defaults to ['test'].

    Returns:
        A pandas DataFrame with columns: 
        ['slice_id', 'model_id', 'period_type', 'metric_name', 'metric_value'].

    Raises:
        TypeError: If `resamples_df` is not a pandas DataFrame.
        TypeError: If `metrics_set` is not a dictionary or None.
        ValueError: If `resamples_df` is not indexed correctly.
        ValueError: If `resamples_df` is missing required columns for the specified period_types.
        ValueError: If `period_types_to_evaluate` contains invalid period types.
    """
    if not isinstance(resamples_df, pd.DataFrame):
        raise TypeError("resamples_df must be a pandas DataFrame.")

    expected_index_names = ['date', 'slice_id', 'model_id']
    if list(resamples_df.index.names) != expected_index_names:
        raise ValueError(f"resamples_df must be indexed by {expected_index_names}.")

    base_required_cols = ['actuals', 'period_type']
    # Check for predictions/fitted_values based on what will be evaluated
    _effective_period_types = period_types_to_evaluate if period_types_to_evaluate is not None else ['test']
    if not _effective_period_types: # Handle empty list case - no metrics to calculate
        warnings.warn("period_types_to_evaluate is empty. No accuracy metrics will be calculated.", UserWarning)
        return pd.DataFrame(columns=['slice_id', 'model_id', 'period_type', 'metric_name', 'metric_value'])
        
    for ptype in _effective_period_types:
        if ptype == 'test' and 'predictions' not in resamples_df.columns:
            raise ValueError("'predictions' column is required when evaluating 'test' period.")
        if ptype == 'train' and 'fitted_values' not in resamples_df.columns:
            raise ValueError("'fitted_values' column is required when evaluating 'train' period.")
        if ptype not in ['train', 'test']:
            raise ValueError(f"Invalid period_type '{ptype}' in period_types_to_evaluate. Only 'train' or 'test' supported.")
            
    required_cols_present = base_required_cols + [col for col in (['predictions'] if 'test' in _effective_period_types else []) + (['fitted_values'] if 'train' in _effective_period_types else []) if col not in base_required_cols]
    if not all(col in resamples_df.columns for col in required_cols_present):
        missing = [col for col in required_cols_present if col not in resamples_df.columns]
        raise ValueError(f"resamples_df must contain columns: {missing} for the specified period_types_to_evaluate.")

    if resamples_df.empty:
        warnings.warn("Input resamples_df is empty. No accuracy metrics will be calculated.", UserWarning)
        return pd.DataFrame(columns=['slice_id', 'model_id', 'period_type', 'metric_name', 'metric_value'])

    if metrics_set is None:
        metrics_to_calc = DEFAULT_METRICS
    elif not isinstance(metrics_set, dict):
        raise TypeError("metrics_set must be a dictionary or None.")
    else:
        metrics_to_calc = metrics_set

    accuracy_results = []
    unique_groups = resamples_df.index.droplevel(0).unique()

    for slice_id, model_id_val in unique_groups:
        group_data = resamples_df.xs((slice_id, model_id_val), level=('slice_id', 'model_id'))
        
        for ptype in _effective_period_types:
            period_data = group_data[group_data['period_type'] == ptype]

            if period_data.empty:
                warnings.warn(f"No '{ptype}' period data found for slice_id {slice_id}, model_id {model_id_val}. Skipping metrics for this period_type.", UserWarning)
                continue

            actuals = period_data['actuals']
            model_output_values = None
            if ptype == 'test':
                model_output_values = period_data['predictions']
            elif ptype == 'train':
                model_output_values = period_data['fitted_values']
            # else case already handled by earlier validation
            
            if model_output_values is None: # Should not be reached if validation is correct
                 warnings.warn(f"Could not determine model output (predictions/fitted) for period_type '{ptype}' for slice {slice_id}, model {model_id_val}. Skipping.", UserWarning)
                 continue

            # Drop NaNs from model outputs and align actuals
            model_output_values = model_output_values.dropna()
            actuals = actuals.loc[model_output_values.index] 

            if actuals.empty or model_output_values.empty:
                warnings.warn(f"Actuals or model outputs (predictions/fitted_values) are empty after filtering for period_type '{ptype}', slice_id {slice_id}, model_id {model_id_val}. Skipping metrics for this period_type.", UserWarning)
                continue

            # This check might be redundant if alignment works as expected, but good for safety.
            if len(actuals) != len(model_output_values):
                warnings.warn(f"For period_type '{ptype}', slice_id {slice_id}, model_id {model_id_val}, actuals and model outputs have different lengths after processing ({len(actuals)} vs {len(model_output_values)}). Skipping metrics for this period_type.", UserWarning)
                continue

            for metric_name, metric_func in metrics_to_calc.items():
                if not callable(metric_func):
                    warnings.warn(f"Metric function for '{metric_name}' is not callable. Skipping this metric.", UserWarning)
                    continue
                try:
                    metric_value = metric_func(actuals, model_output_values)
                    accuracy_results.append({
                        'slice_id': slice_id,
                        'model_id': model_id_val,
                        'period_type': ptype,
                        'metric_name': metric_name,
                        'metric_value': metric_value
                    })
                except Exception as e:
                    warnings.warn(f"Error calculating metric '{metric_name}' for period_type '{ptype}', slice {slice_id}, model {model_id_val}: {e}. Skipping.", UserWarning)

    if not accuracy_results:
        # This warning might be frequent if, for example, period_types_to_evaluate=['train'] but fitted_values are all NaN (which shouldn't happen post fit_resamples)
        warnings.warn("No accuracy metrics were calculated. Check input data, period_types_to_evaluate, and metrics_set.", UserWarning)
        return pd.DataFrame(columns=['slice_id', 'model_id', 'period_type', 'metric_name', 'metric_value'])

    return pd.DataFrame(accuracy_results) 