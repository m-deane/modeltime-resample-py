"""Parallel processing utilities for modeltime_resample_py."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Any, Optional, Union
from joblib import Parallel, delayed
import multiprocessing
from tqdm.auto import tqdm

from .core.modeling import fit_resamples


def fit_resamples_parallel(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    model_spec: Any,
    data: Union[pd.DataFrame, pd.Series],
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    date_column: Optional[str] = None,
    model_id: Optional[str] = "model",
    n_jobs: int = -1,
    verbose: int = 1,
    backend: str = 'loky',
    batch_size: str = 'auto'
) -> pd.DataFrame:
    """
    Parallel version of fit_resamples using joblib.
    
    Fits a model to each CV split in parallel, significantly speeding up
    computation for large datasets or complex models.
    
    Args:
        cv_splits: List of (train_indices, test_indices) tuples
        model_spec: Model with fit() and predict() methods
        data: Time series data
        target_column: Target column name
        feature_columns: Feature column names
        date_column: Date column name
        model_id: Model identifier
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level (0=silent, 1=progress bar, 2=detailed)
        backend: Joblib backend ('loky', 'threading', 'multiprocessing')
        batch_size: Batch size for parallel execution
        
    Returns:
        DataFrame with fitted values and predictions (same as fit_resamples)
        
    Example:
        >>> from modeltime_resample_py.parallel import fit_resamples_parallel
        >>> results = fit_resamples_parallel(
        ...     cv_splits, model, data, 'value',
        ...     n_jobs=-1,  # Use all cores
        ...     verbose=1   # Show progress
        ... )
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs == 0:
        # Fall back to sequential processing
        return fit_resamples(
            cv_splits, model_spec, data, target_column,
            feature_columns, date_column, model_id
        )
    
    # Prepare data once to avoid serialization overhead
    data_dict = _prepare_data_for_parallel(data, target_column, feature_columns, date_column)
    
    # Define function to fit single split
    def fit_single_split(split_idx: int, train_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
        """Fit model on a single CV split."""
        try:
            # Clone model if sklearn is available
            try:
                from sklearn.base import clone
                model_copy = clone(model_spec)
            except:
                import copy
                model_copy = copy.deepcopy(model_spec)
            
            # Fit on single split
            result = _fit_single_resample(
                split_idx, train_idx, test_idx,
                model_copy, data_dict, target_column,
                feature_columns, model_id
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Error in split {split_idx}: {str(e)}")
    
    # Create progress bar if verbose
    if verbose >= 1:
        splits_with_progress = tqdm(
            enumerate(cv_splits),
            total=len(cv_splits),
            desc="Fitting models"
        )
    else:
        splits_with_progress = enumerate(cv_splits)
    
    # Parallel execution
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        delayed(fit_single_split)(i, train, test)
        for i, (train, test) in splits_with_progress
    )
    
    # Combine results
    final_df = pd.concat(results, ignore_index=False)
    final_df = final_df.sort_index()
    
    return final_df


def _prepare_data_for_parallel(
    data: Union[pd.DataFrame, pd.Series],
    target_column: str,
    feature_columns: Optional[List[str]],
    date_column: Optional[str]
) -> dict:
    """Prepare data dictionary for parallel processing."""
    # Convert to DataFrame if Series
    if isinstance(data, pd.Series):
        df = data.to_frame(name=target_column)
    else:
        df = data.copy()
    
    # Extract time index
    if date_column:
        time_index = pd.DatetimeIndex(df[date_column])
    elif isinstance(df.index, pd.DatetimeIndex):
        time_index = df.index
    else:
        raise ValueError("Data must have DatetimeIndex or date_column")
    
    return {
        'data': df,
        'time_index': time_index,
        'target_column': target_column,
        'feature_columns': feature_columns,
        'date_column': date_column
    }


def _fit_single_resample(
    split_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model: Any,
    data_dict: dict,
    target_column: str,
    feature_columns: Optional[List[str]],
    model_id: str
) -> pd.DataFrame:
    """Fit model on single resample split."""
    df = data_dict['data']
    time_index = data_dict['time_index']
    
    # Get train/test data
    train_data = df.iloc[train_idx]
    test_data = df.iloc[test_idx]
    
    # Extract features and target
    y_train = train_data[target_column]
    y_test = test_data[target_column]
    
    if feature_columns:
        X_train = train_data[feature_columns]
        X_test = test_data[feature_columns]
    else:
        # Create dummy feature
        X_train = pd.DataFrame({'dummy': np.zeros(len(train_data))})
        X_test = pd.DataFrame({'dummy': np.zeros(len(test_data))})
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Get predictions
    fitted_values = model.predict(X_train)
    predictions = model.predict(X_test)
    
    # Create results DataFrame
    train_results = pd.DataFrame({
        'date': time_index[train_idx],
        'slice_id': split_idx,
        'model_id': model_id,
        'actuals': y_train.values,
        'fitted_values': fitted_values,
        'predictions': np.nan,
        'residuals': y_train.values - fitted_values,
        'period_type': 'train',
        'train_start_date': time_index[train_idx[0]],
        'train_end_date': time_index[train_idx[-1]],
        'test_start_date': time_index[test_idx[0]],
        'test_end_date': time_index[test_idx[-1]]
    })
    
    test_results = pd.DataFrame({
        'date': time_index[test_idx],
        'slice_id': split_idx,
        'model_id': model_id,
        'actuals': y_test.values,
        'fitted_values': np.nan,
        'predictions': predictions,
        'residuals': y_test.values - predictions,
        'period_type': 'test',
        'train_start_date': time_index[train_idx[0]],
        'train_end_date': time_index[train_idx[-1]],
        'test_start_date': time_index[test_idx[0]],
        'test_end_date': time_index[test_idx[-1]]
    })
    
    # Combine and set index
    results = pd.concat([train_results, test_results])
    results = results.set_index(['date', 'slice_id', 'model_id'])
    
    return results


# Update convenience functions to support parallel processing
def evaluate_model_parallel(
    data: Union[pd.DataFrame, pd.Series],
    model: Any,
    initial: Union[str, int] = '1 year',
    assess: Union[str, int] = '3 months',
    skip: Union[str, int] = '3 months',
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    date_column: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    period_types: Optional[List[str]] = None,
    cumulative: bool = False,
    slice_limit: int = 5,
    model_id: str = "model",
    n_jobs: int = -1,
    verbose: int = 1
) -> pd.DataFrame:
    """
    Parallel version of evaluate_model using multiple cores.
    
    See evaluate_model for parameter descriptions.
    Additional parameters:
        n_jobs: Number of parallel jobs (-1 for all cores)
        verbose: Verbosity level
    """
    from .core.splits import time_series_cv
    from .metrics.accuracy import resample_accuracy, DEFAULT_METRICS
    
    # Generate CV splits
    cv_splits = time_series_cv(
        data=data,
        initial=initial,
        assess=assess,
        skip=skip,
        cumulative=cumulative,
        slice_limit=slice_limit,
        date_column=date_column
    )
    
    # Handle target column for Series
    if isinstance(data, pd.Series) and target_column is None:
        target_column = data.name if data.name else 'value'
    
    # Fit model to all resamples in parallel
    resamples_df = fit_resamples_parallel(
        cv_splits=cv_splits,
        model_spec=model,
        data=data,
        target_column=target_column,
        feature_columns=feature_columns,
        date_column=date_column,
        model_id=model_id,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    # Calculate accuracy metrics
    if metrics is None:
        metrics = ['mae', 'rmse']
    
    if period_types is None:
        period_types = ['test']
    
    metrics_dict = {}
    for metric_name in metrics:
        if metric_name in DEFAULT_METRICS:
            metrics_dict[metric_name] = DEFAULT_METRICS[metric_name]
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    accuracy_df = resample_accuracy(
        resamples_df=resamples_df,
        metrics_set=metrics_dict,
        period_types_to_evaluate=period_types
    )
    
    return accuracy_df 