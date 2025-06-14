"""Convenience functions for common time series modeling workflows."""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any
from .core.splits import time_series_cv
from .core.modeling import fit_resamples
from .metrics.accuracy import resample_accuracy


def evaluate_model(
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
    model_id: str = "model"
) -> pd.DataFrame:
    """
    High-level function to evaluate a model using time series cross-validation.
    
    This convenience function combines time_series_cv, fit_resamples, and 
    resample_accuracy into a single workflow.
    
    Args:
        data: Time series data (DataFrame or Series with DatetimeIndex)
        model: Model object with fit() and predict() methods
        initial: Initial training period size
        assess: Assessment/test period size
        skip: Period to skip between CV folds
        target_column: Name of target column (for DataFrame)
        feature_columns: List of feature column names (for DataFrame)
        date_column: Name of date column if not using DatetimeIndex
        metrics: List of metric names to calculate (default: ['mae', 'rmse'])
        period_types: List of period types to evaluate (default: ['test'])
        cumulative: Whether to use expanding window (True) or rolling window (False)
        slice_limit: Maximum number of CV folds to generate
        model_id: Identifier for the model
        
    Returns:
        DataFrame with accuracy metrics for each CV fold
        
    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression()
        >>> results = evaluate_model(data, model, initial='6 months', assess='1 month')
        >>> print(results.groupby('metric_name')['metric_value'].mean())
    """
    # Default metrics
    if metrics is None:
        metrics = ['mae', 'rmse']
    
    # Default to test-only evaluation
    if period_types is None:
        period_types = ['test']
    
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
    
    # Fit model to all resamples
    resamples_df = fit_resamples(
        cv_splits=cv_splits,
        model_spec=model,
        data=data,
        target_column=target_column,
        feature_columns=feature_columns,
        date_column=date_column,
        model_id=model_id
    )
    
    # Calculate accuracy metrics
    from .metrics.accuracy import DEFAULT_METRICS
    
    # Build metrics dictionary
    metrics_dict = {}
    for metric_name in metrics:
        if metric_name in DEFAULT_METRICS:
            metrics_dict[metric_name] = DEFAULT_METRICS[metric_name]
        else:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(DEFAULT_METRICS.keys())}")
    
    accuracy_df = resample_accuracy(
        resamples_df=resamples_df,
        metrics_set=metrics_dict,
        period_types_to_evaluate=period_types
    )
    
    return accuracy_df


def compare_models(
    data: Union[pd.DataFrame, pd.Series],
    models: Dict[str, Any],
    initial: Union[str, int] = '1 year',
    assess: Union[str, int] = '3 months',
    skip: Union[str, int] = '3 months',
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    date_column: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    period_types: Optional[List[str]] = None,
    cumulative: bool = False,
    slice_limit: int = 5
) -> pd.DataFrame:
    """
    Compare multiple models using the same CV splits.
    
    Args:
        data: Time series data
        models: Dictionary of {model_name: model_object}
        initial, assess, skip: CV parameters
        target_column: Name of target column
        feature_columns: List of feature columns
        date_column: Date column name
        metrics: Metrics to calculate
        period_types: Period types to evaluate
        cumulative: Expanding vs rolling window
        slice_limit: Max CV folds
        
    Returns:
        DataFrame with accuracy metrics for all models
        
    Example:
        >>> from sklearn.linear_model import LinearRegression, Ridge
        >>> models = {
        ...     'linear': LinearRegression(),
        ...     'ridge': Ridge(alpha=1.0)
        ... }
        >>> results = compare_models(data, models)
        >>> # Get average performance by model
        >>> results.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    """
    # Generate CV splits once
    cv_splits = time_series_cv(
        data=data,
        initial=initial,
        assess=assess,
        skip=skip,
        cumulative=cumulative,
        slice_limit=slice_limit,
        date_column=date_column
    )
    
    all_results = []
    
    # Evaluate each model
    for model_name, model in models.items():
        accuracy_df = evaluate_model(
            data=data,
            model=model,
            initial=initial,
            assess=assess,
            skip=skip,
            target_column=target_column,
            feature_columns=feature_columns,
            date_column=date_column,
            metrics=metrics,
            period_types=period_types,
            cumulative=cumulative,
            slice_limit=slice_limit,
            model_id=model_name
        )
        all_results.append(accuracy_df)
    
    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)
    return combined_results


def quick_cv_split(
    data: Union[pd.DataFrame, pd.Series],
    test_size: Union[str, int] = '3 months',
    date_column: Optional[str] = 'date',
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None
) -> tuple:
    """
    Quick single train/test split for time series.
    
    Convenience wrapper around time_series_split for common use case.
    
    Args:
        data: Time series data
        test_size: Size of test set (as periods from end of data)
        date_column: Date column name if DataFrame
        target_column: Target variable column name
        feature_columns: Feature columns for X data
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, X_forecast, y_forecast)
        
    Example:
        >>> X_train, X_test, y_train, y_test, _, _ = quick_cv_split(data, test_size='1 month')
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    from .core.splits import time_series_split
    
    # Calculate split points based on test_size
    if isinstance(data, pd.DataFrame):
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        
        # Convert date column to datetime if needed
        data_copy = data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column])
        dates = data_copy[date_column].sort_values()
        
        # Calculate train_end as the split point before test data
        if isinstance(test_size, int):
            # test_size is number of periods
            train_end_idx = len(dates) - test_size - 1
            if train_end_idx < 0:
                raise ValueError(f"test_size ({test_size}) is larger than data length ({len(dates)})")
            train_end = dates.iloc[train_end_idx]
            test_start = dates.iloc[train_end_idx + 1]
        else:
            # test_size is a time period string
            max_date = dates.max()
            # Use pandas offset to calculate start of test period
            from pandas.tseries.frequencies import to_offset
            offset = to_offset(test_size)
            test_start = max_date - offset
            
            # Find the closest actual date before test_start
            train_end_candidates = dates[dates < test_start]
            if len(train_end_candidates) == 0:
                raise ValueError(f"test_size '{test_size}' covers entire dataset")
            train_end = train_end_candidates.max()
            
            # Find actual test_start date
            test_start_candidates = dates[dates > train_end]
            if len(test_start_candidates) == 0:
                test_start = max_date
            else:
                test_start = test_start_candidates.min()
    
    else:  # Series
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")
        
        dates = data.index.sort_values()
        if isinstance(test_size, int):
            train_end_idx = len(dates) - test_size - 1
            if train_end_idx < 0:
                raise ValueError(f"test_size ({test_size}) is larger than data length ({len(dates)})")
            train_end = dates[train_end_idx]
            test_start = dates[train_end_idx + 1]
        else:
            max_date = dates.max()
            from pandas.tseries.frequencies import to_offset
            offset = to_offset(test_size)
            test_start = max_date - offset
            
            train_end_candidates = dates[dates < test_start]
            if len(train_end_candidates) == 0:
                raise ValueError(f"test_size '{test_size}' covers entire dataset")
            train_end = train_end_candidates.max()
            
            test_start_candidates = dates[dates > train_end]
            if len(test_start_candidates) == 0:
                test_start = max_date
            else:
                test_start = test_start_candidates.min()
    
    return time_series_split(
        data=data,
        train_end=train_end,
        test_start=test_start,
        test_end=dates.max(),
        date_col=date_column,
        X_vars=feature_columns,
        y_var=target_column
    ) 