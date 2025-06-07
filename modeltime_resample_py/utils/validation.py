"""Validation utilities for modeltime_resample_py."""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional


def validate_data(
    data: Union[pd.DataFrame, pd.Series],
    date_column: Optional[str] = None,
    require_sorted: bool = True
) -> pd.DatetimeIndex:
    """
    Validate time series data and return its datetime index.
    
    Args:
        data: Input data (DataFrame or Series)
        date_column: Name of date column if DataFrame
        require_sorted: Whether to require monotonically increasing dates
        
    Returns:
        pd.DatetimeIndex: The validated datetime index
        
    Raises:
        TypeError: If data is not DataFrame or Series
        ValueError: If date validation fails
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Data must be a pandas DataFrame or Series.")
    
    # Extract datetime index
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Series must have a DatetimeIndex.")
        time_index = data.index
    else:  # DataFrame
        if date_column:
            if date_column not in data.columns:
                raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                raise ValueError(f"Column '{date_column}' must be datetime type.")
            time_index = pd.DatetimeIndex(data[date_column])
        elif isinstance(data.index, pd.DatetimeIndex):
            time_index = data.index
        else:
            raise ValueError("DataFrame must have DatetimeIndex or valid date_column.")
    
    # Check if sorted
    if require_sorted and not time_index.is_monotonic_increasing:
        raise ValueError("Time index must be monotonically increasing (sorted).")
    
    return time_index


def validate_splits(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    n_samples: int
) -> None:
    """
    Validate cross-validation splits.
    
    Args:
        splits: List of (train_indices, test_indices) tuples
        n_samples: Total number of samples in data
        
    Raises:
        ValueError: If splits are invalid
    """
    if not splits:
        raise ValueError("Splits list cannot be empty.")
    
    for i, (train_idx, test_idx) in enumerate(splits):
        # Check types
        if not isinstance(train_idx, np.ndarray) or not isinstance(test_idx, np.ndarray):
            raise TypeError(f"Split {i}: indices must be numpy arrays.")
        
        # Check for empty sets
        if len(train_idx) == 0:
            raise ValueError(f"Split {i}: training set is empty.")
        if len(test_idx) == 0:
            raise ValueError(f"Split {i}: test set is empty.")
        
        # Check indices are within bounds
        if train_idx.max() >= n_samples or train_idx.min() < 0:
            raise ValueError(f"Split {i}: training indices out of bounds.")
        if test_idx.max() >= n_samples or test_idx.min() < 0:
            raise ValueError(f"Split {i}: test indices out of bounds.")
        
        # Check for overlap
        if len(np.intersect1d(train_idx, test_idx)) > 0:
            raise ValueError(f"Split {i}: train and test sets overlap.")
        
        # Check indices are sorted
        if not np.all(train_idx[:-1] <= train_idx[1:]):
            raise ValueError(f"Split {i}: training indices must be sorted.")
        if not np.all(test_idx[:-1] <= test_idx[1:]):
            raise ValueError(f"Split {i}: test indices must be sorted.") 