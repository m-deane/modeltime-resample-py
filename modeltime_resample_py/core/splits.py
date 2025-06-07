import pandas as pd
import numpy as np
import warnings # Import warnings module
from typing import Union, Tuple, Optional, List, Any

def _get_pandas_offset(freq_string: str) -> pd.DateOffset:
    """Converts a frequency string (like '1 year', '3 months', 'D', 'W') to a pandas DateOffset."""

    # === Priority Handling for Weeks ===
    # Ensure both "1 week" (R-like) and "W" (pandas) resolve to DateOffset(weeks=1)
    # This seems necessary for consistent sample counting with searchsorted.
    parts = freq_string.split()
    if len(parts) == 2 and parts[0].isdigit():
        num = int(parts[0])
        unit = parts[1].lower().rstrip('s')
        if unit == "week":
            return pd.DateOffset(weeks=num)
        # Defer other R-like parsing
    elif freq_string.upper() == 'W':
         # Explicitly map 'W' to DateOffset(weeks=1)
         return pd.DateOffset(weeks=1)

    # === Standard Pandas Frequency Strings (excluding W) ===
    try:
        # Uppercase single chars like D, M, Y for robustness
        if len(freq_string) == 1 and freq_string.isalpha():
            processed_freq = freq_string.upper()
        else:
            processed_freq = freq_string

        # This will now handle D, M, Y, H, T, S etc., but not W
        offset = pd.tseries.frequencies.to_offset(processed_freq)

        # Check for originally lowercase strings and warn
        if isinstance(offset, pd.DateOffset) and freq_string.islower() and not freq_string.isdigit():
             unit_char = offset.name.rstrip('S') # Get base unit like D, M, Y
             if unit_char in ['D', 'M', 'Q', 'Y', 'H', 'T', 'S']: # Exclude W
                 warnings.warn(
                     f"Frequency string '{freq_string}' uses lowercase units. Using uppercase (e.g., '{unit_char}') is preferred.",
                     FutureWarning # Keep warning minimal
                 )
        return offset
    except ValueError:
        # === Fallback R-like Parsing (Month, Year, Quarter, etc.) ===
        # This section only runs if standard parsing failed AND it wasn't handled above
        parts = freq_string.split() # Recalculate parts
        if len(parts) == 2 and parts[0].isdigit():
            num = int(parts[0])
            unit = parts[1].lower().rstrip('s')

            # Map common R units (excluding week, handled above)
            unit_map = {
                "sec": "seconds", "min": "minutes", "hour": "hours", "day": "days",
            }
            if unit == "month":
                return pd.offsets.MonthBegin(num)
            elif unit == "quarter":
                 return pd.offsets.QuarterBegin(num, startingMonth=1)
            elif unit == "year":
                return pd.offsets.YearBegin(num)
            elif unit in unit_map:
                pd_unit_arg = unit_map[unit]
                try:
                    offset_kwargs = {pd_unit_arg: num}
                    return pd.DateOffset(**offset_kwargs)
                except Exception as e:
                    raise ValueError(f"Could not create DateOffset from '{freq_string}': {e}")
            else:
                 # Only raise if unit wasn't handled (week was handled above)
                 raise ValueError(f"Could not parse R-like frequency string unit: '{unit}'")
        else:
             # If not standard and not R-like, raise final error
             raise ValueError(f"Could not parse frequency string: '{freq_string}'")

def _get_sample_count(period: Union[str, int], time_index: pd.DatetimeIndex, start_idx: int = 0, end_idx: Optional[int] = None) -> int:
    """Calculates the number of samples within a given period relative to a time index slice."""
    n_full = len(time_index)
    if end_idx is None:
        end_idx = n_full

    # Handle potential out-of-bounds / empty slices first
    if start_idx >= n_full or start_idx >= end_idx or end_idx <= 0:
        return 0

    # Clamp indices to valid range [0, n_full]
    start_idx = max(0, start_idx)
    end_idx = min(n_full, end_idx)

    # Recalculate slice_len after clamping
    slice_len = end_idx - start_idx
    if slice_len <= 0:
        return 0 # Return 0 if the effective slice is empty

    if isinstance(period, int):
        if period <= 0:
            raise ValueError(f"Integer period must be positive, got {period}.")
        return min(period, slice_len)
    elif isinstance(period, str):
        offset = _get_pandas_offset(period)
        # Check for non-positive/zero duration offsets
        is_positive = getattr(offset, 'n', 1) > 0 or \
                      any(getattr(offset, attr, 0) > 0 for attr in
                          ['days', 'seconds', 'microseconds', 'minutes', 'hours', 'weeks', 'months', 'years'])
        if not is_positive:
             raise ValueError(f"Period string '{period}' must represent a positive duration.")

        start_date = time_index[start_idx]
        end_date = start_date + offset

        # Use side='left' to find first index >= end_date
        insertion_idx = np.searchsorted(time_index, end_date, side='left')

        # Count is the number of points from start_idx up to (but not including) insertion_idx
        count = insertion_idx - start_idx
        count = max(0, count) # Ensure non-negative
        count = min(count, slice_len) # Ensure count doesn't exceed available slice

        # Raise ValueError only if a positive period resulted in zero samples
        # AND the slice wasn't already empty.
        # This distinguishes between a period being too short for the frequency,
        # and trying to calculate from an already-empty part of the index.
        if count == 0 and slice_len > 0:
            raise ValueError(f"Period '{period}' resulted in zero samples within the slice from index {start_idx} to {end_idx}. Check period duration relative to data frequency and slice length.")

        return count
    else:
        raise TypeError("Period must be an integer or a time period string.")

def _get_skip_count(skip: Union[str, int], time_index: pd.DatetimeIndex, current_assess_start_idx: int) -> int:
    """Calculates the number of samples to skip."""
    if isinstance(skip, int):
        # Allow skip == 0, raise error for negative skip
        if skip < 0:
            raise ValueError(f"Integer skip must be non-negative, got {skip}.")
        # Check index bounds before returning, although CV loop likely handles this too
        if current_assess_start_idx >= len(time_index):
             # This might be overly strict if CV handles it, but good for direct calls
             raise IndexError("current_assess_start_idx is out of bounds before applying skip.")
        return skip # Return 0 or positive integer skip
    elif isinstance(skip, str):
        if current_assess_start_idx >= len(time_index):
             raise IndexError("current_assess_start_idx is out of bounds for skip calculation.")

        offset = _get_pandas_offset(skip)
        # Explicitly check for and disallow zero/negative offsets here
        is_positive = getattr(offset, 'n', 1) > 0 or \
                      any(getattr(offset, attr, 0) > 0 for attr in
                          ['days', 'seconds', 'microseconds', 'minutes', 'hours', 'weeks', 'months', 'years'])
        if not is_positive:
            # Raise error immediately if the resolved offset is not positive
            raise ValueError(f"Skip period string '{skip}' must represent a positive duration.")

        # Find insertion point for the end date *after* the skip
        start_date = time_index[current_assess_start_idx]
        skip_end_date = start_date + offset
        insertion_idx = np.searchsorted(time_index, skip_end_date, side='left')

        # Number of samples TO skip is the difference in index, relative to current position
        skip_count = insertion_idx - current_assess_start_idx

        # Ensure skip_count isn't negative (shouldn't happen with positive offset check)
        skip_count = max(0, skip_count)

        # Added check: if a positive offset string resulted in 0 skip count, raise error
        # This can happen if the skip period is shorter than the time between indices
        if skip_count == 0:
             # Avoid raising if the start index was already the last index (no room to skip)
             if current_assess_start_idx < len(time_index) - 1:
                 raise ValueError(f"Skip period '{skip}' resulted in zero samples skipped from index {current_assess_start_idx}. Check skip duration relative to data frequency.")
             else:
                 # If starting at the last index, a skip count of 0 is expected/valid
                 pass


        return skip_count
    else:
        raise TypeError("Skip must be an integer or a time period string.")

def time_series_split(
    data: Union[pd.DataFrame, pd.Series],
    initial: Union[str, int],
    assess: Union[str, int],
    cumulative: bool = False,
    date_column: Optional[str] = None
) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    """
    Creates a single binary split of the time series into training and testing sets.
    Mimics the behavior of `rsample::initial_time_split` or the first split of `time_series_cv`.

    Args:
        data: A pandas DataFrame or Series with a DatetimeIndex or a date column.
        initial: The amount of data initially used for training. Can be specified as:
            - An integer: Number of samples.
            - A string: Time period (e.g., '5 years', '6 months', '50 days').
        assess: The amount of data used for the assessment/testing set. Specified similarly to `initial`.
        cumulative: If `False` (default), the training set size is fixed at `initial`.
                    If `True`, the training set size grows in subsequent splits (though this function only produces one split).
                    This parameter primarily influences how the split point is determined based on `initial` and `assess` when
                    these are specified as time periods relative to the *end* of the series.
        date_column: The name of the column containing the date information if `data` is a DataFrame
                     and does not have a DatetimeIndex. If `None`, the index is assumed to be the date.

    Returns:
        A tuple containing the training DataFrame/Series and the testing DataFrame/Series.

    Raises:
        ValueError: If input data, indices, or parameters are invalid.
        TypeError: If `data` is not a pandas DataFrame or Series.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Input 'data' must be a pandas DataFrame or Series.")

    if not isinstance(initial, (int, str)):
        raise TypeError("'initial' must be an integer or a time period string.")
    if not isinstance(assess, (int, str)):
        raise TypeError("'assess' must be an integer or a time period string.")

    # --- Index Preparation and Sorting (with warnings) ---
    original_data = data
    if date_column:
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
             raise ValueError(f"Date column '{date_column}' must be of datetime type.")
        time_index = pd.DatetimeIndex(data[date_column])
        if not time_index.is_monotonic_increasing:
             warnings.warn("Date column is not monotonically increasing. Sorting data by date.", UserWarning)
             # Use numpy argsort on the index for potentially faster sorting index retrieval
             sorted_indices = np.argsort(time_index)
             data = original_data.iloc[sorted_indices].reset_index(drop=True)
             time_index = pd.DatetimeIndex(data[date_column]) # Update index after sorting
    elif isinstance(data.index, pd.DatetimeIndex):
        time_index = data.index
        if not time_index.is_monotonic_increasing:
             warnings.warn("DatetimeIndex is not monotonically increasing. Sorting data by index.", UserWarning)
             data = original_data.sort_index()
             time_index = data.index
    else:
        raise ValueError("Data must have a DatetimeIndex or a valid 'date_column' specified.")

    n_total = len(data)
    idx = np.arange(n_total)

    # --- Calculate and Validate Initial Training Size ---
    try:
        n_initial = _get_sample_count(initial, time_index, 0, n_total)
    except ValueError as e:
         # Catch errors from _get_sample_count (e.g., zero period, invalid string)
         raise ValueError(f"Error calculating initial training size for '{initial}': {e}")

    if n_initial <= 0:
         # This case might be caught by _get_sample_count, but double check
         raise ValueError(f"'initial' period/count results in zero or negative training samples ({n_initial}).")
    if n_initial >= n_total:
         raise ValueError(f"'initial' period/count ('{initial}') results in training set size ({n_initial}) that covers or exceeds total samples ({n_total}).")

    # --- Calculate and Validate Assessment Size ---
    try:
        # Assess size is calculated starting *after* the initial block
        n_assess = _get_sample_count(assess, time_index, n_initial, n_total)
    except ValueError as e:
         # Catch errors like zero period or insufficient remaining data for the period
         raise ValueError(f"Error calculating assessment size for '{assess}' after initial {n_initial} samples: {e}")

    if n_assess <= 0:
         # This case should ideally be caught by _get_sample_count raising error for 0 samples
         raise ValueError(f"'assess' period/count ('{assess}') results in zero or negative testing samples ({n_assess}) after initial training set (size {n_initial}).")

    # Final check if calculated initial + assess exceeds total (redundant if _get_sample_count is correct)
    if n_initial + n_assess > n_total:
         raise ValueError(f"The sum of calculated initial ({n_initial}) and assess ({n_assess}) samples exceeds the total number of samples ({n_total}). This may indicate an issue with period calculation near the end of the series.")

    # --- Define split indices ---
    train_indices = idx[:n_initial]
    test_indices = idx[n_initial : n_initial + n_assess]

    # --- Slice the data ---
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    return train_data, test_data

def time_series_cv(
    data: Union[pd.DataFrame, pd.Series],
    initial: Union[str, int],
    assess: Union[str, int],
    skip: Union[str, int] = "1D",
    cumulative: bool = False,
    slice_limit: int = 5,
    date_column: Optional[str] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates time series cross-validation folds using a rolling origin strategy.

    Generates a list of tuples, where each tuple contains the integer indices
    for the training and assessment (testing) sets for each fold.

    Args:
        data: A pandas DataFrame or Series with a DatetimeIndex or a date column.
        initial: The amount of data initially used for training in the first fold.
                 Can be specified as an integer (number of samples) or a string
                 (time period like '5 years', '6 months').
        assess: The amount of data used for the assessment/testing set in each fold.
                Specified similarly to `initial`.
        skip: The number of samples or time period to skip between folds.
              Defaults to "1D" (1 day). An integer skip of 0 means the assessment window moves
              forward by exactly the size of the previous assessment window. String periods
              must represent positive durations.
        cumulative: If `False` (default), the training set size is fixed (rolling window).
                    If `True`, the training set size grows (expanding window).
        slice_limit: The maximum number of slices or folds to generate. Defaults to 5.
        date_column: The name of the column containing date information if `data` is a
                     DataFrame without a DatetimeIndex. If `None`, the index is assumed
                     to be the date.

    Returns:
        A list of tuples. Each tuple contains two numpy arrays:
        `(train_indices, test_indices)` for a single cross-validation fold.

    Raises:
        ValueError: If input data, indices, or parameters are invalid or lead to empty splits.
        TypeError: If `data` is not a pandas DataFrame or Series, or parameters have wrong types.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Input 'data' must be a pandas DataFrame or Series.")

    if not isinstance(initial, (int, str)):
        raise TypeError("'initial' must be an integer or a time period string.")
    if not isinstance(assess, (int, str)):
        raise TypeError("'assess' must be an integer or a time period string.")
    if not isinstance(skip, (int, str)):
         raise TypeError("'skip' must be an integer or a time period string.")
    if isinstance(skip, int) and skip < 0:
         raise ValueError(f"Integer skip must be non-negative, got {skip}.")
    if not isinstance(slice_limit, int) or slice_limit <= 0:
        raise ValueError("slice_limit must be a positive integer.")

    # --- Data and Index Preparation ---
    original_data = data # Keep reference if sorting happens
    if date_column:
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
             raise ValueError(f"Date column '{date_column}' must be of datetime type.")
        time_index = pd.DatetimeIndex(data[date_column])
        if not time_index.is_monotonic_increasing:
             warnings.warn("Date column is not monotonically increasing. Sorting data by date.", UserWarning)
             sorted_indices = np.argsort(time_index)
             data = original_data.iloc[sorted_indices].reset_index(drop=True)
             time_index = pd.DatetimeIndex(data[date_column]) # Update index after sorting
    elif isinstance(data.index, pd.DatetimeIndex):
        time_index = data.index
        if not time_index.is_monotonic_increasing:
             warnings.warn("DatetimeIndex is not monotonically increasing. Sorting data by index.", UserWarning)
             data = original_data.sort_index()
             time_index = data.index
    else:
        raise ValueError("Data must have a DatetimeIndex or a valid 'date_column' specified.")

    n_total = len(data)
    idx = np.arange(n_total)

    # --- Parameter Validation using Helpers (BEFORE loop) ---
    try:
        n_initial = _get_sample_count(initial, time_index, 0, n_total)
    except ValueError as e:
         raise ValueError(f"Error validating 'initial' ('{initial}'): {e}")

    if n_initial <= 0: # Redundant check if _get_sample_count raises, but safe
        raise ValueError(f"'initial' period/count results in zero/negative training samples ({n_initial}).")
    if n_initial >= n_total:
        raise ValueError(f"'initial' period/count ('{initial}') results in training set size ({n_initial}) that covers or exceeds total samples ({n_total}).")

    try:
        n_assess_first = _get_sample_count(assess, time_index, n_initial, n_total)
    except ValueError as e:
        raise ValueError(f"Error validating 'assess' ('{assess}') after initial {n_initial} samples: {e}")

    if n_assess_first <= 0:
        raise ValueError(f"'assess' period/count results in zero/negative testing samples ({n_assess_first}) after initial training set (size {n_initial}).")

    # --- Validate skip value using _get_skip_count (before loop) ---
    # Use the starting index of the *first* assessment window for validation
    first_assess_start_idx = n_initial
    if first_assess_start_idx < n_total: # Only validate skip if there's room for assessment
        try:
            # Call _get_skip_count just to validate the skip parameter itself
            _ = _get_skip_count(skip, time_index, first_assess_start_idx)
        except (ValueError, IndexError) as e:
             # Reraise validation error originating from skip parameter
             raise ValueError(f"Error validating 'skip' ('{skip}'): {e}")

    # --- Final check for initial+assess feasibility ---
    if n_initial + n_assess_first > n_total:
        # This check should now be more reliable
        raise ValueError(f"The sum of initial ({n_initial}) and the first assess ({n_assess_first}) samples exceeds the total number of samples ({n_total}).")

    # --- Generate Splits ---
    splits = []
    current_assess_start_idx = n_initial

    # --- DEBUG FLAG --- (Keep disabled)
    debug_case = False
    # if debug_case: print(...) # Keep debug prints commented out

    for i in range(slice_limit):
        # if debug_case: print(...) # Keep debug prints commented out

        # Check if the current assessment start index is already out of bounds
        if current_assess_start_idx >= n_total:
            # if debug_case: print(...) # Keep debug prints commented out
            break

        # --- Calculate current assessment indices ---
        try:
            if isinstance(assess, str):
                n_assess_current = _get_sample_count(assess, time_index, current_assess_start_idx, n_total)
            else: # assess is int
                n_assess_current = min(assess, n_total - current_assess_start_idx)
        except ValueError as e:
            # if debug_case: print(...) # Keep debug prints commented out
            break

        # if debug_case: print(...) # Keep debug prints commented out
        if n_assess_current <= 0:
            # if debug_case: print(...) # Keep debug prints commented out
            break

        current_assess_end_idx = current_assess_start_idx + n_assess_current
        current_assess_end_idx = min(current_assess_end_idx, n_total)
        n_assess_current = current_assess_end_idx - current_assess_start_idx
        if n_assess_current <= 0:
            # if debug_case: print(...) # Keep debug prints commented out
            break
        # if debug_case: print(...) # Keep debug prints commented out

        # --- Calculate current training indices ---
        if cumulative:
            current_train_start_idx = 0
        else: # Rolling window
            if isinstance(initial, str):
                 try:
                      n_initial_current = _get_sample_count(initial, time_index, 0, current_assess_start_idx)
                      current_train_start_idx = current_assess_start_idx - n_initial_current
                 except ValueError:
                      current_train_start_idx = current_assess_start_idx - n_initial # Fallback
            else: # initial is int
                current_train_start_idx = current_assess_start_idx - initial
            current_train_start_idx = max(0, current_train_start_idx)
        # if debug_case: print(...) # Keep debug prints commented out

        train_indices = idx[current_train_start_idx:current_assess_start_idx]
        test_indices = idx[current_assess_start_idx:current_assess_end_idx]

        # --- Validate split sizes and Append ---
        if len(train_indices) == 0 or len(test_indices) == 0:
            warnings.warn(f"Skipping split generation at assess_start_idx {current_assess_start_idx} as it resulted in empty train ({len(train_indices)}) or test ({len(test_indices)}) set.", UserWarning)
            # if debug_case: print(...) # Keep debug prints commented out
            break

        splits.append((train_indices, test_indices))
        # if debug_case: print(...) # Keep debug prints commented out

        # --- Calculate skip count for the *next* iteration ---
        try:
            n_skip_current = _get_skip_count(skip, time_index, current_assess_start_idx)
        except (ValueError, IndexError) as e:
            warnings.warn(f"Stopping split generation due to error during skip calculation: {e}", UserWarning)
            # if debug_case: print(...) # Keep debug prints commented out
            break
        # if debug_case: print(...) # Keep debug prints commented out

        # --- Determine next assessment start index ---
        if isinstance(skip, int) and skip == 0:
            next_assess_start_idx = current_assess_end_idx
        else:
             next_assess_start_idx = current_assess_start_idx + n_skip_current
        # if debug_case: print(...) # Keep debug prints commented out

        # *** Check if the next start index is valid BEFORE continuing loop ***
        if next_assess_start_idx >= n_total:
            # if debug_case: print(...) # Keep debug prints commented out
            break # Stop loop, don't update current_assess_start_idx

        # If valid, update for the next iteration
        current_assess_start_idx = next_assess_start_idx

    # if debug_case: print(...) # Keep debug prints commented out

    # --- Final Warning if No Splits Generated ---
    if not splits:
         warnings.warn("No valid cross-validation splits generated. Check input parameters relative to data length and frequency.", UserWarning)

    return splits
 