"""Core modeling functionality for time series resampling."""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional, List, Any
import warnings

try:
    from sklearn.base import clone
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def fit_resamples(
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    model_spec: Any,
    data: Union[pd.DataFrame, pd.Series],
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    date_column: Optional[str] = None,
    model_id: Optional[str] = "model"
) -> pd.DataFrame:
    """
    Fits a model to each training split from time_series_cv and collects detailed 
    results including actuals, fitted values (for train), predictions (for test), 
    residuals, and split boundary dates.

    The output is a long-format DataFrame indexed by ['date', 'slice_id', 'model_id'].

    Args:
        cv_splits: List of (train_indices, test_indices) tuples from time_series_cv.
        model_spec: A model object with fit(X, y) and predict(X) methods.
                    Uses sklearn.base.clone if available to refit for each fold.
        data: The original pandas DataFrame or Series.
              If DataFrame and `date_column` is None, it must have a DatetimeIndex.
              If Series, its index must be a DatetimeIndex.
        target_column: Name of the column to be predicted.
        feature_columns: List of column names to be used as features (X).
                         If `None`:
                           - If `data` is a `pd.DataFrame`, a dummy feature column (e.g., for
                             intercept-only models) is created.
                           - If `data` is a `pd.Series`, it implies a univariate model where
                             feature engineering (e.g., from lags or the index itself) is
                             handled by the `model_spec`.
        date_column: Name of the date column if data is a DataFrame and not using DatetimeIndex.
                     If data is a Series, this is ignored and the Series' index is used.
        model_id: An identifier for the model being fitted.

    Returns:
        A pandas DataFrame with columns:
        - 'slice_id': (int) Identifier for the cross-validation split.
        - 'model_id': (str) Identifier for the model.
        - 'date': (datetime) The timestamp for the observation.
        - 'actuals': (float) The actual value of the target variable.
        - 'fitted_values': (float) The model's fitted value on the training data (NaN for test data).
        - 'predictions': (float) The model's prediction on the test data (NaN for train data).
        - 'residuals': (float) The difference between actuals and fitted_values/predictions.
        - 'period_type': (str) 'train' or 'test'.
        - 'train_start_date': (datetime) Start date of the training set for this slice.
        - 'train_end_date': (datetime) End date of the training set for this slice.
        - 'test_start_date': (datetime) Start date of the test set for this slice.
        - 'test_end_date': (datetime) End date of the test set for this slice.
        The DataFrame is indexed by ['date', 'slice_id', 'model_id'].

    Raises:
        ValueError: If `model_spec` lacks 'fit' or 'predict' methods.
        ValueError: If `target_column` is not found in `data` (DataFrame).
        ValueError: If `date_column` is specified but not found in `data` (DataFrame).
        ValueError: If `date_column` cannot be converted to datetime.
        ValueError: If `data` is a Series without a name and `target_column` is not provided.
        ValueError: If `data` is a Series and its index is not a DatetimeIndex.
        ValueError: If `data` is a DataFrame without a DatetimeIndex or a valid `date_column`.
        ValueError: If specified `feature_columns` are not in `data`.
        TypeError: If `data` is not a pandas DataFrame or Series.
        RuntimeError: If an error occurs during model fitting or prediction within a split.
    """
    if not hasattr(model_spec, 'fit') or not hasattr(model_spec, 'predict'):
        raise ValueError("model_spec must have 'fit' and 'predict' methods.")

    model_instance_for_loop: Any
    if SKLEARN_AVAILABLE:
        # sklearn.base.clone will be used inside the loop
        model_instance_for_loop = model_spec
    else:
        warnings.warn(
            "scikit-learn not available. Attempting to deepcopy the model. "
            "If deepcopy fails or is not sufficient, the model might be refit in place. "
            "For safety, pass a new instance of your model or ensure it can be refit without side effects.",
            UserWarning
        )
        try:
            import copy
            model_instance_for_loop = copy.deepcopy(model_spec)
        except Exception as e:
            warnings.warn(
                f"Failed to deepcopy model_spec (error: {e}). Model will be used as is and potentially refit in place. "
                "This might lead to unexpected behavior if the model retains state across folds.",
                UserWarning
            )
            model_instance_for_loop = model_spec

    all_split_data = []
    original_data_input = data.copy()

    if isinstance(original_data_input, pd.Series):
        if original_data_input.name is None and target_column is None:
            raise ValueError("If data is a Series without a name, target_column must be provided.")
        if target_column is None:
            target_column = original_data_input.name
        elif original_data_input.name is not None and original_data_input.name != target_column:
            warnings.warn(f"data Series name '{original_data_input.name}' differs from target_column '{target_column}'. Using target_column.")

        if not isinstance(original_data_input.index, pd.DatetimeIndex):
            raise ValueError("If data is a Series, its index must be a DatetimeIndex.")
        time_index_master = original_data_input.index
        original_data_df = original_data_input.to_frame(name=target_column)
        master_date_col_name = time_index_master.name if time_index_master.name else 'date'
        if master_date_col_name not in original_data_df.columns:
             original_data_df[master_date_col_name] = time_index_master

    elif isinstance(original_data_input, pd.DataFrame):
        original_data_df = original_data_input
        if target_column not in original_data_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")

        if date_column:
            if date_column not in original_data_df.columns:
                raise ValueError(f"Date column '{date_column}' not found in data.")
            if not pd.api.types.is_datetime64_any_dtype(original_data_df[date_column]):
                try:
                    original_data_df[date_column] = pd.to_datetime(original_data_df[date_column])
                except Exception as e:
                    raise ValueError(f"Date column '{date_column}' could not be converted to datetime: {e}")
            time_index_master = pd.DatetimeIndex(original_data_df[date_column])
            master_date_col_name = date_column
        elif isinstance(original_data_df.index, pd.DatetimeIndex):
            time_index_master = original_data_df.index
            master_date_col_name = time_index_master.name if time_index_master.name else 'date'
            if master_date_col_name not in original_data_df.columns:
                 original_data_df[master_date_col_name] = time_index_master
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a valid 'date_column' specified.")

        if not time_index_master.is_monotonic_increasing:
            warnings.warn(f"Data is not sorted by its time index ('{master_date_col_name}' or DataFrame index). Sorting for consistency.", UserWarning)
            original_data_df = original_data_df.iloc[np.argsort(time_index_master)].reset_index(drop=True)
            time_index_master = pd.DatetimeIndex(original_data_df[master_date_col_name])
    else:
        raise TypeError("Data must be a pandas DataFrame or Series.")

    if master_date_col_name not in original_data_df.columns and isinstance(original_data_df.index, pd.DatetimeIndex) and original_data_df.index.name == master_date_col_name:
        original_data_df[master_date_col_name] = original_data_df.index

    for i, (train_idx, test_idx) in enumerate(cv_splits):
        train_data_fold = original_data_df.iloc[train_idx]
        test_data_fold = original_data_df.iloc[test_idx]

        y_train = train_data_fold[target_column]
        y_test = test_data_fold[target_column]

        train_start_date = time_index_master[train_idx[0]]
        train_end_date = time_index_master[train_idx[-1]]
        test_start_date = time_index_master[test_idx[0]]
        test_end_date = time_index_master[test_idx[-1]]

        if feature_columns:
            missing_features = [col for col in feature_columns if col not in original_data_df.columns]
            if missing_features:
                raise ValueError(f"Feature columns {missing_features} not found in data.")
            X_train = train_data_fold[feature_columns]
            X_test = test_data_fold[feature_columns]
        else: # feature_columns is None
            warnings.warn(
                "feature_columns is None. A dummy feature will be created for compatibility. "
                "If your model is univariate and handles its own feature engineering (e.g., from y or index), "
                "ensure it's compatible with this approach or consider adapting the model.", UserWarning
            )
            X_train = pd.DataFrame({'dummy_feature': np.zeros(len(train_data_fold))}, index=train_data_fold.index)
            X_test = pd.DataFrame({'dummy_feature': np.zeros(len(test_data_fold))}, index=test_data_fold.index)

        current_model: Any
        if SKLEARN_AVAILABLE:
            current_model = clone(model_instance_for_loop) # model_instance_for_loop is model_spec here
        else:
            # model_instance_for_loop is already the deepcopied (or original if copy failed) model
            # If it's not deepcopyable, this will reuse the same instance, which was warned about.
            current_model = model_instance_for_loop
        
        try:
            current_model.fit(X_train, y_train)
            fitted_values_train_arr = current_model.predict(X_train)
            predictions_test_arr = current_model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Error during model fitting or prediction on split {i} for model '{model_id}': {e}. "
                               "Ensure X_train/X_test are correctly formatted for your model.")

        fitted_values_train = pd.Series(fitted_values_train_arr, index=y_train.index, name='fitted_values')
        train_residuals = y_train - fitted_values_train
        
        train_df_split = pd.DataFrame({
            'date': train_data_fold[master_date_col_name],
            'actuals': y_train,
            'fitted_values': fitted_values_train,
            'predictions': np.nan,
            'residuals': train_residuals,
            'period_type': 'train'
        })

        predictions_test = pd.Series(predictions_test_arr, index=y_test.index, name='predictions')
        test_residuals = y_test - predictions_test

        test_df_split = pd.DataFrame({
            'date': test_data_fold[master_date_col_name],
            'actuals': y_test,
            'fitted_values': np.nan,
            'predictions': predictions_test,
            'residuals': test_residuals,
            'period_type': 'test'
        })
        
        split_df = pd.concat([train_df_split, test_df_split], ignore_index=False)
        split_df['slice_id'] = i
        split_df['model_id'] = model_id
        split_df['train_start_date'] = train_start_date
        split_df['train_end_date'] = train_end_date
        split_df['test_start_date'] = test_start_date
        split_df['test_end_date'] = test_end_date
        
        all_split_data.append(split_df)

    if not all_split_data:
        warnings.warn("No results generated. cv_splits might be empty.", UserWarning)
        final_cols = ['date', 'slice_id', 'model_id', 'actuals', 'fitted_values',
                        'predictions', 'residuals', 'period_type', 'train_start_date',
                        'train_end_date', 'test_start_date', 'test_end_date']
        empty_df = pd.DataFrame(columns=final_cols)
        empty_df = empty_df.astype({
            'date': 'datetime64[ns]', 
            'slice_id': 'int64', 
            'model_id': 'object',
            # Ensure other columns have appropriate dtypes for an empty df if needed
            'actuals': 'float64', 'fitted_values': 'float64', 'predictions': 'float64',
            'residuals': 'float64', 'period_type': 'object',
            'train_start_date': 'datetime64[ns]', 'train_end_date': 'datetime64[ns]',
            'test_start_date': 'datetime64[ns]', 'test_end_date': 'datetime64[ns]'
        }).set_index(['date', 'slice_id', 'model_id'])
        return empty_df

    final_df = pd.concat(all_split_data, ignore_index=False)
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.set_index(['date', 'slice_id', 'model_id'])
    final_df = final_df.sort_index()

    return final_df

# Note: resample_accuracy is now in modeltime_resample_py.metrics.accuracy
# Note: plot_resamples is now in modeltime_resample_py.plot.resamples 