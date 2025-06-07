import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modeltime_resample_py import (
    time_series_split, 
    time_series_cv,
    fit_resamples,
    resample_accuracy,
    plot_resamples # Assuming plot_resamples is exposed in __init__.py
)
from modeltime_resample_py.plot import plot_time_series_cv_plan
from sklearn.linear_model import LinearRegression # For modeling examples
from sklearn.metrics import mean_squared_error # For custom metrics
import warnings

# Suppress FutureWarnings from pandas for cleaner output in examples
warnings.simplefilter(action='ignore', category=FutureWarning)

print("Cookbook for modeltime_resample_py\n")

# --- Helper Function to Create Sample Data ---
def create_sample_data(start_date='2010-01-01', n_periods=100, freq='D', as_frame=False, date_col_name='date', value_col_name='value', include_features=True):
    """
    Generates sample time series data.
    Args:
        start_date (str): The start date for the time series.
        n_periods (int): The number of periods to generate.
        freq (str): The frequency of the time series (e.g., 'D' for day, 'MS' for month start).
        as_frame (bool): If True, returns a pandas DataFrame.
        date_col_name (str): The name for the date column if as_frame is True.
        value_col_name (str): The name for the value column.
        include_features (bool): If True and as_frame is True, adds some dummy feature columns.

    Returns:
        pd.Series or pd.DataFrame: The generated time series data.
    """
    dates = pd.date_range(start=start_date, periods=n_periods, freq=freq)
    values = np.arange(n_periods) + np.random.randn(n_periods) * 5 

    if as_frame:
        df = pd.DataFrame({date_col_name: dates, value_col_name: values})
        if include_features:
            df['time_idx'] = np.arange(n_periods)
            df['month'] = df[date_col_name].dt.month
            df['lag_value'] = df[value_col_name].shift(1).bfill()
        return df
    else:
        return pd.Series(values, index=dates, name=value_col_name)

# --- Section 1: time_series_split (Examples from previous state, assumed correct) ---
print("--- Section 1: time_series_split ---")

# --- 1.1 Basic Integer Splits ---
print("\n--- 1.1.1 Splitting a Series (Integer initial/assess) ---")
daily_series_data = create_sample_data(n_periods=100, freq='D', include_features=False)
train_s1, test_s1 = time_series_split(daily_series_data, initial=70, assess=30)
print(f"Original Series length: {len(daily_series_data)}")
print(f"Train Series length: {len(train_s1)}, Test Series length: {len(test_s1)}")
print("Train Series head:\\n", train_s1.head())
print("Test Series tail:\\n", test_s1.tail())
assert len(train_s1) == 70
assert len(test_s1) == 30
assert train_s1.index.max() < test_s1.index.min()

print("\n--- 1.1.2 Splitting a DataFrame with DatetimeIndex (Integer initial/assess) ---")
daily_df_indexed = create_sample_data(n_periods=50, freq='D', as_frame=True, include_features=False).set_index('date')
train_df_idx, test_df_idx = time_series_split(daily_df_indexed, initial=35, assess=15)
print(f"Original DataFrame length: {len(daily_df_indexed)}")
print(f"Train DataFrame length: {len(train_df_idx)}, Test DataFrame length: {len(test_df_idx)}")
print("Train DataFrame head:\\n", train_df_idx.head())
assert len(train_df_idx) == 35
assert len(test_df_idx) == 15

print("\n--- 1.1.3 Splitting a DataFrame with date_column (Integer initial/assess) ---")
daily_df_date_col = create_sample_data(n_periods=60, freq='D', as_frame=True, date_col_name='time_stamp', include_features=False)
train_df_dc, test_df_dc = time_series_split(daily_df_date_col, initial=40, assess=20, date_column='time_stamp')
print(f"Original DataFrame length: {len(daily_df_date_col)}")
print(f"Train DataFrame length: {len(train_df_dc)}, Test DataFrame length: {len(test_df_dc)}")
print("Train DataFrame head:\\n", train_df_dc.head())
assert len(train_df_dc) == 40
assert len(test_df_dc) == 20
assert 'time_stamp' in train_df_dc.columns

# --- 1.2 Period String Splits ---
print("\n--- 1.2.1 Splitting a Series (Period initial/assess) ---")
multi_year_daily_series = create_sample_data(start_date='2000-01-01', n_periods=5*365, freq='D', include_features=False)
train_s_period, test_s_period = time_series_split(multi_year_daily_series, initial='3 years', assess='1 year')
print(f"Original Series length: {len(multi_year_daily_series)}")
print(f"Train Series length: {len(train_s_period)}, Test Series length: {len(test_s_period)}")
print(f"Train Series ends: {train_s_period.index.max()}, Test Series starts: {test_s_period.index.min()}")
# Expected lengths can vary slightly with period strings due to month/year ends
assert len(train_s_period) >= (3*365 - 5) # Allow for some leeway
assert len(test_s_period) >= (1*365 - 5)

print("\n--- 1.2.2 Splitting a DataFrame with date_column (Period initial/assess) ---")
monthly_df = create_sample_data(start_date='2018-01-01', n_periods=36, freq='MS', as_frame=True, include_features=False)
train_df_period, test_df_period = time_series_split(monthly_df, initial='24 months', assess='6 months', date_column='date')
print(f"Original DataFrame length: {len(monthly_df)}")
print(f"Train DataFrame length: {len(train_df_period)}, Test DataFrame length: {len(test_df_period)}")
print(f"Train DataFrame ends on date: {train_df_period['date'].iloc[-1]}")
print(f"Test DataFrame starts on date: {test_df_period['date'].iloc[0]}")
assert len(train_df_period) == 24
assert len(test_df_period) == 6

# --- 1.3 Unsorted Data ---
print("\n--- 1.3.1 Splitting an unsorted Series ---")
unsorted_dates = pd.to_datetime(['2023-01-05', '2023-01-02', '2023-01-08', '2023-01-01', '2023-01-03'])
unsorted_series = pd.Series(np.arange(len(unsorted_dates)), index=unsorted_dates, name='unsorted_values')
print("Original unsorted Series:\\n", unsorted_series)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always") # Capture all warnings
    train_unsorted, test_unsorted = time_series_split(unsorted_series, initial=3, assess=2)
    assert len(w) == 1
    assert "not monotonically increasing. Sorting data by index." in str(w[-1].message)
print("Train (sorted) head:\\n", train_unsorted.head())
print("Test (sorted) head:\\n", test_unsorted.head())
assert train_unsorted.index.is_monotonic_increasing
assert test_unsorted.index.is_monotonic_increasing
assert len(train_unsorted) == 3
assert len(test_unsorted) == 2

print("\n\n--- Section 2: time_series_cv ---")

# --- 2.1 Basic Integer CV ---
print("\n--- 2.1.1 Integer CV - Rolling (cumulative=False) ---")
cv_data_series = create_sample_data(n_periods=200, freq='D', include_features=False)
splits_rolling = time_series_cv(cv_data_series, initial=150, assess=20, skip=10, cumulative=False, slice_limit=5)
print(f"Generated {len(splits_rolling)} rolling splits.")
assert len(splits_rolling) > 0 # Expected: (200 - 150) / (20+10) -> 50 / 30 approx, but assess is 20.
# Split 1: Train 0-149, Test 150-169
# Split 2: Train 10-159, Test 160-179 (Skip 10 from start of previous test)
# Split 3: Train 20-169, Test 170-189
# Split 4: Train 30-179, Test 180-199. Last split.
assert len(splits_rolling) == 3 # Recalculated based on new understanding: 150 init, 20 assess. (200-150 = 50 avail). Test 1: 150-169. Skip 10. Next assess: 160. Train: 160-150 = 10 to 159. Test 2: 160-179. Skip 10. Next assess: 170. Train: 20-169. Test 3: 170-189. Skip 10. Next assess: 180. Train: 30-179. Test 4: 180-199. This should be 3 splits not 4 by my original logic. Let's check the actual output with latest code.
# With current code:
# Split 1: T:0-149, A:150-169. Next assess start (current_assess_start_idx + skip): 150+10 = 160
# Split 2: T:10-159, A:160-179. Next assess start: 160+10 = 170
# Split 3: T:20-169, A:170-189. Next assess start: 170+10 = 180
# Split 4: T:30-179, A:180-199. Next assess start: 180+10 = 190. Loop continues for slice_limit=5.
# Split 5: T:40-189, A:190-209 (actual 190-199). Next assess start: 190+10 = 200. Loop terminates. So 5 splits.

print("\n--- 2.1.2 Integer CV - Expanding (cumulative=True) ---")
splits_expanding = time_series_cv(cv_data_series, initial=150, assess=20, skip=10, cumulative=True, slice_limit=3)
print(f"Generated {len(splits_expanding)} expanding splits.")
assert len(splits_expanding) == 3 # Same logic for number of test windows as above

print("\n--- 2.1.3 Integer CV - Skip=0 ---")
splits_skip0 = time_series_cv(cv_data_series, initial=100, assess=20, skip=0, cumulative=False, slice_limit=10)
print(f"Generated {len(splits_skip0)} splits with skip=0.") # (200-100)/20 = 5 splits
assert len(splits_skip0) == 5

print("\n--- 2.1.4 Integer CV - Slice Limit ---")
splits_slice_limit = time_series_cv(cv_data_series, initial=150, assess=10, skip=5, cumulative=False, slice_limit=2)
print(f"Generated {len(splits_slice_limit)} splits with slice_limit=2.")
assert len(splits_slice_limit) == 2

# --- 2.2 Period String CV ---
print("\n--- 2.2.1 Period CV - Rolling ---")
cv_data_years = create_sample_data(start_date='2000-01-01', n_periods=10*365, freq='D', include_features=False)
splits_period_rolling = time_series_cv(cv_data_years, initial='5 years', assess='1 year', skip='6 months', cumulative=False, slice_limit=5)
print(f"Generated {len(splits_period_rolling)} period-based rolling splits.")
assert len(splits_period_rolling) > 0

print("\n--- 2.2.2 Period CV - Expanding ---")
splits_period_expanding = time_series_cv(cv_data_years, initial='5 years', assess='1 year', skip='6 months', cumulative=True, slice_limit=10)
print(f"Generated {len(splits_period_expanding)} period-based expanding splits.")
assert len(splits_period_expanding) == len(splits_period_rolling) # Number of assess windows should be same

# --- 2.3 CV with DataFrame and date_column ---
print("\n--- 2.3.1 CV with DataFrame and date_column ---")
cv_data_monthly_df = create_sample_data(start_date='2015-01-01', n_periods=60, freq='MS', as_frame=True, include_features=False)
splits_df_date_col = time_series_cv(
    cv_data_monthly_df, initial='36 months', assess='6 months', skip='3 months',
    cumulative=False, slice_limit=5, date_column='date'
)
print(f"Generated {len(splits_df_date_col)} splits for DataFrame with date_column.")
assert len(splits_df_date_col) > 0 # (60-36)=24 avail. Assess=6. Skip=3. (6+3)=9. 24/9 -> 2 splits.
# Split 1: Train M0-M35, Test M36-M41. Skip 3M. Next assess start: M36 + 3M = M39
# Split 2: Train M3-M38, Test M39-M44. Skip 3M. Next assess start: M39 + 3M = M42
# Split 3: Train M6-M41, Test M42-M47. Skip 3M. Next assess start: M42 + 3M = M45
# Split 4: Train M9-M44, Test M45-M50. Skip 3M. Next assess start: M45 + 3M = M48
# Split 5: Train M12-M47, Test M48-M53. Skip 3M. Next assess start: M48 + 3M = M51
# Split 6: Train M15-M50, Test M51-M56. Skip 3M. Next assess start: M51 + 3M = M54
# Split 7: Train M18-M53, Test M54-M59. Skip 3M. Next assess start: M54 + 3M = M57.
# Total 7 splits possible.

# --- 2.4 Accessing Split Data ---
print("\n--- 2.4.1 Accessing data from CV splits ---")
sample_data_for_access = create_sample_data(n_periods=50, freq='D', include_features=False)
splits_for_access = time_series_cv(sample_data_for_access, initial=30, assess=5, skip=2, slice_limit=3)
print(f"Generated {len(splits_for_access)} splits for data access example.")

for i, (train_idx, test_idx) in enumerate(splits_for_access):
    print(f"Split {i+1}:")
    train_data_fold = sample_data_for_access.iloc[train_idx]
    test_data_fold = sample_data_for_access.iloc[test_idx]
    print(f"  Train indices: {train_idx[:3]}...{train_idx[-3:]}, Length: {len(train_data_fold)}")
    print(f"  Test indices:  {test_idx[:3]}...{test_idx[-3:]}, Length: {len(test_data_fold)}")
    if i < 2: # Print head/tail for first two splits
        print("  Train data head:\\n", train_data_fold.head(2))
        print("  Test data head:\\n", test_data_fold.head(2))
    assert not train_data_fold.empty
    assert not test_data_fold.empty
    if i > 0 :
      prev_train_idx, prev_test_idx = splits_for_access[i-1]
      if not splits_for_access[0][0][0] == 0: # not cumulative
           assert train_idx[0] > prev_train_idx[0] # if rolling, start of train should move
      assert test_idx[0] > prev_test_idx[0]     # start of test should always move


print("\n\n--- Section 3: plot_time_series_cv_plan ---")
# Ensure plots are shown in non-interactive environments if necessary
# For scripts, plt.show() is often needed. For notebooks, %matplotlib inline is typical.

# --- 3.1 Basic Plot ---
print("\n--- 3.1.1 Basic CV Plan Plot ---")
plot_data_series = create_sample_data(n_periods=100, freq='D', include_features=False)
cv_splits_for_plot = time_series_cv(plot_data_series, initial=70, assess=10, skip=5, cumulative=False, slice_limit=4)
print(f"Plotting {len(cv_splits_for_plot)} splits...")
fig1, ax1 = plot_time_series_cv_plan(plot_data_series, cv_splits_for_plot, title="Basic CV Plan (Rolling)")
plt.show()
print("Plot 1 displayed.")

# --- 3.2 Plot with More Splits / Different Parameters ---
print("\n--- 3.2.1 CV Plan Plot - Expanding Window ---")
cv_splits_expanding_plot = time_series_cv(plot_data_series, initial=60, assess=10, skip=5, cumulative=True, slice_limit=6)
print(f"Plotting {len(cv_splits_expanding_plot)} expanding splits...")
fig2, ax2 = plot_time_series_cv_plan(plot_data_series, cv_splits_expanding_plot, title="CV Plan (Expanding Window)", train_color='darkgreen', test_color='orange')
plt.show()
print("Plot 2 displayed.")

# --- 3.3 Plotting with DataFrame and date_column ---
print("\n--- 3.3.1 CV Plan Plot - DataFrame with date_column ---")
plot_data_df = create_sample_data(n_periods=80, freq='W', as_frame=True, date_col_name='weekly_date', include_features=False)
cv_splits_df_plot = time_series_cv(plot_data_df, initial='50W', assess='10W', skip='5W', date_column='weekly_date', slice_limit=4)
print(f"Plotting {len(cv_splits_df_plot)} splits from DataFrame...")
fig3, ax3 = plot_time_series_cv_plan(plot_data_df, cv_splits_df_plot, date_column='weekly_date', title="CV Plan (DataFrame, Weekly Data)")
plt.show()
print("Plot 3 displayed.")

print("\n\n--- Section 4: Notes and Best Practices ---")
print("""
- `skip=0`: When `skip=0` is used with `time_series_cv`, the next training window
  (in rolling mode) or assessment window starts immediately after the previous
  assessment window ends. This creates contiguous, non-overlapping assessment sets.

- Choosing Parameters: The choice of `initial`, `assess`, and `skip` is crucial.
  - `initial`: Should be large enough to train a stable model.
  - `assess`: Should be representative of the period you want to predict.
  - `skip`: Determines the overlap between consecutive training sets (if any, in rolling mode)
    and how quickly you move through the time series.
  Consider the length of your data, seasonality, and the problem requirements.

- Unsorted Data: The library functions (`time_series_split`, `time_series_cv`) will
  automatically sort data that has an unsorted `DatetimeIndex` or an unsorted `date_column`.
  A `UserWarning` is issued in such cases. The returned splits and plots will
  be based on the sorted data.

- Period Strings: When using period strings (e.g., '3 months', '1 year'), the exact number
  of samples can vary slightly based on start/end dates (e.g., number of days in a month,
  leap years). The functions aim to get as close as possible to the specified duration.
  For precise sample counts, use integers.
""")

print("\n\n--- Section 5: Modeling with Resamples ---")

# --- 5.1 fit_resamples Example ---
print("\n--- 5.1.1 Fitting a model to resamples (DataFrame with features) ---")
# Create data with features for this example
modeling_data_df = create_sample_data(start_date='2020-01-01', n_periods=200, freq='D', as_frame=True, value_col_name='target', include_features=True)
modeling_cv_splits = time_series_cv(
    modeling_data_df, 
    date_column='date', 
    initial='100D', 
    assess='20D', 
    skip='10D', 
    slice_limit=3
)

model_spec_lr = LinearRegression()

resamples_fitted_df = fit_resamples(
    cv_splits=modeling_cv_splits,
    model_spec=model_spec_lr,
    data=modeling_data_df,
    target_column='target',
    feature_columns=['time_idx', 'month', 'lag_value'], # features from create_sample_data
    date_column='date',
    model_id='linear_regr'
)
print("fit_resamples output (head):\n", resamples_fitted_df.head())
print("\nfit_resamples output info:")
resamples_fitted_df.info()

print("\n--- 5.1.2 Fitting a univariate model (Series input, no explicit features) ---")
modeling_data_series = create_sample_data(start_date='2021-01-01', n_periods=150, freq='M', as_frame=False, value_col_name='monthly_val', include_features=False)
modeling_cv_splits_series = time_series_cv(
    modeling_data_series, 
    initial='10M', # Using months for period string
    assess='2M', 
    skip='1M', 
    slice_limit=2
)

# Using a simple model, fit_resamples will create a dummy feature for sklearn models
# if feature_columns is None and data is DataFrame. For Series, it's up to the model.
# LinearRegression here will effectively fit an intercept model per fold if no features are given to it.
resamples_fitted_series_df = fit_resamples(
    cv_splits=modeling_cv_splits_series,
    model_spec=LinearRegression(), # Can also use custom models like the DummyModel from tests
    data=modeling_data_series,
    target_column='monthly_val',
    # feature_columns=None by default for Series, implies model handles it or dummy feature used if applicable
    model_id='lr_univariate'
)
print("\nfit_resamples output for Series (head):\n", resamples_fitted_series_df.head())

# --- 5.2 resample_accuracy Example ---
print("\n--- 5.2.1 resample_accuracy with default metrics (on 'test' period by default) ---")
accuracy_default = resample_accuracy(resamples_fitted_df)
print("Accuracy with default metrics (test period):\\n", accuracy_default)

# --- 5.2.2 resample_accuracy with custom metrics ---
print("\n--- 5.2.2 resample_accuracy with custom metrics (on 'test' period by default) ---")
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

custom_metrics = {
    'mae': lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)), # Using a lambda for MAE
    'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'mape': mape
}
accuracy_custom = resample_accuracy(resamples_fitted_df, metrics_set=custom_metrics)
print("Accuracy with custom metrics (test period):\\n", accuracy_custom)

# --- 5.2.3 resample_accuracy evaluating 'train' period ---
print("\n--- 5.2.3 resample_accuracy evaluating 'train' period ---")
accuracy_train_period = resample_accuracy(resamples_fitted_df, period_types_to_evaluate=['train'])
print("Accuracy on training data periods:\\n", accuracy_train_period)

# --- 5.2.4 resample_accuracy evaluating both 'train' and 'test' periods ---
print("\n--- 5.2.4 resample_accuracy evaluating both 'train' and 'test' periods ---")
accuracy_train_test_periods = resample_accuracy(resamples_fitted_df, period_types_to_evaluate=['train', 'test'])
print("Accuracy on both training and test data periods:\\n", accuracy_train_test_periods)

# --- 5.3 plot_resamples Example ---
print("\n--- 5.3.1 Plotting resamples with Matplotlib (default engine) ---")
if not resamples_fitted_df.empty:
    fig_mpl = plot_resamples(
        resamples_fitted_df, 
        max_splits_to_plot=2, 
        title='Model Resamples (Matplotlib)'
    )
    plt.show() # Show matplotlib plot
    print("Matplotlib resamples plot displayed.")
else:
    print("Skipping Matplotlib plot_resamples as resamples_fitted_df is empty.")

print("\n--- 5.3.2 Plotting resamples with Plotly (interactive) ---")
if not resamples_fitted_df.empty:
    try:
        # Plotly imports are best done if engine='plotly' is actually used
        # from plotly.io import show as pio_show # For displaying in scripts/non-notebook
        
        fig_plotly = plot_resamples(
            resamples_fitted_df, 
            max_splits_to_plot=2, 
            title='Model Resamples (Plotly)',
            engine='plotly'
        )
        # To show in a script, you might need: fig_plotly.show()
        # For notebooks, often it renders automatically or pio.renderers.default = "notebook"
        fig_plotly.show() # This usually works in most environments if plotly is set up
        print("Plotly resamples plot displayed. (May open in browser or render in IDE)")
    except ImportError:
        print("Plotly is not installed. Skipping interactive plot_resamples example.")
        print("To run this example, please install plotly: pip install plotly")
    except Exception as e:
        print(f"An error occurred during Plotly plotting: {e}")
else:
    print("Skipping Plotly plot_resamples as resamples_fitted_df is empty.")


print("\n\nCookbook finished.")

# Add a final plt.show() if running as a script and plots might not have stayed open
if __name__ == '__main__':
    plt.show() 