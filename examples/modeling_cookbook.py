import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import from your library
from modeltime_resample_py import (
    time_series_cv,
    fit_resamples,
    resample_accuracy,
    plot_resamples
)
import warnings
import inspect

# Suppress FutureWarnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # Suppress some of our own warnings for cleaner cookbook output

print("Cookbook for modeltime_resample_py.modeling functions\n")

# --- 1. Helper Function to Create Sample Data & Features ---
def create_featured_ts_data(
    start_date='2020-01-01',
    n_periods=100,
    freq='D',
    as_frame=True,
    date_col_name='date',
    target_col_name='value'
):
    dates = pd.date_range(start=start_date, periods=n_periods, freq=freq)
    # Target variable (e.g., sales)
    base_trend = np.arange(n_periods)
    seasonality = 10 * np.sin(2 * np.pi * base_trend / (30 if freq == 'D' else 12)) # Monthly seasonality
    noise = np.random.randn(n_periods) * 5
    values = base_trend + seasonality + noise + 50
    values = np.maximum(0, values) # Ensure non-negative values like sales

    if not as_frame:
        return pd.Series(values, index=dates, name=target_col_name)

    df = pd.DataFrame({date_col_name: dates, target_col_name: values})
    
    # Feature engineering
    df['time_idx'] = np.arange(n_periods) # Time index
    df['month'] = df[date_col_name].dt.month
    df['lag_1'] = df[target_col_name].shift(1).fillna(method='bfill') # Lagged target

    # Create some dummy external regressors
    df['promo_A'] = np.random.choice([0,1], size=n_periods, p=[0.8, 0.2])
    df['temp'] = 20 + 5 * np.sin(2 * np.pi * base_trend / (365 if freq == 'D' else 12*3)) + np.random.randn(n_periods)*2
    
    return df

# --- 2. Generate Sample Data and CV Splits ---
print("--- Generating Sample Data & CV Splits ---")
data_df = create_featured_ts_data(n_periods=200, freq='D')
target_variable = 'value'
date_column_name = 'date'

# For a model with features
feature_names = ['time_idx', 'month', 'lag_1', 'promo_A', 'temp']

# Ensure no NaNs in features or target for the portion used in modeling
data_df_cleaned = data_df.dropna(subset=[target_variable] + feature_names).reset_index(drop=True)

print(f"Cleaned data shape: {data_df_cleaned.shape}")
print("Cleaned data head:\n", data_df_cleaned.head())

cv_splits_list = time_series_cv(
    data_df_cleaned,
    initial='90D',    # Initial training period
    assess='30D',     # Assessment/test period
    skip='30D',       # Skip period
    cumulative=False,
    slice_limit=4,
    date_column=date_column_name
)
print(f"\nGenerated {len(cv_splits_list)} CV splits.")

# --- 3. Define Model and Fit Resamples ---
print("\n--- Fitting Resamples ---")
# Using a simple Linear Regression model from scikit-learn
# model_specification = LinearRegression(fit_intercept=True) # fit_intercept is True by default

# Test with a model that might not have fit_intercept
class SimpleModel:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._estimator_type = "regressor" # For sklearn.base.is_regressor

    def fit(self, X, y):
        # Simple model: y = X.mean(axis=1) * coef + intercept
        # This is just a placeholder to ensure fit/predict work.
        # For LinearRegression, X would be 2D.
        if X.empty or X.shape[1] == 0: # Handle empty X for univariate tests
            self.coef_ = np.array([1.0]) # Dummy coef
            self.intercept_ = np.mean(y) if len(y) > 0 else 0
        else:
            # Treat as if X has one feature for simplicity if many columns
            X_mean_feature = X.mean(axis=1) if X.shape[1] > 1 else X.iloc[:,0]
            # Simple OLS for one feature for demonstration
            X_m = np.vstack([X_mean_feature, np.ones(len(X_mean_feature))]).T
            try:
                self.coef_, self.intercept_ = np.linalg.lstsq(X_m, y, rcond=None)[0]
            except Exception: # Fallback if lstsq fails
                self.coef_ = np.array([1.0]) 
                self.intercept_ = np.mean(y) if len(y) > 0 else 0
        return self

    def predict(self, X):
        if X.empty or X.shape[1] == 0: # Handle empty X
            return np.full(X.shape[0], self.intercept_ if self.intercept_ is not None else 0)
        
        X_mean_feature = X.mean(axis=1) if X.shape[1] > 1 else X.iloc[:,0]
        return (X_mean_feature * (self.coef_[0] if isinstance(self.coef_, np.ndarray) else self.coef_)) + \
               (self.intercept_ if self.intercept_ is not None else 0)


model_specification = LinearRegression()
# model_specification = SimpleModel() # Example with a custom simple model

# Example 1: Using feature_columns
print("\nExample 1: fit_resamples with feature_columns")
resamples_results_df = fit_resamples(
    cv_splits=cv_splits_list,
    model_spec=model_specification,
    data=data_df_cleaned,
    target_column=target_variable,
    feature_columns=feature_names, # Pass the list of feature names
    date_column=date_column_name,
    model_id="linear_regression_with_features"
)

print(f"Shape of resamples_results_df: {resamples_results_df.shape}")
print("Resamples results head:\n", resamples_results_df.head(2))
if not resamples_results_df.empty:
    print("Columns:", resamples_results_df.columns)
    print("Data types:\n", resamples_results_df.dtypes)
    # Inspecting one row's series
    print("First row actuals (type):", type(resamples_results_df['actuals'].iloc[0]))

# Example 2: Univariate case (model must handle its own lags or use only y)
# For LinearRegression, this will likely perform poorly without explicit lag features,
# as X_train becomes an empty DataFrame if feature_columns=None and data is a DataFrame.
# The current fit_resamples passes an empty DataFrame for X in this univariate context,
# which LinearRegression can handle (it will fit an intercept-only model effectively if fit_intercept=True).

print("\nExample 2: fit_resamples for a 'univariate' setup (LinearRegression will be intercept-only)")
# We need to ensure X is not empty for LinearRegression if no features are given.
# The current fit_resamples provides an empty X_train if feature_columns=None.
# For this example, let's use the SimpleModel or accept intercept-only LR.

# To make LinearRegression work in a "univariate" way (predicting mean),
# fit_resamples currently passes an empty DataFrame.
# If fit_intercept is True, it predicts the mean of y_train.
univariate_model_spec = LinearRegression(fit_intercept=True)

resamples_univariate_df = fit_resamples(
    cv_splits=cv_splits_list,
    model_spec=univariate_model_spec,
    data=data_df_cleaned[[date_column_name, target_variable]], # Pass only target and date
    target_column=target_variable,
    feature_columns=None, # Explicitly None
    date_column=date_column_name,
    model_id="intercept_only_model"
)
print(f"Shape of univariate_resamples_df: {resamples_univariate_df.shape}")
print("Univariate resamples results head:\n", resamples_univariate_df.head(2))


# --- 4. Calculate Resample Accuracy ---
print("\n--- Calculating Resample Accuracy ---")

# Using default metrics (MAE, RMSE)
print("\nUsing default metrics:")
accuracy_default_df = resample_accuracy(resamples_results_df)
print(accuracy_default_df)

# Using custom metrics
custom_metrics = {
    "mape": lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100,
    "mse": mean_squared_error
}
print("\nUsing custom metrics:")
accuracy_custom_df = resample_accuracy(
    resamples_results_df,
    metrics_set=custom_metrics
)
print(accuracy_custom_df)

# --- 5. Plot Resamples ---
print("\n--- Plotting Resamples ---")
if not resamples_results_df.empty:
    try:
        print(f"Inspecting plot_resamples: {inspect.signature(plot_resamples)}")
        fig = plot_resamples(
            resamples_results_df,
            max_splits_to_plot=3, # Plot first 3 splits
            title="Model Resamples: Actuals vs. Predictions (Features Model)",
            engine='plotly'
        )
        plt.show(block=False) # Show the plot (use block=True if it closes immediately)
        print("Resamples plot generated for feature model.")
    except Exception as e:
        print(f"Could not generate plot_resamples for feature model: {e}")

if not resamples_univariate_df.empty:
    try:
        print(f"Inspecting plot_resamples (univariate): {inspect.signature(plot_resamples)}")
        fig_uni = plot_resamples(
            resamples_univariate_df,
            engine='plotly',
            show_legend=True,
            max_splits_to_plot=2,
            title="Model Resamples: Actuals vs. Predictions (Intercept-Only Model)"
        )
        plt.show(block=False)
        print("Resamples plot generated for intercept-only model.")
    except Exception as e:
        print(f"Could not generate plot_resamples for intercept-only model: {e}")
else:
    print("Skipping plot for univariate model as no results were generated.")


print("\n--- Modeling Cookbook Finished ---")
# Keep plots open if running as a script
if __name__ == '__main__':
    plt.show() 