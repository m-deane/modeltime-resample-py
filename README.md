# Modeltime Resample Py

A Python package for time series cross-validation and resampling, inspired by the R `modeltime.resample` package.

[![PyPI version](https://badge.fury.io/py/modeltime-resample-py.svg)](https://badge.fury.io/py/modeltime-resample-py) <!-- Placeholder - replace if/when published -->

## Features

*   Time series splitting (`time_series_split`)
*   Time series cross-validation (`time_series_cv`) with rolling and expanding windows.
*   Support for integer counts and time-based periods (e.g., '5 years', '3 months').
*   Visualization of CV plan (`plot_time_series_cv_plan`).

## Installation

```bash
# Install from source (recommended for development)
pip install -e .

# Or install the package if published (replace with actual name if different)
# pip install modeltime_resample_py
```

**Dependencies:**

*   pandas>=1.0.0
*   numpy>=1.18.0
*   scikit-learn>=0.22.0 (Primarily for potential compatibility, not a core dependency for splitting)
*   matplotlib>=3.3.0 (For plotting)

## Usage

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import modeltime_resample_py as mtrp

# 1. Create Sample Data (Replace with your actual time series)
periods = 365 * 3 # 3 years of daily data
dates = pd.date_range(start='2020-01-01', periods=periods, freq='D')
values = np.random.randn(periods).cumsum() # Random walk
data = pd.Series(values, index=dates, name='value')

print("--- Sample Data ---")
print(data.head())

# 2. Create a Single Train/Test Split
train_data, test_data = mtrp.time_series_split(
    data,
    initial='2 years',
    assess='1 year'
)

print("\n--- Single Split ---")
print(f"Train Length: {len(train_data)}")
print(f"Test Length: {len(test_data)}")
print(f"Train Dates: {train_data.index.min()} to {train_data.index.max()}")
print(f"Test Dates: {test_data.index.min()} to {test_data.index.max()}")

# 3. Create Cross-Validation Splits (Rolling Origin)
cv_splits = mtrp.time_series_cv(
    data,
    initial='1 year',     # Start with 1 year of training data
    assess='3 months',   # Assess on the next 3 months
    skip='3 months',    # Skip 3 months between folds
    cumulative=False,    # Rolling window (training data size stays ~constant)
    slice_limit=6        # Generate up to 6 folds
)

print(f"\n--- Cross-Validation Splits (Rolling) ---")
print(f"Number of splits generated: {len(cv_splits)}")
for i, (train_idx, test_idx) in enumerate(cv_splits):
    train_slice = data.iloc[train_idx]
    test_slice = data.iloc[test_idx]
    print(
        f"Split {i+1}: "
        f"Train={train_slice.index.min().date()} to {train_slice.index.max().date()} ({len(train_slice)}), "
        f"Test={test_slice.index.min().date()} to {test_slice.index.max().date()} ({len(test_slice)})"
    )

# 4. Visualize the CV Plan
fig, ax = plt.subplots()
mtrp.plot_time_series_cv_plan(data, cv_splits, ax=ax, title="Rolling Origin CV Plan (1Y Initial, 3M Assess, 3M Skip)")
# plt.show() # Uncomment to display plot interactively
plt.savefig('cv_plan_example.png') # Save the plot
print("\nSaved CV plan visualization to cv_plan_example.png")

```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (add `LICENSE` file). 