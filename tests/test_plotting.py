import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modeltime_resample_py.core.splits import time_series_cv
from modeltime_resample_py.plot.plan import plot_time_series_cv_plan

# Helper function from test_splits.py (could be moved to a conftest.py)
def create_ts_data(start='2010-01-01', periods=100, freq='D', name='value'):
    dates = pd.date_range(start=start, periods=periods, freq=freq)
    values = np.arange(periods)
    return pd.Series(values, index=dates, name=name)

def create_ts_dataframe(start='2010-01-01', periods=100, freq='D'):
    dates = pd.date_range(start=start, periods=periods, freq=freq)
    values = np.arange(periods)
    values2 = np.arange(periods) * 2
    return pd.DataFrame({'date': dates, 'value': values, 'value2': values2})

@pytest.fixture(scope="module", autouse=True)
def setup_matplotlib_backend():
    """Set matplotlib backend to non-interactive for testing."""
    initial_backend = plt.get_backend()
    plt.switch_backend('Agg') # Use non-interactive backend
    yield
    plt.switch_backend(initial_backend) # Restore backend
    plt.close('all') # Close any figures created during tests

# Basic test to ensure the plot function runs without errors
def test_plot_time_series_cv_plan_runs():
    data = create_ts_data(periods=50)
    splits = time_series_cv(data, initial=30, assess=5, skip=5, slice_limit=3)
    try:
        ax = plot_time_series_cv_plan(data, splits)
        assert isinstance(ax, plt.Axes)
        assert ax.get_title() == "Time Series CV Plan"
        assert ax.get_xlabel() == "Date"
        assert ax.get_ylabel() == "Split ID"
        # Check if legend items were created (at least Train and Test)
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert 'Train' in legend_texts
        assert 'Test' in legend_texts
    except Exception as e:
        pytest.fail(f"plot_time_series_cv_plan raised an exception: {e}")
    finally:
        plt.close(ax.figure) # Close the figure after test

# Test with DataFrame and date_column
def test_plot_time_series_cv_plan_dataframe():
    data_df = create_ts_dataframe(periods=60)
    splits = time_series_cv(data_df, initial=40, assess=5, skip=3, slice_limit=4, date_column='date')
    try:
        ax = plot_time_series_cv_plan(data_df, splits, date_column='date', title="Test Plot")
        assert isinstance(ax, plt.Axes)
        assert ax.get_title() == "Test Plot"
    except Exception as e:
        pytest.fail(f"plot_time_series_cv_plan raised an exception with DataFrame: {e}")
    finally:
        plt.close(ax.figure)

# Test with unsorted data (should plot based on sorted order)
def test_plot_time_series_cv_plan_unsorted():
    dates = pd.to_datetime(['2020-01-10', '2020-01-02', '2020-01-05', '2020-01-01', '2020-01-08'])
    data = pd.Series(range(5), index=dates)
    # Expect warning from time_series_cv for sorting
    with pytest.warns(UserWarning, match='DatetimeIndex is not monotonically increasing. Sorting data by index.'):
         splits = time_series_cv(data, initial=3, assess=1, skip=1, cumulative=False, slice_limit=2)

    ax = None # Initialize ax to ensure it exists for finally block
    try:
        # Expect warning from plot function for plotting sorted data
        with pytest.warns(UserWarning, match='DatetimeIndex is not monotonically increasing. Plotting based on sorted data order.'):
             ax = plot_time_series_cv_plan(data, splits)
        assert isinstance(ax, plt.Axes)
        # Check x-axis limits reflect sorted data
        x_lim = ax.get_xlim()
        # Convert numeric xlim back to datetime approx using recommended method
        x_lim_converted = ax.xaxis.convert_units(x_lim)
        # Handle potential timezone differences if necessary (matplotlib might use UTC)
        x_lim_dates = pd.to_datetime(x_lim_converted, unit='d', origin='unix').tz_localize(None) # Basic conversion assuming days since epoch

        assert x_lim_dates[0] < pd.Timestamp('2020-01-02') # Should start around 2020-01-01
        assert x_lim_dates[1] > pd.Timestamp('2020-01-09') # Should end around 2020-01-10
    except Exception as e:
        pytest.fail(f"plot_time_series_cv_plan raised an exception with unsorted data: {e}")
    finally:
        if ax is not None:
             plt.close(ax.figure)

# Test with empty splits list (should raise ValueError)
def test_plot_time_series_cv_plan_empty_splits():
    data = create_ts_data(periods=10)
    with pytest.raises(ValueError, match="Input 'splits' list cannot be empty."):
        plot_time_series_cv_plan(data, [])

# Test passing an existing axes object
def test_plot_time_series_cv_plan_existing_ax():
    data = create_ts_data(periods=50)
    splits = time_series_cv(data, initial=30, assess=5, skip=5, slice_limit=3)
    fig, ax_existing = plt.subplots()
    try:
        ax = plot_time_series_cv_plan(data, splits, ax=ax_existing)
        assert ax is ax_existing # Should return the same axes object
    except Exception as e:
        pytest.fail(f"plot_time_series_cv_plan raised an exception with existing ax: {e}")
    finally:
        plt.close(fig) # Close the figure 