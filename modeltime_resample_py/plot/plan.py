import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings # Import warnings
from typing import Union, List, Tuple, Optional

def plot_time_series_cv_plan(
    data: Union[pd.DataFrame, pd.Series],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    date_column: Optional[str] = None,
    title: str = "Time Series CV Plan",
    x_lab: str = "Date",
    y_lab: str = "Split ID",
    train_color: str = '#2c3e50', # Dark blue
    test_color: str = '#e74c3c',  # Red
    figsize: Tuple[int, int] = (10, 5),
    base_size: int = 11,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualizes the cross-validation plan created by `time_series_cv`.

    Args:
        data: The original pandas DataFrame or Series used for `time_series_cv`.
        splits: The list of (train_indices, test_indices) tuples returned by `time_series_cv`.
        date_column: The name of the date column if `data` is a DataFrame without a DatetimeIndex.
        title: The title for the plot.
        x_lab: The label for the x-axis (time).
        y_lab: The label for the y-axis (splits).
        train_color: Color for the training data segments.
        test_color: Color for the testing (assessment) data segments.
        figsize: Figure size tuple (width, height) in inches.
        base_size: Base font size for plot elements.
        ax: An existing matplotlib Axes object to plot on. If None, a new figure and axes are created.

    Returns:
        A matplotlib Axes object containing the plot.

    Raises:
        ValueError: If `splits` is empty or data/indices are incompatible.
        TypeError: If `data` is not a pandas DataFrame or Series.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Input 'data' must be a pandas DataFrame or Series.")
    if not splits:
        raise ValueError("Input 'splits' list cannot be empty.")

    # --- Data and Index Preparation ---
    original_index_used = True # Flag to track if original index is used
    if date_column:
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
             raise ValueError(f"Date column '{date_column}' must be of datetime type.")
        time_index = pd.DatetimeIndex(data[date_column])
        if not time_index.is_monotonic_increasing:
             warnings.warn("Date column is not monotonically increasing. Plotting based on sorted data order.", UserWarning)
             sorted_indices = np.argsort(time_index)
             time_index = time_index[sorted_indices]
             original_index_used = False
    elif isinstance(data.index, pd.DatetimeIndex):
        time_index = data.index
        if not time_index.is_monotonic_increasing:
             warnings.warn("DatetimeIndex is not monotonically increasing. Plotting based on sorted data order.", UserWarning)
             data_sorted = data.sort_index()
             time_index = data_sorted.index
             original_index_used = False
    else:
        raise ValueError("Data must have a DatetimeIndex or a valid 'date_column' specified.")

    n_splits = len(splits)
    n_total = len(time_index)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    plt.rcParams.update({'font.size': base_size})

    for i, (train_idx, test_idx) in enumerate(splits):
        split_id = i + 1 # 1-based indexing for plot

        if train_idx.max() >= n_total or test_idx.max() >= n_total:
             warnings.warn(f"Split {split_id} indices exceed data length. Skipping this split in plot.", UserWarning)
             continue
        if len(train_idx) == 0 or len(test_idx) == 0:
             warnings.warn(f"Split {split_id} has empty train or test set. Skipping this split in plot.", UserWarning)
             continue

        # Get date ranges from indices
        train_start_date = time_index[train_idx[0]]
        train_end_date = time_index[train_idx[-1]]
        test_start_date = time_index[test_idx[0]]
        test_end_date = time_index[test_idx[-1]]

        # Plot horizontal lines for train and test ranges
        # Use plot instead of hlines for easier legend handling
        ax.plot([train_start_date, train_end_date], [split_id, split_id],
                color=train_color, linewidth=base_size * 0.5, solid_capstyle='butt',
                label='Train' if i == 0 else "_nolegend_") # Only label first instance
        ax.plot([test_start_date, test_end_date], [split_id, split_id],
                color=test_color, linewidth=base_size * 0.5, solid_capstyle='butt',
                label='Test' if i == 0 else "_nolegend_")

        # Add points at start/end for clarity (optional)
        # ax.scatter([train_start_date, train_end_date], [split_id, split_id], color=train_color, s=base_size * 2)
        # ax.scatter([test_start_date, test_end_date], [split_id, split_id], color=test_color, s=base_size * 2)

    # --- Formatting ---
    ax.set_title(title, fontsize=base_size * 1.2)
    ax.set_xlabel(x_lab, fontsize=base_size)
    ax.set_ylabel(y_lab, fontsize=base_size)

    # Format y-axis to show integer split IDs
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, min_n_ticks=1))
    ax.invert_yaxis() # Show Split 1 at the top
    ax.tick_params(axis='both', which='major', labelsize=base_size * 0.9)

    # Format x-axis for dates
    fig.autofmt_xdate()

    # Add legend
    ax.legend(fontsize=base_size * 0.9)

    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    return ax 