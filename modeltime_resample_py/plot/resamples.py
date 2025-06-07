"""Plotting functions for model resamples."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Union, Tuple, Optional, Any

def plot_resamples(
    resamples_df: pd.DataFrame, # Output from fit_resamples (new long format)
    max_splits_to_plot: int = 5,
    title: Optional[str] = "Resamples: Actuals, Fitted Values, and Predictions",
    actual_color: str = "darkblue",
    actual_linestyle: str = "--",
    actual_linewidth: float = 2.5,
    fitted_color: str = "orange",
    fitted_linestyle: str = "-",
    fitted_linewidth: float = 2,
    prediction_color: str = "red",
    prediction_linestyle: str = "-",
    prediction_linewidth: float = 2,
    show_legend: bool = True,
    figure_size: Tuple[int, int] = (12, 6),
    engine: str = 'matplotlib' # 'matplotlib' or 'plotly'
) -> Union[plt.Figure, Any]: # Any for plotly.graph_objects.Figure to avoid direct import
    """
    Plots actuals, fitted values (on train set), and predictions (on test set) 
    for a selection of resample splits from the long-format output of fit_resamples.
    Supports both Matplotlib (default) and Plotly interactive plots.

    Args:
        resamples_df: DataFrame output from fit_resamples (new long format).
                      Expected to be indexed by ['date', 'slice_id', 'model_id'] and contain
                      columns: 'actuals', 'fitted_values', 'predictions', 'period_type'.
        max_splits_to_plot: Maximum number of unique (slice_id, model_id) combinations to display.
        title: Overall title for the plot.
        actual_color: Color for the actual values line.
        actual_linestyle: Linestyle for the actual values line (matplotlib style).
        actual_linewidth: Linewidth for the actual values line.
        fitted_color: Color for the fitted values line on the training period.
        fitted_linestyle: Linestyle for the fitted values line (matplotlib style).
        fitted_linewidth: Linewidth for the fitted values line.
        prediction_color: Color for the predicted values line on the test period.
        prediction_linestyle: Linestyle for the predicted values line (matplotlib style).
        prediction_linewidth: Linewidth for the predicted values line.
        show_legend: Whether to display the legend.
        figure_size: Tuple specifying the figure size (width, height) for Matplotlib.
                     Plotly uses its own sizing defaults but can be adjusted post-creation.
        engine: Plotting engine to use, either 'matplotlib' (default) or 'plotly'.

    Returns:
        A matplotlib.figure.Figure if engine='matplotlib',
        or a plotly.graph_objects.Figure if engine='plotly'.

    Raises:
        TypeError: If `resamples_df` is not a pandas DataFrame.
        ValueError: If `resamples_df` is not indexed correctly.
        ValueError: If `resamples_df` is missing required columns.
        ValueError: If an unsupported `engine` is specified.
        ImportError: If `engine='plotly'` and Plotly is not installed.
    """
    if not isinstance(resamples_df, pd.DataFrame):
        raise TypeError("resamples_df must be a pandas DataFrame.")

    expected_index_names = ['date', 'slice_id', 'model_id']
    if list(resamples_df.index.names) != expected_index_names:
        raise ValueError(f"resamples_df must be indexed by {expected_index_names}.")

    required_cols = ['actuals', 'fitted_values', 'predictions', 'period_type']
    if not all(col in resamples_df.columns for col in required_cols):
        raise ValueError(f"resamples_df must contain columns: {required_cols}")

    if engine not in ['matplotlib', 'plotly']:
        raise ValueError(f"Unsupported engine: '{engine}'. Choose 'matplotlib' or 'plotly'.")

    if resamples_df.empty:
        warnings.warn("Input resamples_df is empty. Cannot generate plot.", UserWarning)
        if engine == 'matplotlib':
            return plt.figure(figsize=figure_size)
        elif engine == 'plotly':
            try:
                import plotly.graph_objects as go
                return go.Figure()
            except ImportError:
                raise ImportError("Plotly is not installed. Please install it to use engine='plotly'.")

    unique_groups_to_plot = resamples_df.index.droplevel(0).unique()
    
    if not unique_groups_to_plot.empty:
        unique_groups_to_plot = unique_groups_to_plot[:max_splits_to_plot]
        n_plots = len(unique_groups_to_plot)
    else:
        n_plots = 0

    if n_plots == 0:
        warnings.warn("No (slice_id, model_id) groups found to plot based on input data and max_splits_to_plot.", UserWarning)
        if engine == 'matplotlib':
            return plt.figure(figsize=figure_size)
        elif engine == 'plotly':
            try:
                import plotly.graph_objects as go
                return go.Figure()
            except ImportError:
                raise ImportError("Plotly is not installed. Please install it to use engine='plotly'.")

    if engine == 'matplotlib':
        fig, axes = plt.subplots(n_plots, 1, figsize=(figure_size[0], figure_size[1] * n_plots), squeeze=False)
        axes = axes.flatten()
        fig.suptitle(title, fontsize=16)

        for i, (slice_id_val, model_id_val) in enumerate(unique_groups_to_plot):
            ax = axes[i]
            group_data = resamples_df.xs((slice_id_val, model_id_val), level=('slice_id', 'model_id'))
            dates = group_data.index
            actuals = group_data['actuals']
            fitted_values = group_data['fitted_values']
            predictions = group_data['predictions']

            ax.plot(dates, actuals, color=actual_color, linestyle=actual_linestyle, 
                    linewidth=actual_linewidth, label='Actuals' if i == 0 else '_nolegend_')
            ax.plot(dates, fitted_values, color=fitted_color, linestyle=fitted_linestyle, 
                    linewidth=fitted_linewidth, label='Fitted' if i == 0 else '_nolegend_')
            ax.plot(dates, predictions, color=prediction_color, linestyle=prediction_linestyle, 
                    linewidth=prediction_linewidth, label='Predictions' if i == 0 else '_nolegend_')

            ax.set_title(f'Slice: {slice_id_val}, Model: {model_id_val}', fontsize=12)
            ax.set_ylabel('Value', fontsize=10)
            if i == n_plots - 1:
                ax.set_xlabel('Date', fontsize=10)
            ax.grid(True, alpha=0.3)

        if show_legend:
            axes[0].legend(loc='upper right', fontsize=10)

        plt.tight_layout()
        return fig

    elif engine == 'plotly':
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly is not installed. Please install it to use engine='plotly'.")

        def mpl_to_plotly_dash(style):
            """Convert matplotlib linestyle to plotly dash style."""
            dash_map = {
                '-': 'solid',
                '--': 'dash',
                '-.': 'dashdot',
                ':': 'dot'
            }
            return dash_map.get(style, 'solid')

        fig = make_subplots(rows=n_plots, cols=1, 
                            subplot_titles=[f'Slice: {s}, Model: {m}' for s, m in unique_groups_to_plot],
                            vertical_spacing=0.1)

        show_legend_for_first = True
        
        for i, (slice_id_val, model_id_val) in enumerate(unique_groups_to_plot):
            group_data = resamples_df.xs((slice_id_val, model_id_val), level=('slice_id', 'model_id'))
            dates = group_data.index
            actuals = group_data['actuals']
            fitted_values = group_data['fitted_values']
            predictions = group_data['predictions']

            row_num = i + 1

            fig.add_trace(
                go.Scatter(x=dates, y=actuals, name='Actuals',
                          line=dict(color=actual_color, dash=mpl_to_plotly_dash(actual_linestyle), 
                                    width=actual_linewidth),
                          showlegend=show_legend_for_first),
                row=row_num, col=1
            )

            fig.add_trace(
                go.Scatter(x=dates, y=fitted_values, name='Fitted',
                          line=dict(color=fitted_color, dash=mpl_to_plotly_dash(fitted_linestyle), 
                                    width=fitted_linewidth),
                          showlegend=show_legend_for_first),
                row=row_num, col=1
            )

            fig.add_trace(
                go.Scatter(x=dates, y=predictions, name='Predictions',
                          line=dict(color=prediction_color, dash=mpl_to_plotly_dash(prediction_linestyle), 
                                    width=prediction_linewidth),
                          showlegend=show_legend_for_first),
                row=row_num, col=1
            )

            show_legend_for_first = False

            fig.update_xaxes(title_text="Date" if i == n_plots - 1 else "", row=row_num, col=1)
            fig.update_yaxes(title_text="Value", row=row_num, col=1)

        fig.update_layout(
            title_text=title,
            title_font_size=16,
            height=300 * n_plots,
            showlegend=show_legend
        )

        return fig 