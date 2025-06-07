import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Attempt to import plot_resamples from the user's library
try:
    from modeltime_resample_py.modeling import plot_resamples
except ImportError:
    print("WARNING: plot_resamples function not found. Plotting will be basic.")
    # Define a dummy plot_resamples if the import fails, so the app can run
    def plot_resamples(resamples_df, title="", engine='plotly', **kwargs):
        if engine == 'plotly':
            fig = go.Figure()
            fig.update_layout(title_text=title if title else "Plot Placeholder (plot_resamples not found)")
            if not resamples_df.empty:
                # Basic plot of actuals if available
                for (slice_id, model_id), group in resamples_df.groupby(level=['slice_id', 'model_id']):
                    fig.add_trace(go.Scatter(x=group.index.get_level_values('date'), y=group['actuals'], name=f"Actuals {slice_id}-{model_id}"))
            return fig
        else: # matplotlib
            # Placeholder for matplotlib if needed later, or raise error
            raise NotImplementedError("Matplotlib placeholder for dummy plot_resamples.")


# --- Helper Functions ---
def generate_sample_resamples_df(num_days=100, num_slices=3, num_models=2):
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
    data_frames = []

    for slice_idx in range(num_slices):
        for model_idx in range(num_models):
            model_id = f"model_{model_idx + 1}"
            actuals = np.random.rand(num_days) * 100 + (slice_idx + model_idx) * 10
            
            # Create period_type: roughly half train, half test
            period_type = ['train'] * (num_days // 2) + ['test'] * (num_days - num_days // 2)
            np.random.shuffle(period_type)

            fitted_values = np.where(np.array(period_type) == 'train', actuals * (0.9 + np.random.rand(num_days) * 0.2 - 0.1), np.nan)
            predictions = np.where(np.array(period_type) == 'test', actuals * (0.8 + np.random.rand(num_days) * 0.4 - 0.2), np.nan)
            
            df = pd.DataFrame({
                'date': dates,
                'slice_id': slice_idx,
                'model_id': model_id,
                'actuals': actuals,
                'fitted_values': fitted_values,
                'predictions': predictions,
                'period_type': period_type
            })
            data_frames.append(df)

    final_df = pd.concat(data_frames)
    final_df = final_df.set_index(['date', 'slice_id', 'model_id'])
    return final_df

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0: # Avoid division by zero if all true values are zero
        return np.nan
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# --- Global Data ---
# For development, generate sample data. In production, you might load this.
resamples_df_global = generate_sample_resamples_df()
# Ensure the index is sorted for proper slicing and plotting
resamples_df_global = resamples_df_global.sort_index()

unique_dates = resamples_df_global.index.get_level_values('date').unique()
min_date = unique_dates.min()
max_date = unique_dates.max()

# Extract unique (slice_id, model_id) combinations for the dropdown
unique_groups = resamples_df_global.index.droplevel('date').unique().tolist()
split_model_options = [{'label': f"Slice {s}, Model {m}", 'value': f"{s}_{m}"} for s, m in unique_groups]
split_model_options.insert(0, {'label': 'All Splits - Aggregated View', 'value': 'all_aggregated'})


# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# --- App Layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Interactive Resample Analysis"), width=12), className="mb-4 mt-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Controls"),
                dbc.CardBody([
                    dbc.Label("Select Date Range:"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        min_date_allowed=min_date.date(),
                        max_date_allowed=max_date.date(),
                        initial_visible_month=min_date.date(),
                        start_date=min_date.date(),
                        end_date=max_date.date(),
                        className="mb-3"
                    ),
                    
                    dbc.Label("Select Split/Model:"),
                    dcc.Dropdown(
                        id='split-model-selector',
                        options=split_model_options,
                        value=split_model_options[0]['value'] if split_model_options else None, # Default to aggregated or first
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    dbc.Label("Select Performance Metrics:"),
                    dcc.Dropdown(
                        id='metric-selector',
                        options=[
                            {'label': 'MAE (Mean Absolute Error)', 'value': 'mae'},
                            {'label': 'RMSE (Root Mean Squared Error)', 'value': 'rmse'},
                            {'label': 'MAPE (Mean Absolute Percentage Error)', 'value': 'mape'}
                        ],
                        value=['mae', 'rmse'], # Default selected metrics
                        multi=True,
                        className="mb-3"
                    ),
                    
                    html.Button(
                        'Update View / Calculate Metrics', 
                        id='update-button', 
                        n_clicks=0, 
                        className="btn btn-primary w-100" # Bootstrap primary button, full width
                    )
                ])
            ])
        ], md=4), # Control panel column
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Resample Plot"),
                dbc.CardBody([
                    dcc.Graph(id='resample-plot', figure=go.Figure()) # Initial empty figure
                ])
            ]),
            dbc.Card([
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='metrics-table',
                        columns=[], # To be populated by callback
                        data=[],     # To be populated by callback
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px'},
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        }
                    )
                ])
            ], className="mt-4") # Metrics table card
        ], md=8) # Plot and metrics column
    ])
], fluid=True)


# --- Callbacks ---
@app.callback(
    [Output('resample-plot', 'figure'),
     Output('metrics-table', 'columns'),
     Output('metrics-table', 'data')],
    [Input('update-button', 'n_clicks')],
    [State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('split-model-selector', 'value'),
     State('metric-selector', 'value')]
)
def update_view_and_metrics(n_clicks, start_date_str, end_date_str, selected_split_model, selected_metrics):
    if n_clicks == 0 or not selected_metrics:
        empty_fig = go.Figure()
        empty_fig.update_layout(title_text="Select options and click 'Update' to view plot and metrics.")
        return empty_fig, [], []

    # Convert dates
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    # Filter data based on date range
    df_filtered_date_range = resamples_df_global[
        (resamples_df_global.index.get_level_values('date') >= start_date) &
        (resamples_df_global.index.get_level_values('date') <= end_date)
    ]

    if df_filtered_date_range.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title_text="No data available for the selected date range.")
        return empty_fig, [], []

    metrics_results = []
    figure_to_display = go.Figure()

    # Define metric functions
    metric_functions = {
        'mae': mean_absolute_error,
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': calculate_mape
    }

    groups_to_process = []
    plot_title = "Resample Plot"

    if selected_split_model == 'all_aggregated':
        groups_to_process = [(group_name, df_filtered_date_range.xs(group_name, level=('slice_id', 'model_id'), drop_level=False)) 
                             for group_name in df_filtered_date_range.index.droplevel('date').unique()]
        plot_title = f"Aggregated View: {start_date_str} to {end_date_str}"
        # For aggregated view, we might call plot_resamples on the full (but date-filtered) df
        # Or, for simplicity now, let's plot all underlying series in the selected date range
        try:
            # Use the imported plot_resamples for the main plot if 'all_aggregated'
            # This requires careful handling of how plot_resamples uses the data,
            # especially if it expects the full split data vs. date-filtered data.
            # For now, a simpler plot or adapting plot_resamples might be needed.
            # Let's use the original plot_resamples if available, passing only the date-filtered data.
            # This might not be ideal if plot_resamples expects full splits.
            # A more robust way would be to iterate and build the plot with plotly.
            
            # Create a temporary df for plotting all selected groups within the date range
            plot_df_all_selected = pd.concat([group_data for _, group_data in groups_to_process])
            if not plot_df_all_selected.empty:
                 figure_to_display = plot_resamples(plot_df_all_selected.reset_index().set_index(['date', 'slice_id', 'model_id']), title=plot_title, engine='plotly', max_splits_to_plot=len(groups_to_process))

        except Exception as e:
            print(f"Error using imported plot_resamples for aggregated view: {e}")
            figure_to_display.update_layout(title_text=f"Plot Error (Aggregated): {e}")


    else: # Individual split/model selected
        # Correctly parse slice_id and model_id
        # Model ID can contain underscores, so we split only once.
        parts = selected_split_model.split('_', 1)
        slice_id_str = parts[0]
        model_id_str = parts[1] if len(parts) > 1 else '' # model_id is the rest of the string
        
        slice_id = int(slice_id_str)
        # model_id_str is now correctly assigned the full model ID e.g. "model_1"
        
        try:
            # Use .xs to select the specific group, then filter by date.
            # df_filtered_date_range already has the date filter.
            # We need to select the specific group from this date-filtered df.
            
            # Check if group exists after date filtering
            if (slice_id, model_id_str) in df_filtered_date_range.index.droplevel('date').unique():
                group_data = df_filtered_date_range.xs((slice_id, model_id_str), level=('slice_id', 'model_id'), drop_level=False)
                groups_to_process.append(((slice_id, model_id_str), group_data))
                plot_title = f"Slice {slice_id}, Model {model_id_str}: {start_date_str} to {end_date_str}"
                 # Plot only the selected group
                if not group_data.empty:
                    figure_to_display = plot_resamples(group_data.reset_index().set_index(['date', 'slice_id', 'model_id']), title=plot_title, engine='plotly', max_splits_to_plot=1)
            else:
                 figure_to_display.update_layout(title_text=f"No data for Slice {slice_id}, Model {model_id_str} in selected date range.")

        except KeyError:
             figure_to_display.update_layout(title_text=f"Data not found for Slice {slice_id}, Model {model_id_str}")
        except Exception as e:
            print(f"Error using imported plot_resamples for single view: {e}")
            figure_to_display.update_layout(title_text=f"Plot Error (Single): {e}")


    # Calculate metrics for each group
    aggregated_metrics_data = {metric: {'train': [], 'test': []} for metric in selected_metrics}

    for (current_slice_id, current_model_id), group_df in groups_to_process:
        if group_df.empty:
            continue

        # Train period metrics
        train_data = group_df[group_df['period_type'] == 'train']
        if not train_data.empty and 'fitted_values' in train_data.columns:
            actuals_train = train_data['actuals'].dropna()
            fitted_vals = train_data['fitted_values'].dropna()
            # Align series
            common_index_train = actuals_train.index.intersection(fitted_vals.index)
            actuals_train = actuals_train.loc[common_index_train]
            fitted_vals = fitted_vals.loc[common_index_train]

            if not actuals_train.empty:
                for metric_name in selected_metrics:
                    if metric_name in metric_functions:
                        try:
                            value = metric_functions[metric_name](actuals_train, fitted_vals)
                            metrics_results.append({
                                'Slice ID': current_slice_id, 'Model ID': current_model_id,
                                'Period Type': 'Train (in range)', 'Metric': metric_name.upper(), 'Value': f"{value:.3f}"
                            })
                            if selected_split_model == 'all_aggregated':
                                aggregated_metrics_data[metric_name]['train'].append(value)
                        except Exception as e:
                            print(f"Error calculating train metric {metric_name} for {current_slice_id}-{current_model_id}: {e}")
                            metrics_results.append({
                                'Slice ID': current_slice_id, 'Model ID': current_model_id,
                                'Period Type': 'Train (in range)', 'Metric': metric_name.upper(), 'Value': "Error"
                            })


        # Test period metrics
        test_data = group_df[group_df['period_type'] == 'test']
        if not test_data.empty and 'predictions' in test_data.columns:
            actuals_test = test_data['actuals'].dropna()
            predictions_test = test_data['predictions'].dropna()
            # Align series
            common_index_test = actuals_test.index.intersection(predictions_test.index)
            actuals_test = actuals_test.loc[common_index_test]
            predictions_test = predictions_test.loc[common_index_test]
            
            if not actuals_test.empty:
                for metric_name in selected_metrics:
                    if metric_name in metric_functions:
                        try:
                            value = metric_functions[metric_name](actuals_test, predictions_test)
                            metrics_results.append({
                                'Slice ID': current_slice_id, 'Model ID': current_model_id,
                                'Period Type': 'Test (in range)', 'Metric': metric_name.upper(), 'Value': f"{value:.3f}"
                            })
                            if selected_split_model == 'all_aggregated':
                                aggregated_metrics_data[metric_name]['test'].append(value)
                        except Exception as e:
                            print(f"Error calculating test metric {metric_name} for {current_slice_id}-{current_model_id}: {e}")
                            metrics_results.append({
                                'Slice ID': current_slice_id, 'Model ID': current_model_id,
                                'Period Type': 'Test (in range)', 'Metric': metric_name.upper(), 'Value': "Error"
                            })
    
    # If aggregated view, calculate mean of metrics
    if selected_split_model == 'all_aggregated':
        aggregated_summary_results = []
        for metric_name in selected_metrics:
            for period in ['train', 'test']:
                values = [v for v in aggregated_metrics_data[metric_name][period] if not np.isnan(v)] # filter out NaNs from MAPE
                if values:
                    mean_val = np.mean(values)
                    aggregated_summary_results.append({
                        'Slice ID': 'ALL (Mean)', 'Model ID': 'ALL (Mean)',
                        'Period Type': f'{period.capitalize()} (in range)', 
                        'Metric': metric_name.upper(), 'Value': f"{mean_val:.3f}"
                    })
        # Prepend aggregated summary to individual results if needed, or display separately.
        # For this DataTable, let's add them to the main table.
        metrics_results = aggregated_summary_results + metrics_results


    table_columns = [{"name": i, "id": i} for i in ['Slice ID', 'Model ID', 'Period Type', 'Metric', 'Value']]
    
    # Update figure title if it hasn't been set explicitly by plot_resamples
    if not figure_to_display.layout.title.text and plot_title:
         figure_to_display.update_layout(title_text=plot_title)
    elif not figure_to_display.layout.title.text:
         figure_to_display.update_layout(title_text="Resample Plot")


    return figure_to_display, table_columns, metrics_results


# --- Run the App ---
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)  