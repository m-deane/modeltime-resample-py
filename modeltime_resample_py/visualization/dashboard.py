"""Enhanced interactive dashboard for exploring model resampling results."""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


class EnhancedResamplesDashboard:
    """Enhanced interactive dashboard for exploring time series model results."""
    
    def __init__(
        self,
        resamples_df: pd.DataFrame,
        accuracy_df: Optional[pd.DataFrame] = None,
        title: str = "Time Series Model Analysis Dashboard"
    ):
        """
        Initialize the enhanced dashboard.
        
        Args:
            resamples_df: Output from fit_resamples
            accuracy_df: Output from resample_accuracy (optional)
            title: Dashboard title
        """
        self.resamples_df = resamples_df.copy()
        self.accuracy_df = accuracy_df.copy() if accuracy_df is not None else None
        self.title = title
        
        # Ensure proper index and data structure
        self._prepare_data()
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._setup_layout()
        self._setup_callbacks()
        
        # Add client-side callback to trigger initial load
        self.app.clientside_callback(
            """
            function(id) {
                if (id) {
                    // Trigger the update button after a short delay
                    setTimeout(function() {
                        var updateBtn = document.getElementById('update-button');
                        if (updateBtn) {
                            updateBtn.click();
                        }
                    }, 1000);
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output('update-button', 'n_clicks'),
            Input('main-tabs', 'id')
        )
    
    def _prepare_data(self):
        """Prepare and validate data for dashboard use."""
        # Ensure the index is sorted for proper slicing and plotting
        self.resamples_df = self.resamples_df.sort_index()
        
        # Add residuals if not present
        if 'residuals' not in self.resamples_df.columns:
            train_mask = self.resamples_df['period_type'] == 'train'
            test_mask = self.resamples_df['period_type'] == 'test'
            
            residuals = pd.Series(index=self.resamples_df.index, dtype=float)
            residuals[train_mask] = (self.resamples_df.loc[train_mask, 'actuals'] - 
                                   self.resamples_df.loc[train_mask, 'fitted_values'])
            residuals[test_mask] = (self.resamples_df.loc[test_mask, 'actuals'] - 
                                  self.resamples_df.loc[test_mask, 'predictions'])
            
            self.resamples_df['residuals'] = residuals
        
        # Get unique values for filters
        self.unique_models = self.resamples_df.index.get_level_values('model_id').unique().tolist()
        self.unique_slices = self.resamples_df.index.get_level_values('slice_id').unique().tolist()
        
        # Date range
        dates = self.resamples_df.index.get_level_values('date')
        self.min_date = dates.min()
        self.max_date = dates.max()
        
        # Create model options for filtering
        self.model_options = [
            {'label': model, 'value': model} for model in self.unique_models
        ]
        
        # Create split/model combinations
        unique_groups = self.resamples_df.index.droplevel('date').unique().tolist()
        self.split_model_options = [
            {'label': f"Slice {s}, Model {m}", 'value': f"{s}_{m}"} 
            for s, m in unique_groups
        ]
        self.split_model_options.insert(0, {'label': 'All Splits - Aggregated View', 'value': 'all_aggregated'})
        self.split_model_options.insert(1, {'label': 'All Splits - Separate Plots', 'value': 'all_separate'})
        
        # Create split options for multi-select
        self.split_options = [
            {'label': f"Slice {s}", 'value': s} for s in self.unique_slices
        ]
    
    def _setup_layout(self):
        """Create the enhanced dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(self.title, className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Summary Statistics Bar (moved to top)
            dbc.Row([
                dbc.Col([
                    dbc.Alert(
                        id='summary-stats',
                        color="info",
                        className="text-center mb-4"
                    )
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Controls & Filters"),
                        dbc.CardBody([
                            # Date range picker
                            dbc.Label("📅 Select Date Range:"),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                min_date_allowed=self.min_date.date(),
                                max_date_allowed=self.max_date.date(),
                                initial_visible_month=self.min_date.date(),
                                start_date=self.min_date.date(),
                                end_date=self.max_date.date(),
                                className="mb-3"
                            ),
                            
                            # Model selector
                            dbc.Label("🤖 Select Models:"),
                            dcc.Dropdown(
                                id='model-selector',
                                options=self.model_options,
                                value=self.unique_models,  # Select all by default
                                multi=True,
                                clearable=False,
                                className="mb-3"
                            ),
                            
                            # Split selector
                            dbc.Label("🎯 Select Splits:"),
                            dcc.Dropdown(
                                id='split-selector',
                                options=self.split_options,
                                value=self.unique_slices,  # Select all by default
                                multi=True,
                                clearable=False,
                                className="mb-3"
                            ),
                            
                            # View mode selector
                            dbc.Label("📊 View Mode:"),
                            dcc.Dropdown(
                                id='view-mode-selector',
                                options=[
                                    {'label': 'All Splits - Aggregated View', 'value': 'all_aggregated'},
                                    {'label': 'All Splits - Separate Plots', 'value': 'all_separate'}
                                ],
                                value='all_aggregated',
                                clearable=False,
                                className="mb-3"
                            ),
                            
                            # Performance metrics selector
                            dbc.Label("📈 Select Performance Metrics:"),
                            dcc.Dropdown(
                                id='metric-selector',
                                options=[
                                    {'label': 'MAE (Mean Absolute Error)', 'value': 'mae'},
                                    {'label': 'RMSE (Root Mean Squared Error)', 'value': 'rmse'},
                                    {'label': 'MAPE (Mean Absolute Percentage Error)', 'value': 'mape'}
                                ],
                                value=['mae', 'rmse'],
                                multi=True,
                                className="mb-3"
                            ),
                            
                            # View options
                            dbc.Label("👁️ Display Options:"),
                            dbc.Checklist(
                                id='display-options',
                                options=[
                                    {'label': 'Show Train Period', 'value': 'show_train'},
                                    {'label': 'Show Test Period', 'value': 'show_test'},
                                    {'label': 'Show Residuals', 'value': 'show_residuals'},
                                    {'label': 'Show Confidence Bands', 'value': 'show_confidence'}
                                ],
                                value=['show_train', 'show_test'],
                                className="mb-3"
                            ),
                            
                            # Update button
                            dbc.Button(
                                '🔄 Update View / Calculate Metrics',
                                id='update-button',
                                n_clicks=0,
                                color="primary",
                                className="w-100 mb-3"
                            ),
                            
                            # Export button
                            dbc.Button(
                                '💾 Export Data',
                                id='export-button',
                                color="secondary",
                                className="w-100"
                            ),
                            dcc.Download(id="download-data")
                        ])
                    ])
                ], md=3),
                
                # Main content area
                dbc.Col([
                    # Tabs for different views
                    dcc.Tabs(id='main-tabs', value='plot-tab', children=[
                        dcc.Tab(label='📊 Time Series Plot', value='plot-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Time Series Visualization"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-plot",
                                        type="default",
                                        children=dcc.Graph(
                                            id='resample-plot',
                                            figure=go.Figure(),
                                            style={'height': '600px'}
                                        )
                                    )
                                ])
                            ])
                        ]),
                        
                        dcc.Tab(label='📈 Performance Metrics', value='metrics-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Model Performance Analysis"),
                                dbc.CardBody([
                                    # Performance controls
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("📊 Metrics Granularity:"),
                                            dcc.Dropdown(
                                                id='metrics-granularity-dropdown',
                                                options=[
                                                    {'label': 'Overall', 'value': 'overall'},
                                                    {'label': 'By Model', 'value': 'by_model'},
                                                    {'label': 'By Split', 'value': 'by_split'},
                                                    {'label': 'By Model & Split', 'value': 'by_model_split'}
                                                ],
                                                value='by_model',
                                                clearable=False,
                                                className="mb-3"
                                            )
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("🎯 Split Type Filter:"),
                                            dcc.Dropdown(
                                                id='metrics-split-type-filter-dropdown',
                                                options=[
                                                    {'label': 'All', 'value': 'all'},
                                                    {'label': 'Train Only', 'value': 'train'},
                                                    {'label': 'Test Only', 'value': 'test'}
                                                ],
                                                value='all',
                                                clearable=False,
                                                className="mb-3"
                                            )
                                        ], md=4),
                                        dbc.Col([
                                            dbc.Label("📋 Comparison Mode:"),
                                            dbc.Switch(
                                                id='comparison-mode-switch',
                                                label="Enable Comparison",
                                                value=False,
                                                className="mb-3"
                                            )
                                        ], md=4)
                                    ]),
                                    
                                    # Baseline model selector (shown when comparison mode is on)
                                    html.Div(id='performance-baseline-container', children=[
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("🎯 Baseline Model:"),
                                                dcc.Dropdown(
                                                    id='performance-baseline-dropdown',
                                                    options=[],
                                                    value=None,
                                                    clearable=False,
                                                    disabled=True,
                                                    className="mb-3"
                                                )
                                            ], md=6),
                                            dbc.Col([
                                                html.Div(id='comparison-mode-info', className="mt-4")
                                            ], md=6)
                                        ])
                                    ]),
                                    
                                    # Metrics column selector
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("📊 Select Metric Columns:"),
                                            dbc.Checklist(
                                                id='metrics-column-toggle-checklist',
                                                options=[
                                                    {'label': 'MAE', 'value': 'mae'},
                                                    {'label': 'RMSE', 'value': 'rmse'},
                                                    {'label': 'MAPE', 'value': 'mape'},
                                                    {'label': 'R²', 'value': 'r2'},
                                                    {'label': 'Mean Error', 'value': 'mean_error'},
                                                    {'label': 'Std Error', 'value': 'std_error'}
                                                ],
                                                value=['mae', 'rmse', 'mape'],
                                                inline=True,
                                                className="mb-3"
                                            )
                                        ])
                                    ]),
                                    
                                    # Performance metrics table container
                                    dcc.Loading(
                                        id="loading-metrics",
                                        type="default",
                                        children=[
                                            html.Div(id='metrics-summary', className="mb-4"),
                                            html.Div(id='performance-metrics-table-container')
                                        ]
                                    )
                                ])
                            ])
                        ]),
                        
                        dcc.Tab(label='🔍 Residual Analysis', value='residuals-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Residual Analysis"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-residuals",
                                        type="default",
                                        children=dcc.Graph(
                                            id='residuals-plot',
                                            style={'height': '600px'}
                                        )
                                    )
                                ])
                            ])
                        ]),
                        
                        dcc.Tab(label='📋 Data Table', value='data-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Raw Data Explorer"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-table",
                                        type="default",
                                        children=html.Div(id='data-table-content')
                                    )
                                ])
                            ])
                        ]),
                        
                        dcc.Tab(label='📊 Model Comparison', value='comparison-tab', children=[
                            dbc.Card([
                                dbc.CardHeader("Model Comparison Dashboard"),
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading-comparison",
                                        type="default",
                                        children=html.Div(id='comparison-content')
                                    )
                                ])
                            ])
                        ])
                    ])
                ], md=9)
            ])
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Set up enhanced dashboard callbacks."""
        
        @self.app.callback(
            [Output('resample-plot', 'figure'),
             Output('metrics-summary', 'children'),
             Output('residuals-plot', 'figure'),
             Output('data-table-content', 'children'),
             Output('comparison-content', 'children'),
             Output('summary-stats', 'children')],
            [Input('update-button', 'n_clicks'),
             Input('display-options', 'value'),
             Input('view-mode-selector', 'value')],
            [State('date-picker-range', 'start_date'),
             State('date-picker-range', 'end_date'),
             State('model-selector', 'value'),
             State('split-selector', 'value'),
             State('metric-selector', 'value')]
        )
        def update_dashboard(n_clicks, display_options, view_mode, start_date_str, end_date_str, selected_models, 
                           selected_splits, selected_metrics):
            
            if n_clicks == 0 or not selected_metrics or not selected_models or not selected_splits:
                empty_fig = go.Figure()
                empty_fig.update_layout(title_text="Select options and click 'Update' to view analysis.")
                return (empty_fig, "Select metrics to view summary", empty_fig, 
                       "No data selected", "No data selected", "Click Update to load data")
            
            # Filter data
            filtered_df = self._filter_data(start_date_str, end_date_str, selected_models, selected_splits)
            
            if filtered_df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(title_text="No data available for the selected filters.")
                return (empty_fig, "No data available", empty_fig,
                       "No data available", "No data available", "No data for selected filters")
                        
            # Create visualizations
            plot_fig = self._create_enhanced_timeseries_plot(filtered_df, display_options, view_mode)
            residuals_fig = self._create_enhanced_residuals_plot(filtered_df)
            
            # Calculate metrics
            metrics_data, metrics_summary = self._calculate_enhanced_metrics(
                filtered_df, selected_metrics, view_mode
            )
            
            # Create data table
            data_table = self._create_enhanced_data_table(filtered_df)
            
            # Create comparison content
            comparison_content = self._create_model_comparison(filtered_df, selected_metrics)
            
            # Create summary
            summary = self._create_enhanced_summary(filtered_df, view_mode)
            
            return (plot_fig, metrics_summary, residuals_fig, data_table, comparison_content, summary)
        
        @self.app.callback(
            Output("download-data", "data"),
            Input("export-button", "n_clicks"),
            [State('date-picker-range', 'start_date'),
             State('date-picker-range', 'end_date'),
             State('model-selector', 'value'),
             State('split-selector', 'value')],
            prevent_initial_call=True
        )
        def export_data(n_clicks, start_date_str, end_date_str, selected_models, selected_splits):
            filtered_df = self._filter_data(start_date_str, end_date_str, selected_models, selected_splits)
            
            # Reset index to make it exportable
            export_df = filtered_df.reset_index()
            
            return dcc.send_data_frame(
                export_df.to_csv,
                f"resample_data_{start_date_str}_{end_date_str}.csv",
                index=False
            )
        
        # Enhanced Performance Metrics Callbacks
        @self.app.callback(
            [Output('performance-baseline-dropdown', 'options'),
             Output('performance-baseline-dropdown', 'disabled'),
             Output('comparison-mode-info', 'children')],
            [Input('comparison-mode-switch', 'value'),
             Input('model-selector', 'value')]
        )
        def manage_performance_baseline_dropdown(comparison_mode, selected_models):
            if not comparison_mode or not selected_models or len(selected_models) < 2:
                return [], True, dbc.Alert("Select at least 2 models to enable comparison mode.", color="warning")
            
            options = [{'label': model, 'value': model} for model in selected_models]
            info = dbc.Alert(f"Comparing {len(selected_models)} models. Select baseline for relative metrics.", color="info")
            return options, False, info
        
        @self.app.callback(
            Output('performance-metrics-table-container', 'children'),
            [Input('update-button', 'n_clicks')],
            [State('model-selector', 'value'),
             State('split-selector', 'value'),
             State('date-picker-range', 'start_date'),
             State('date-picker-range', 'end_date'),
             State('metrics-granularity-dropdown', 'value'),
             State('metrics-split-type-filter-dropdown', 'value'),
             State('metrics-column-toggle-checklist', 'value'),
             State('comparison-mode-switch', 'value'),
             State('performance-baseline-dropdown', 'value')]
        )
        def generate_performance_metrics_table(n_clicks, selected_models, selected_splits, 
                                             start_date_str, end_date_str, granularity, 
                                             split_filter, selected_metric_columns, 
                                             comparison_mode, baseline_model):
            if n_clicks == 0 or not selected_models or not selected_splits:
                return html.Div("Click 'Update View' to generate performance metrics table.")
            
            # Filter data
            filtered_df = self._filter_data(start_date_str, end_date_str, selected_models, selected_splits)
            
            if filtered_df.empty:
                return html.Div("No data available for selected filters.")
            
            return self._create_enhanced_performance_table(
                filtered_df, granularity, split_filter, selected_metric_columns, 
                comparison_mode, baseline_model
            )
    
    def _filter_data(self, start_date_str, end_date_str, selected_models, selected_splits):
        """Filter data based on selections."""
        # Convert dates
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        
        # Filter by date range
        df_filtered = self.resamples_df[
            (self.resamples_df.index.get_level_values('date') >= start_date) &
            (self.resamples_df.index.get_level_values('date') <= end_date)
        ]
        
        # Filter by model selection
        if selected_models:
            df_filtered = df_filtered[
                df_filtered.index.get_level_values('model_id').isin(selected_models)
            ]
        
        # Filter by split selection
        if selected_splits:
            df_filtered = df_filtered[
                df_filtered.index.get_level_values('slice_id').isin(selected_splits)
            ]
        
        return df_filtered
    
    def _create_enhanced_timeseries_plot(self, df, display_options, view_mode=None):
        """Create enhanced time series plot with multiple options."""
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title_text="No data to display")
            return fig
        
        # Check if we need separate plots for each slice/model combination
        if view_mode == 'all_separate':
            return self._create_separate_plots(df, display_options)
        
        fig = go.Figure()
        
        # Color palette for different models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Group by model and slice
        for i, ((slice_id, model_id), group) in enumerate(df.groupby(level=['slice_id', 'model_id'])):
            color = colors[i % len(colors)]
            dates = group.index.get_level_values('date')
            
            # Plot actuals - heavy dashed dark blue line
            fig.add_trace(go.Scatter(
                x=dates,
                y=group['actuals'],
                mode='lines',
                name=f'Actuals - {model_id} (Slice {slice_id})',
                line=dict(color='#1e3a8a', width=4, dash='dash'),  # Heavy dashed dark blue
                opacity=1.0
            ))
            
            # Plot fitted values (train period) - solid orange line
            if 'show_train' in display_options:
                train_data = group[group['period_type'] == 'train']
                if not train_data.empty:
                    fig.add_trace(go.Scatter(
                        x=train_data.index.get_level_values('date'),
                        y=train_data['fitted_values'],
                        mode='lines',
                        name=f'Train - {model_id} (Slice {slice_id})',
                        line=dict(color='#f97316', width=2, dash='solid'),  # Solid orange
                        opacity=1.0
                    ))
            
            # Plot predictions (test period) - solid red line
            if 'show_test' in display_options:
                test_data = group[group['period_type'] == 'test']
                if not test_data.empty:
                    fig.add_trace(go.Scatter(
                        x=test_data.index.get_level_values('date'),
                        y=test_data['predictions'],
                        mode='lines',
                        name=f'Test - {model_id} (Slice {slice_id})',
                        line=dict(color='#dc2626', width=2, dash='solid'),  # Solid red
                        opacity=1.0
                    ))
        
        fig.update_layout(
            title="Time Series Analysis: Actuals vs Fitted/Predicted Values",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=600
        )
        
        return fig
    
    def _create_separate_plots(self, df, display_options):
        """Create separate subplots for each slice/model combination."""
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title_text="No data to display")
            return fig
        
        # Get unique combinations
        unique_combinations = df.index.droplevel('date').unique().tolist()
        n_plots = len(unique_combinations)
        
        if n_plots == 0:
            fig = go.Figure()
            fig.update_layout(title_text="No data to display")
            return fig
        
        # Calculate subplot layout (prefer more rows than columns for better readability)
        n_cols = min(2, n_plots)  # Maximum 2 columns
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
        
        # Create subplot titles
        subplot_titles = [f"Slice {slice_id}, Model {model_id}" 
                         for slice_id, model_id in unique_combinations]
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # Color palette for different models
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        model_colors = {}
        
        # Assign consistent colors to models
        for i, model_id in enumerate(df.index.get_level_values('model_id').unique()):
            model_colors[model_id] = colors[i % len(colors)]
        
        # Plot each combination in its own subplot
        for idx, (slice_id, model_id) in enumerate(unique_combinations):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            # Get data for this combination
            try:
                group = df.xs((slice_id, model_id), level=('slice_id', 'model_id'))
                dates = group.index
                color = model_colors[model_id]
                
                # Plot actuals - heavy dashed dark blue line
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=group['actuals'],
                    mode='lines',
                    name=f'Actuals',
                    line=dict(color='#1e3a8a', width=4, dash='dash'),  # Heavy dashed dark blue
                    opacity=1.0,
                    showlegend=(idx == 0),  # Only show legend for first plot
                    legendgroup='actuals'
                ), row=row, col=col)
                
                # Plot fitted values (train period) - solid orange line
                if 'show_train' in display_options:
                    train_data = group[group['period_type'] == 'train']
                    if not train_data.empty:
                        fig.add_trace(go.Scatter(
                            x=train_data.index,
                            y=train_data['fitted_values'],
                            mode='lines',
                            name=f'Train',
                            line=dict(color='#f97316', width=2, dash='solid'),  # Solid orange
                            opacity=1.0,
                            showlegend=(idx == 0),
                            legendgroup='train'
                        ), row=row, col=col)
                
                # Plot predictions (test period) - solid red line
                if 'show_test' in display_options:
                    test_data = group[group['period_type'] == 'test']
                    if not test_data.empty:
                        fig.add_trace(go.Scatter(
                            x=test_data.index,
                            y=test_data['predictions'],
                            mode='lines',
                            name=f'Test',
                            line=dict(color='#dc2626', width=2, dash='solid'),  # Solid red
                            opacity=1.0,
                            showlegend=(idx == 0),
                            legendgroup='test'
                        ), row=row, col=col)
                
                # Add vertical line to separate train/test periods if both are shown
                if 'show_train' in display_options and 'show_test' in display_options:
                    train_data = group[group['period_type'] == 'train']
                    test_data = group[group['period_type'] == 'test']
                    
                    if not train_data.empty and not test_data.empty:
                        split_date = test_data.index.min()
                        fig.add_vline(
                            x=split_date,
                            line_dash="dash",
                            line_color="gray",
                            opacity=0.5,
                            row=row, col=col
                        )
                
            except KeyError:
                # Handle case where combination doesn't exist
                continue
        
        # Update layout - extend plots down the page with consistent dimensions
        fig.update_layout(
            title="Time Series Analysis: Individual Plots by Slice and Model",
            height=max(600, 400 * n_rows),  # Larger height to extend down the page
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Update axes labels
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(title_text="Date" if i == n_rows else "", row=i, col=j)
                fig.update_yaxes(title_text="Value" if j == 1 else "", row=i, col=j)
        
        return fig
    
    def _create_enhanced_performance_table(self, df, granularity, split_filter, selected_metric_columns, 
                                         comparison_mode, baseline_model):
        """Create enhanced performance metrics table similar to md-views app."""
        if df.empty:
            return html.Div("No data available.")
        
        # Filter by split type if specified
        if split_filter != 'all':
            df = df[df['period_type'] == split_filter]
        
        # Calculate metrics based on granularity
        metrics_df = self._calculate_performance_metrics(df, granularity)
        
        if metrics_df.empty:
            return html.Div("No metrics calculated for current selection.")
        
        # Filter columns based on selection
        available_columns = [col for col in selected_metric_columns if col in metrics_df.columns]
        if not available_columns:
            return html.Div("No selected metrics available in data.")
        
        # Create base table with selected columns
        display_columns = ['model_id'] if 'model_id' in metrics_df.columns else []
        if 'slice_id' in metrics_df.columns:
            display_columns.append('slice_id')
        if 'period_type' in metrics_df.columns:
            display_columns.append('period_type')
        display_columns.extend(available_columns)
        
        table_df = metrics_df[display_columns].copy()
        
        # Apply comparison mode if enabled
        if comparison_mode and baseline_model and baseline_model in table_df.get('model_id', []):
            table_df = self._apply_comparison_mode(table_df, baseline_model, available_columns)
        
        # Format numeric columns
        for col in available_columns:
            if col in table_df.columns:
                table_df[col] = table_df[col].round(4)
        
        # Create styled data table
        table_columns = [{"name": col.replace('_', ' ').title(), "id": col} for col in table_df.columns]
        
        # Add conditional styling for better/worse performance
        style_data_conditional = [
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
        
        # Add color coding for comparison mode
        if comparison_mode and baseline_model:
            for col in available_columns:
                if col in table_df.columns:
                    # Green for better, red for worse (assuming lower is better for most metrics)
                    style_data_conditional.extend([
                        {
                            'if': {
                                'filter_query': f'{{{col}}} < 0',
                                'column_id': col
                            },
                            'backgroundColor': '#d4edda',
                            'color': 'black',
                        },
                        {
                            'if': {
                                'filter_query': f'{{{col}}} > 0',
                                'column_id': col
                            },
                            'backgroundColor': '#f8d7da',
                            'color': 'black',
                        }
                    ])
        
        data_table = dash_table.DataTable(
            data=table_df.to_dict('records'),
            columns=table_columns,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=style_data_conditional,
            sort_action="native",
            filter_action="native",
            export_format="csv"
        )
        
        # Add summary statistics
        summary_stats = self._create_performance_summary(table_df, available_columns, comparison_mode)
        
        return html.Div([
            summary_stats,
            html.Hr(),
            data_table
        ])
    
    def _calculate_performance_metrics(self, df, granularity):
        """Calculate performance metrics based on granularity."""
        from sklearn.metrics import r2_score
        
        metrics_list = []
        
        if granularity == 'overall':
            # Overall metrics across all models and splits
            for period in df['period_type'].unique():
                period_data = df[df['period_type'] == period]
                if not period_data.empty:
                    metrics = self._compute_metrics_for_group(period_data)
                    metrics['period_type'] = period
                    metrics_list.append(metrics)
        
        elif granularity == 'by_model':
            # Metrics by model
            for model_id in df.index.get_level_values('model_id').unique():
                model_data = df[df.index.get_level_values('model_id') == model_id]
                for period in model_data['period_type'].unique():
                    period_data = model_data[model_data['period_type'] == period]
                    if not period_data.empty:
                        metrics = self._compute_metrics_for_group(period_data)
                        metrics['model_id'] = model_id
                        metrics['period_type'] = period
                        metrics_list.append(metrics)
        
        elif granularity == 'by_split':
            # Metrics by split
            for slice_id in df.index.get_level_values('slice_id').unique():
                split_data = df[df.index.get_level_values('slice_id') == slice_id]
                for period in split_data['period_type'].unique():
                    period_data = split_data[split_data['period_type'] == period]
                    if not period_data.empty:
                        metrics = self._compute_metrics_for_group(period_data)
                        metrics['slice_id'] = slice_id
                        metrics['period_type'] = period
                        metrics_list.append(metrics)
        
        elif granularity == 'by_model_split':
            # Metrics by model and split
            for (slice_id, model_id) in df.index.droplevel('date').unique():
                group_data = df.xs((slice_id, model_id), level=('slice_id', 'model_id'))
                for period in group_data['period_type'].unique():
                    period_data = group_data[group_data['period_type'] == period]
                    if not period_data.empty:
                        metrics = self._compute_metrics_for_group(period_data)
                        metrics['model_id'] = model_id
                        metrics['slice_id'] = slice_id
                        metrics['period_type'] = period
                        metrics_list.append(metrics)
        
        return pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
    
    def _compute_metrics_for_group(self, data):
        """Compute metrics for a group of data."""
        from sklearn.metrics import r2_score
        
        # Determine which values to use based on period type
        if 'train' in data['period_type'].values:
            y_true = data['actuals']
            y_pred = data['fitted_values']
        else:
            y_true = data['actuals']
            y_pred = data['predictions']
        
        # Remove any NaN values
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {}
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = calculate_mape(y_true, y_pred)
        
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan
        
        residuals = y_true - y_pred
        mean_error = np.mean(residuals)
        std_error = np.std(residuals)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'mean_error': mean_error,
            'std_error': std_error,
            'n_observations': len(y_true)
        }
    
    def _apply_comparison_mode(self, table_df, baseline_model, metric_columns):
        """Apply comparison mode to show relative performance."""
        if 'model_id' not in table_df.columns:
            return table_df
        
        baseline_data = table_df[table_df['model_id'] == baseline_model]
        if baseline_data.empty:
            return table_df
        
        result_df = table_df.copy()
        
        for col in metric_columns:
            if col in table_df.columns:
                baseline_value = baseline_data[col].iloc[0] if len(baseline_data) > 0 else 0
                if baseline_value != 0:
                    # Calculate percentage difference
                    result_df[col] = ((table_df[col] - baseline_value) / baseline_value * 100).round(2)
                else:
                    result_df[col] = 0
        
        return result_df
    
    def _create_performance_summary(self, table_df, metric_columns, comparison_mode):
        """Create performance summary cards."""
        if table_df.empty:
            return html.Div("No data for summary.")
        
        summary_cards = []
        
        for col in metric_columns:
            if col in table_df.columns:
                values = table_df[col].dropna()
                if len(values) > 0:
                    if comparison_mode:
                        avg_val = values.mean()
                        card_color = "success" if avg_val < 0 else "danger" if avg_val > 0 else "info"
                        card_text = f"Avg: {avg_val:.2f}%"
                    else:
                        avg_val = values.mean()
                        std_val = values.std()
                        card_color = "info"
                        card_text = f"Avg: {avg_val:.4f} ± {std_val:.4f}"
                    
                    card = dbc.Card([
                        dbc.CardBody([
                            html.H6(col.upper(), className="card-title"),
                            html.P(card_text, className="card-text")
                        ])
                    ], color=card_color, outline=True, className="mb-2")
                    
                    summary_cards.append(dbc.Col(card, md=2))
        
        if summary_cards:
            return dbc.Row(summary_cards, className="mb-3")
        else:
            return html.Div()

    def _create_enhanced_residuals_plot(self, df):
        """Create enhanced residuals analysis plot."""
        if df.empty:
            fig = go.Figure()
            fig.update_layout(title_text="No data for residuals analysis")
            return fig
        
        # Create subplots for different residual analyses
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals Over Time', 'Residuals Distribution', 
                          'Q-Q Plot', 'Residuals vs Fitted'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, ((slice_id, model_id), group) in enumerate(df.groupby(level=['slice_id', 'model_id'])):
            color = colors[i % len(colors)]
            dates = group.index.get_level_values('date')
            residuals = group['residuals'].dropna()
            
            if len(residuals) == 0:
                continue
            
            # Residuals over time
            fig.add_trace(
                go.Scatter(x=dates, y=group['residuals'], mode='markers',
                          name=f'{model_id} (Slice {slice_id})', marker_color=color),
                row=1, col=1
            )
            
            # Residuals distribution
            fig.add_trace(
                go.Histogram(x=residuals, name=f'{model_id} (Slice {slice_id})',
                           marker_color=color, opacity=0.7, nbinsx=20),
                row=1, col=2
            )
            
            # Q-Q plot (simplified)
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = np.linspace(-3, 3, len(sorted_residuals))
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                          mode='markers', name=f'{model_id} (Slice {slice_id})',
                          marker_color=color),
                row=2, col=1
            )
            
            # Residuals vs Fitted
            fitted_values = group['fitted_values'].fillna(group['predictions'])
            fig.add_trace(
                go.Scatter(x=fitted_values, y=group['residuals'],
                          mode='markers', name=f'{model_id} (Slice {slice_id})',
                          marker_color=color),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, title_text="Residual Analysis Dashboard")
        return fig
    
    def _calculate_enhanced_metrics(self, df, selected_metrics, view_mode):
        """Calculate enhanced performance metrics."""
        metrics_data = []
        
        # Define metric functions
        metric_functions = {
            'mae': mean_absolute_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': calculate_mape
        }
        
        # Calculate metrics for each group
        for (slice_id, model_id), group in df.groupby(level=['slice_id', 'model_id']):
            # Train metrics
            train_data = group[group['period_type'] == 'train']
            if not train_data.empty and 'fitted_values' in train_data.columns:
                actuals_train = train_data['actuals'].dropna()
                fitted_vals = train_data['fitted_values'].dropna()
                common_idx = actuals_train.index.intersection(fitted_vals.index)
                
                if len(common_idx) > 0:
                    for metric_name in selected_metrics:
                        try:
                            value = metric_functions[metric_name](
                                actuals_train.loc[common_idx], 
                                fitted_vals.loc[common_idx]
                            )
                            metrics_data.append({
                                'Slice ID': slice_id,
                                'Model ID': model_id,
                                'Period': 'Train',
                                'Metric': metric_name.upper(),
                                'Value': f"{value:.4f}",
                                'Count': len(common_idx)
                            })
                        except Exception as e:
                            metrics_data.append({
                                'Slice ID': slice_id,
                                'Model ID': model_id,
                                'Period': 'Train',
                                'Metric': metric_name.upper(),
                                'Value': "Error",
                                'Count': 0
                            })
            
            # Test metrics
            test_data = group[group['period_type'] == 'test']
            if not test_data.empty and 'predictions' in test_data.columns:
                actuals_test = test_data['actuals'].dropna()
                predictions = test_data['predictions'].dropna()
                common_idx = actuals_test.index.intersection(predictions.index)
                
                if len(common_idx) > 0:
                    for metric_name in selected_metrics:
                        try:
                            value = metric_functions[metric_name](
                                actuals_test.loc[common_idx], 
                                predictions.loc[common_idx]
                            )
                            metrics_data.append({
                                'Slice ID': slice_id,
                                'Model ID': model_id,
                                'Period': 'Test',
                                'Metric': metric_name.upper(),
                                'Value': f"{value:.4f}",
                                'Count': len(common_idx)
                            })
                        except Exception as e:
                            metrics_data.append({
                                'Slice ID': slice_id,
                                'Model ID': model_id,
                                'Period': 'Test',
                                'Metric': metric_name.upper(),
                                'Value': "Error",
                                'Count': 0
                            })
        
        # Create metrics summary
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df['Value_Numeric'] = pd.to_numeric(metrics_df['Value'], errors='coerce')
            
            summary_stats = metrics_df.groupby(['Metric', 'Period'])['Value_Numeric'].agg(['mean', 'std', 'count'])
            
            summary_cards = []
            for (metric, period), stats in summary_stats.iterrows():
                if not pd.isna(stats['mean']):
                    summary_cards.append(
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4(f"{metric} ({period})", className="card-title"),
                                    html.H2(f"{stats['mean']:.4f}", className="text-primary"),
                                    html.P(f"±{stats['std']:.4f} (n={int(stats['count'])})", 
                                          className="text-muted")
                                ])
                            ])
                        ], md=3)
                    )
            
            metrics_summary = dbc.Row(summary_cards) if summary_cards else "No valid metrics calculated"
        else:
            metrics_summary = "No metrics data available"
        
        return metrics_data, metrics_summary
    
    def _create_enhanced_data_table(self, df):
        """Create enhanced data table with filtering and sorting."""
        if df.empty:
            return "No data available"
        
        # Reset index for table display
        table_df = df.reset_index()
        
        # Round numeric columns
        numeric_cols = table_df.select_dtypes(include=[np.number]).columns
        table_df[numeric_cols] = table_df[numeric_cols].round(4)
        
        return dash_table.DataTable(
            data=table_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in table_df.columns],
            style_table={'overflowX': 'auto', 'height': '500px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '12px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{period_type} = train'},
                    'backgroundColor': 'rgba(0, 123, 255, 0.1)'
                },
                {
                    'if': {'filter_query': '{period_type} = test'},
                    'backgroundColor': 'rgba(255, 193, 7, 0.1)'
                }
            ],
            sort_action="native",
            filter_action="native",
            page_action="native",
            page_size=20
        )
    
    def _create_model_comparison(self, df, selected_metrics):
        """Create model comparison dashboard."""
        if df.empty or len(self.unique_models) < 2:
            return "Need multiple models for comparison"
        
        # Calculate average metrics by model
        comparison_data = []
        metric_functions = {
            'mae': mean_absolute_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': calculate_mape
        }
        
        for model_id in self.unique_models:
            model_data = df[df.index.get_level_values('model_id') == model_id]
            if model_data.empty:
                continue
            
            # Test period metrics
            test_data = model_data[model_data['period_type'] == 'test']
            if not test_data.empty:
                actuals = test_data['actuals'].dropna()
                predictions = test_data['predictions'].dropna()
                common_idx = actuals.index.intersection(predictions.index)
                
                if len(common_idx) > 0:
                    model_metrics = {'Model': model_id}
                    for metric_name in selected_metrics:
                        try:
                            value = metric_functions[metric_name](
                                actuals.loc[common_idx], 
                                predictions.loc[common_idx]
                            )
                            model_metrics[metric_name.upper()] = value
                        except:
                            model_metrics[metric_name.upper()] = np.nan
                    
                    comparison_data.append(model_metrics)
        
        if not comparison_data:
            return "No comparison data available"
        
        # Create comparison visualization
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create radar chart for model comparison
        fig = go.Figure()
        
        for _, row in comparison_df.iterrows():
            metrics_values = [row[col] for col in comparison_df.columns if col != 'Model']
            metrics_names = [col for col in comparison_df.columns if col != 'Model']
            
            # Normalize values for radar chart (invert for error metrics)
            normalized_values = []
            for val in metrics_values:
                if pd.isna(val):
                    normalized_values.append(0)
                else:
                    # For error metrics, lower is better, so invert
                    normalized_values.append(1 / (1 + val))
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=metrics_names,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison (Higher = Better)"
        )
        
        # Create comparison table
        comparison_table = dash_table.DataTable(
            data=comparison_df.round(4).to_dict('records'),
            columns=[{"name": i, "id": i} for i in comparison_df.columns],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
        
        return html.Div([
            dcc.Graph(figure=fig, style={'height': '400px'}),
            html.Hr(),
            html.H5("Detailed Comparison Table"),
            comparison_table
        ])
    
    def _create_enhanced_summary(self, df, view_mode):
        """Create enhanced summary statistics."""
        if df.empty:
            return "No data selected"
        
        n_models = len(df.index.get_level_values('model_id').unique())
        n_splits = len(df.index.get_level_values('slice_id').unique())
        n_obs = len(df)
        date_range = f"{df.index.get_level_values('date').min():%Y-%m-%d} to {df.index.get_level_values('date').max():%Y-%m-%d}"
        
        train_count = len(df[df['period_type'] == 'train'])
        test_count = len(df[df['period_type'] == 'test'])
        
        return f"📊 Models: {n_models} | 🔄 Splits: {n_splits} | 📈 Observations: {n_obs:,} | 🗓️ Range: {date_range} | 🏋️ Train: {train_count:,} | 🧪 Test: {test_count:,}"
    
    def run(self, debug=False, port=8050):
        """Run the enhanced dashboard."""
        print(f"🚀 Starting Enhanced Time Series Dashboard on http://localhost:{port}")
        print("📊 Features available:")
        print("   • Interactive time series visualization")
        print("   • Comprehensive performance metrics")
        print("   • Residual analysis dashboard")
        print("   • Model comparison tools")
        print("   • Data export capabilities")
        print("   • Advanced filtering and exploration")
        self.app.run_server(debug=debug, port=port)


def create_interactive_dashboard(
    resamples_df: pd.DataFrame,
    accuracy_df: Optional[pd.DataFrame] = None,
    title: str = "Time Series Model Analysis Dashboard",
    port: int = 8050,
    debug: bool = False
) -> EnhancedResamplesDashboard:
    """
    Create and optionally run an enhanced interactive dashboard for exploring results.
    
    Args:
        resamples_df: Output from fit_resamples
        accuracy_df: Output from resample_accuracy (optional)
        title: Dashboard title
        port: Port to run the dashboard on
        debug: Whether to run in debug mode
        
    Returns:
        EnhancedResamplesDashboard instance
        
    Example:
        >>> dashboard = create_interactive_dashboard(
        ...     resamples_df=results,
        ...     accuracy_df=accuracy,
        ...     title="My Enhanced Model Analysis"
        ... )
        >>> dashboard.run(port=8050)
    """
    dashboard = EnhancedResamplesDashboard(resamples_df, accuracy_df, title)
    return dashboard 