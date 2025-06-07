"""Interactive dashboard for exploring model resampling results."""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime


class ResamplesDashboard:
    """Interactive dashboard for exploring time series model results."""
    
    def __init__(
        self,
        resamples_df: pd.DataFrame,
        accuracy_df: Optional[pd.DataFrame] = None,
        title: str = "Time Series Model Analysis Dashboard"
    ):
        """
        Initialize the dashboard.
        
        Args:
            resamples_df: Output from fit_resamples
            accuracy_df: Output from resample_accuracy (optional)
            title: Dashboard title
        """
        self.resamples_df = resamples_df
        self.accuracy_df = accuracy_df
        self.title = title
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Create the dashboard layout."""
        # Get unique values for filters
        unique_models = self.resamples_df.index.get_level_values('model_id').unique()
        unique_slices = self.resamples_df.index.get_level_values('slice_id').unique()
        
        # Date range
        dates = self.resamples_df.index.get_level_values('date')
        min_date = dates.min()
        max_date = dates.max()
        
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1(self.title, className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Filters"),
                        dbc.CardBody([
                            # Model selector
                            html.Label("Select Models:"),
                            dcc.Dropdown(
                                id='model-selector',
                                options=[{'label': m, 'value': m} for m in unique_models],
                                value=list(unique_models),
                                multi=True,
                                className="mb-3"
                            ),
                            
                            # Split selector
                            html.Label("Select Splits:"),
                            dcc.Dropdown(
                                id='split-selector',
                                options=[{'label': f'Split {s}', 'value': s} for s in unique_slices],
                                value=list(unique_slices)[:5],  # Show first 5 by default
                                multi=True,
                                className="mb-3"
                            ),
                            
                            # Date range picker
                            html.Label("Date Range:"),
                            dcc.DatePickerRange(
                                id='date-range-picker',
                                start_date=min_date,
                                end_date=max_date,
                                display_format='YYYY-MM-DD',
                                className="mb-3"
                            ),
                            
                            # View type selector
                            html.Label("View Type:"),
                            dcc.RadioItems(
                                id='view-type',
                                options=[
                                    {'label': 'Time Series', 'value': 'timeseries'},
                                    {'label': 'Residuals', 'value': 'residuals'},
                                    {'label': 'Metrics', 'value': 'metrics'}
                                ],
                                value='timeseries',
                                className="mb-3"
                            ),
                            
                            # Update button
                            dbc.Button(
                                "Update View",
                                id="update-button",
                                color="primary",
                                className="w-100"
                            )
                        ])
                    ])
                ], md=3),
                
                # Main content area
                dbc.Col([
                    # Tabs for different views
                    dcc.Tabs(id='main-tabs', value='plot-tab', children=[
                        dcc.Tab(label='Visualization', value='plot-tab', children=[
                            dcc.Loading(
                                id="loading-plot",
                                type="default",
                                children=dcc.Graph(id='main-plot', style={'height': '600px'})
                            )
                        ]),
                        dcc.Tab(label='Statistics', value='stats-tab', children=[
                            dcc.Loading(
                                id="loading-stats",
                                type="default",
                                children=html.Div(id='stats-content')
                            )
                        ]),
                        dcc.Tab(label='Data Table', value='data-tab', children=[
                            dcc.Loading(
                                id="loading-table",
                                type="default",
                                children=html.Div(id='table-content')
                            )
                        ])
                    ])
                ], md=9)
            ]),
            
            # Footer with summary statistics
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.Div(id='summary-stats', className="text-center")
                ])
            ], className="mt-4")
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            [Output('main-plot', 'figure'),
             Output('stats-content', 'children'),
             Output('table-content', 'children'),
             Output('summary-stats', 'children')],
            [Input('update-button', 'n_clicks')],
            [State('model-selector', 'value'),
             State('split-selector', 'value'),
             State('date-range-picker', 'start_date'),
             State('date-range-picker', 'end_date'),
             State('view-type', 'value')]
        )
        def update_dashboard(n_clicks, models, splits, start_date, end_date, view_type):
            if not models or not splits:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="Please select at least one model and split",
                                       xref="paper", yref="paper", x=0.5, y=0.5)
                return empty_fig, "No data selected", "No data selected", "No data selected"
            
            # Filter data
            filtered_df = self._filter_data(models, splits, start_date, end_date)
            
            # Create visualizations based on view type
            if view_type == 'timeseries':
                fig = self._create_timeseries_plot(filtered_df)
            elif view_type == 'residuals':
                fig = self._create_residuals_plot(filtered_df)
            else:  # metrics
                fig = self._create_metrics_plot(filtered_df)
            
            # Create statistics content
            stats_content = self._create_stats_content(filtered_df)
            
            # Create table content
            table_content = self._create_table_content(filtered_df)
            
            # Create summary
            summary = self._create_summary(filtered_df)
            
            return fig, stats_content, table_content, summary
    
    def _filter_data(self, models, splits, start_date, end_date):
        """Filter the resamples data based on selections."""
        # Convert dates
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter by date
        date_mask = (
            (self.resamples_df.index.get_level_values('date') >= start_date) &
            (self.resamples_df.index.get_level_values('date') <= end_date)
        )
        
        # Filter by model and split
        model_mask = self.resamples_df.index.get_level_values('model_id').isin(models)
        split_mask = self.resamples_df.index.get_level_values('slice_id').isin(splits)
        
        return self.resamples_df[date_mask & model_mask & split_mask]
    
    def _create_timeseries_plot(self, df):
        """Create time series plot."""
        fig = go.Figure()
        
        # Get unique model/split combinations
        for (slice_id, model_id), group in df.groupby(level=['slice_id', 'model_id']):
            # Reset index to get date column
            group_reset = group.reset_index()
            
            # Add actuals
            fig.add_trace(go.Scatter(
                x=group_reset['date'],
                y=group_reset['actuals'],
                mode='lines',
                name=f'Actuals (S{slice_id}, {model_id})',
                line=dict(dash='dash'),
                opacity=0.7
            ))
            
            # Add predictions (test data)
            test_data = group_reset[group_reset['period_type'] == 'test']
            if not test_data.empty:
                fig.add_trace(go.Scatter(
                    x=test_data['date'],
                    y=test_data['predictions'],
                    mode='lines+markers',
                    name=f'Predictions (S{slice_id}, {model_id})',
                    marker=dict(size=4)
                ))
            
            # Add fitted values (train data)
            train_data = group_reset[group_reset['period_type'] == 'train']
            if not train_data.empty:
                fig.add_trace(go.Scatter(
                    x=train_data['date'],
                    y=train_data['fitted_values'],
                    mode='lines',
                    name=f'Fitted (S{slice_id}, {model_id})',
                    line=dict(dash='dot'),
                    opacity=0.6
                ))
        
        fig.update_layout(
            title="Time Series: Actuals vs Predictions",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        return fig
    
    def _create_residuals_plot(self, df):
        """Create residuals analysis plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals Over Time', 'Residual Distribution',
                          'Q-Q Plot', 'Residuals vs Fitted'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Prepare residuals data
        df_reset = df.reset_index()
        
        for model_id, model_group in df_reset.groupby('model_id'):
            residuals = model_group['residuals'].dropna()
            
            # 1. Residuals over time
            fig.add_trace(
                go.Scatter(
                    x=model_group['date'],
                    y=model_group['residuals'],
                    mode='markers',
                    name=f'{model_id}',
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # 2. Residual distribution
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name=f'{model_id}',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=2
            )
            
            # 3. Q-Q plot
            sorted_residuals = np.sort(residuals)
            norm_quantiles = np.random.normal(0, residuals.std(), len(residuals))
            norm_quantiles.sort()
            
            fig.add_trace(
                go.Scatter(
                    x=norm_quantiles,
                    y=sorted_residuals,
                    mode='markers',
                    name=f'{model_id}',
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
            
            # 4. Residuals vs Fitted
            train_data = model_group[model_group['period_type'] == 'train']
            test_data = model_group[model_group['period_type'] == 'test']
            
            if not train_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=train_data['fitted_values'],
                        y=train_data['residuals'],
                        mode='markers',
                        name=f'{model_id} (train)',
                        marker=dict(size=4)
                    ),
                    row=2, col=2
                )
            
            if not test_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=test_data['predictions'],
                        y=test_data['residuals'],
                        mode='markers',
                        name=f'{model_id} (test)',
                        marker=dict(size=4, symbol='square')
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Fitted/Predicted Values", row=2, col=2)
        fig.update_yaxes(title_text="Residuals", row=2, col=2)
        
        # Add zero line to residual plots
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="Residual Analysis")
        
        return fig
    
    def _create_metrics_plot(self, df):
        """Create metrics visualization."""
        if self.accuracy_df is None:
            fig = go.Figure()
            fig.add_annotation(
                text="No accuracy metrics available. Run resample_accuracy() first.",
                xref="paper", yref="paper", x=0.5, y=0.5
            )
            return fig
        
        # Filter accuracy data to match selected models
        df_models = df.index.get_level_values('model_id').unique()
        metrics_df = self.accuracy_df[
            self.accuracy_df['model_id'].isin(df_models)
        ]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Metrics by Model', 'Metrics by Split'),
            horizontal_spacing=0.15
        )
        
        # Metrics by model
        model_summary = metrics_df.groupby(['model_id', 'metric_name'])['metric_value'].agg(['mean', 'std'])
        
        for metric in metrics_df['metric_name'].unique():
            metric_data = model_summary.xs(metric, level='metric_name')
            
            fig.add_trace(
                go.Bar(
                    x=metric_data.index,
                    y=metric_data['mean'],
                    error_y=dict(type='data', array=metric_data['std']),
                    name=metric
                ),
                row=1, col=1
            )
        
        # Metrics by split
        for model_id in df_models:
            model_metrics = metrics_df[metrics_df['model_id'] == model_id]
            
            for metric in model_metrics['metric_name'].unique():
                metric_data = model_metrics[model_metrics['metric_name'] == metric]
                
                fig.add_trace(
                    go.Scatter(
                        x=metric_data['slice_id'],
                        y=metric_data['metric_value'],
                        mode='lines+markers',
                        name=f'{model_id} - {metric}'
                    ),
                    row=1, col=2
                )
        
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=1, col=1)
        fig.update_xaxes(title_text="Split ID", row=1, col=2)
        fig.update_yaxes(title_text="Metric Value", row=1, col=2)
        
        fig.update_layout(height=500, title_text="Model Performance Metrics")
        
        return fig
    
    def _create_stats_content(self, df):
        """Create statistics content."""
        stats = []
        
        # Overall statistics
        stats.append(html.H4("Overall Statistics"))
        
        overall_stats = pd.DataFrame({
            'Mean Actual': [df['actuals'].mean()],
            'Std Actual': [df['actuals'].std()],
            'Mean Residual': [df['residuals'].mean()],
            'Std Residual': [df['residuals'].std()],
            'Total Observations': [len(df)]
        }).T
        
        stats.append(
            dbc.Table.from_dataframe(
                overall_stats.reset_index(),
                striped=True,
                bordered=True,
                hover=True,
                index=False,
                columns=[{'name': 'Metric', 'id': 'index'}, {'name': 'Value', 'id': 0}]
            )
        )
        
        # Model-specific statistics
        stats.append(html.H4("Model Statistics", className="mt-4"))
        
        model_stats = df.groupby(level='model_id').agg({
            'residuals': ['mean', 'std', 'count'],
            'actuals': 'mean'
        }).round(3)
        
        stats.append(
            dbc.Table.from_dataframe(
                model_stats.reset_index(),
                striped=True,
                bordered=True,
                hover=True,
                index=False
            )
        )
        
        return html.Div(stats)
    
    def _create_table_content(self, df):
        """Create data table content."""
        # Prepare data for table
        table_df = df.reset_index().round(3)
        
        # Select columns to display
        columns = ['date', 'slice_id', 'model_id', 'actuals', 
                  'fitted_values', 'predictions', 'residuals', 'period_type']
        
        return dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in columns],
            data=table_df[columns].to_dict('records'),
            page_size=20,
            style_cell={'textAlign': 'left'},
            style_data_conditional=[
                {
                    'if': {'column_id': 'period_type', 'filter_query': '{period_type} = train'},
                    'backgroundColor': 'rgb(248, 248, 255)'
                },
                {
                    'if': {'column_id': 'period_type', 'filter_query': '{period_type} = test'},
                    'backgroundColor': 'rgb(255, 248, 248)'
                }
            ],
            filter_action="native",
            sort_action="native",
            page_action="native"
        )
    
    def _create_summary(self, df):
        """Create summary statistics."""
        n_models = len(df.index.get_level_values('model_id').unique())
        n_splits = len(df.index.get_level_values('slice_id').unique())
        n_obs = len(df)
        date_range = f"{df.index.get_level_values('date').min():%Y-%m-%d} to {df.index.get_level_values('date').max():%Y-%m-%d}"
        
        return html.Div([
            html.Span(f"Models: {n_models} | ", className="mr-3"),
            html.Span(f"Splits: {n_splits} | ", className="mr-3"),
            html.Span(f"Observations: {n_obs} | ", className="mr-3"),
            html.Span(f"Date Range: {date_range}")
        ])
    
    def run(self, debug=False, port=8050):
        """Run the dashboard."""
        self.app.run_server(debug=debug, port=port)


def create_interactive_dashboard(
    resamples_df: pd.DataFrame,
    accuracy_df: Optional[pd.DataFrame] = None,
    title: str = "Time Series Model Analysis Dashboard",
    port: int = 8050,
    debug: bool = False
) -> ResamplesDashboard:
    """
    Create and optionally run an interactive dashboard for exploring results.
    
    Args:
        resamples_df: Output from fit_resamples
        accuracy_df: Output from resample_accuracy (optional)
        title: Dashboard title
        port: Port to run the dashboard on
        debug: Whether to run in debug mode
        
    Returns:
        ResamplesDashboard instance
        
    Example:
        >>> dashboard = create_interactive_dashboard(
        ...     resamples_df=results,
        ...     accuracy_df=accuracy,
        ...     title="My Model Analysis"
        ... )
        >>> dashboard.run(port=8050)
    """
    dashboard = ResamplesDashboard(resamples_df, accuracy_df, title)
    return dashboard 