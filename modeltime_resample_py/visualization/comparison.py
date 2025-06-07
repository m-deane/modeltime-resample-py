"""Model comparison matrix visualization."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings


def plot_model_comparison_matrix(
    accuracy_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    plot_type: str = 'heatmap',
    title: str = "Model Performance Comparison Matrix",
    figsize: Tuple[int, int] = (12, 8),
    engine: str = 'plotly',
    show_values: bool = True,
    value_format: str = '.3f',
    colorscale: str = 'RdYlGn',
    **kwargs
) -> Union[go.Figure, plt.Figure]:
    """
    Create a visual model performance comparison matrix.
    
    Args:
        accuracy_df: Output from resample_accuracy
        metrics: List of metrics to include (None for all)
        models: List of models to include (None for all)
        plot_type: Type of plot ('heatmap', 'radar', 'parallel')
        title: Plot title
        figsize: Figure size
        engine: 'plotly' or 'matplotlib' 
        show_values: Whether to show values in heatmap
        value_format: Format string for values
        colorscale: Color scale for plotly
        **kwargs: Additional plot arguments
        
    Returns:
        Plotly figure or matplotlib figure
        
    Example:
        >>> fig = plot_model_comparison_matrix(
        ...     accuracy_df=accuracy,
        ...     metrics=['rmse', 'mape'],
        ...     plot_type='heatmap'
        ... )
    """
    # Filter data
    if metrics is not None:
        accuracy_df = accuracy_df[accuracy_df['metric_name'].isin(metrics)]
    if models is not None:
        accuracy_df = accuracy_df[accuracy_df['model_id'].isin(models)]
    
    if plot_type == 'heatmap':
        return _create_heatmap_comparison(
            accuracy_df, title, figsize, engine, 
            show_values, value_format, colorscale, **kwargs
        )
    elif plot_type == 'radar':
        return _create_radar_comparison(
            accuracy_df, title, engine, **kwargs
        )
    elif plot_type == 'parallel':
        return _create_parallel_comparison(
            accuracy_df, title, engine, **kwargs
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


def _create_heatmap_comparison(
    accuracy_df: pd.DataFrame,
    title: str,
    figsize: Tuple[int, int],
    engine: str,
    show_values: bool,
    value_format: str,
    colorscale: str,
    **kwargs
) -> Union[go.Figure, plt.Figure]:
    """Create heatmap comparison."""
    # Calculate mean metrics for each model
    summary = accuracy_df.groupby(['model_id', 'metric_name'])['metric_value'].agg(['mean', 'std'])
    
    # Pivot for heatmap
    mean_matrix = summary['mean'].reset_index().pivot(
        index='model_id', columns='metric_name', values='mean'
    )
    
    if engine == 'plotly':
        # Create annotations if showing values
        if show_values:
            # Also get std for annotations
            std_matrix = summary['std'].reset_index().pivot(
                index='model_id', columns='metric_name', values='std'
            )
            
            annotations = []
            for i, model in enumerate(mean_matrix.index):
                for j, metric in enumerate(mean_matrix.columns):
                    mean_val = mean_matrix.loc[model, metric]
                    std_val = std_matrix.loc[model, metric]
                    
                    if pd.notna(mean_val):
                        text = f"{mean_val:{value_format}}<br>±{std_val:{value_format}}"
                        annotations.append(
                            dict(
                                text=text,
                                x=j,
                                y=i,
                                xref='x',
                                yref='y',
                                showarrow=False,
                                font=dict(size=10)
                            )
                        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=mean_matrix.values,
            x=mean_matrix.columns,
            y=mean_matrix.index,
            colorscale=colorscale,
            text=mean_matrix.values if not show_values else None,
            texttemplate='%{text:{value_format}}' if not show_values else None,
            hovertemplate='Model: %{y}<br>Metric: %{x}<br>Value: %{z:{value_format}}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(title="Metric", tickangle=-45),
            yaxis=dict(title="Model"),
            height=figsize[1] * 80,
            width=figsize[0] * 80,
            annotations=annotations if show_values else None
        )
        
        return fig
        
    else:  # matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            mean_matrix,
            annot=show_values,
            fmt=value_format[1:],  # Remove the dot
            cmap='RdYlGn_r',
            ax=ax,
            cbar_kws={'label': 'Metric Value'},
            **kwargs
        )
        
        # Add title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig


def _create_radar_comparison(
    accuracy_df: pd.DataFrame,
    title: str,
    engine: str,
    **kwargs
) -> go.Figure:
    """Create radar chart comparison."""
    if engine != 'plotly':
        warnings.warn("Radar charts only supported with plotly engine")
    
    # Calculate mean metrics
    summary = accuracy_df.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    summary = summary.reset_index().pivot(
        index='model_id', columns='metric_name', values='metric_value'
    )
    
    # Normalize metrics to 0-1 scale for better visualization
    normalized = summary.copy()
    for col in normalized.columns:
        col_min = normalized[col].min()
        col_max = normalized[col].max()
        if col_max > col_min:
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
    
    # Create radar chart
    fig = go.Figure()
    
    categories = list(normalized.columns)
    
    for model in normalized.index:
        values = normalized.loc[model].values.tolist()
        values += values[:1]  # Complete the circle
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model,
            hovertemplate='%{theta}: %{r:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title=title,
        height=600,
        width=700
    )
    
    return fig


def _create_parallel_comparison(
    accuracy_df: pd.DataFrame,
    title: str,
    engine: str,
    **kwargs
) -> go.Figure:
    """Create parallel coordinates plot comparison."""
    if engine != 'plotly':
        warnings.warn("Parallel coordinates only supported with plotly engine")
    
    # Prepare data
    summary = accuracy_df.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    summary = summary.reset_index().pivot(
        index='model_id', columns='metric_name', values='metric_value'
    )
    
    # Create dimensions for parallel coordinates
    dimensions = []
    for metric in summary.columns:
        dimensions.append(
            dict(
                label=metric,
                values=summary[metric],
                range=[summary[metric].min(), summary[metric].max()]
            )
        )
    
    # Create color scale based on average performance
    avg_performance = summary.mean(axis=1)
    
    # Create parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=avg_performance,
                colorscale='Viridis',
                showscale=True,
                cmin=avg_performance.min(),
                cmax=avg_performance.max()
            ),
            dimensions=dimensions,
            labelangle=-45,
            labelside='bottom'
        )
    )
    
    # Add model labels
    for i, model in enumerate(summary.index):
        fig.add_annotation(
            x=1.02,
            y=i / (len(summary) - 1),
            text=model,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(size=10)
        )
    
    fig.update_layout(
        title=title,
        height=500,
        margin=dict(l=100, r=150, t=50, b=100)
    )
    
    return fig


def create_comparison_report(
    accuracy_df: pd.DataFrame,
    output_path: Optional[str] = None,
    include_plots: List[str] = ['heatmap', 'radar', 'parallel'],
    metrics: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    title: str = "Model Comparison Report",
    engine: str = 'plotly'
) -> Dict[str, Any]:
    """
    Create a comprehensive model comparison report.
    
    Args:
        accuracy_df: Output from resample_accuracy
        output_path: Path to save HTML report (optional)
        include_plots: Types of plots to include
        metrics: Metrics to include (None for all)
        models: Models to include (None for all)
        title: Report title
        engine: Plotting engine
        
    Returns:
        Dictionary containing:
            - 'figures': Dict of plot type -> figure
            - 'summary_stats': DataFrame of summary statistics
            - 'rankings': DataFrame of model rankings by metric
            - 'html': HTML report string (if output_path provided)
            
    Example:
        >>> report = create_comparison_report(
        ...     accuracy_df=accuracy,
        ...     output_path='model_comparison.html',
        ...     include_plots=['heatmap', 'radar']
        ... )
    """
    # Filter data
    if metrics is not None:
        accuracy_df = accuracy_df[accuracy_df['metric_name'].isin(metrics)]
    if models is not None:
        accuracy_df = accuracy_df[accuracy_df['model_id'].isin(models)]
    
    # Create figures
    figures = {}
    for plot_type in include_plots:
        try:
            figures[plot_type] = plot_model_comparison_matrix(
                accuracy_df, 
                plot_type=plot_type,
                engine=engine
            )
        except Exception as e:
            print(f"Warning: Could not create {plot_type} plot: {str(e)}")
    
    # Calculate summary statistics
    summary_stats = accuracy_df.groupby(['model_id', 'metric_name'])['metric_value'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(4)
    
    # Calculate rankings
    mean_metrics = accuracy_df.groupby(['model_id', 'metric_name'])['metric_value'].mean()
    mean_pivot = mean_metrics.reset_index().pivot(
        index='model_id', columns='metric_name', values='metric_value'
    )
    
    # Rank models (lower is better for error metrics)
    rankings = pd.DataFrame(index=mean_pivot.index)
    for metric in mean_pivot.columns:
        rankings[f'{metric}_rank'] = mean_pivot[metric].rank()
    
    # Calculate average rank
    rankings['avg_rank'] = rankings.mean(axis=1)
    rankings = rankings.sort_values('avg_rank')
    
    # Create HTML report if requested
    html_report = None
    if output_path and engine == 'plotly':
        html_parts = [
            f"<html><head><title>{title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2 { color: #333; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }",
            "th { background-color: #f2f2f2; }",
            ".plot-container { margin: 30px 0; }",
            "</style></head><body>",
            f"<h1>{title}</h1>",
            f"<p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        ]
        
        # Add summary section
        html_parts.extend([
            "<h2>Model Rankings</h2>",
            rankings.round(2).to_html(),
            "<h2>Summary Statistics</h2>",
            summary_stats.to_html()
        ])
        
        # Add plots
        for plot_type, fig in figures.items():
            if isinstance(fig, go.Figure):
                html_parts.extend([
                    f'<div class="plot-container">',
                    f'<h2>{plot_type.title()} Comparison</h2>',
                    fig.to_html(include_plotlyjs='cdn'),
                    '</div>'
                ])
        
        html_parts.append("</body></html>")
        html_report = '\n'.join(html_parts)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_report)
        
        print(f"Report saved to: {output_path}")
    
    return {
        'figures': figures,
        'summary_stats': summary_stats,
        'rankings': rankings,
        'html': html_report
    } 