"""Tests for visualization features."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from modeltime_resample_py import (
    TimeSeriesDataPrep,
    fit_resamples,
    resample_accuracy,
    create_interactive_dashboard,
    plot_model_comparison_matrix,
    create_comparison_report
)
from modeltime_resample_py.visualization.dashboard import ResamplesDashboard
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    dates = pd.date_range(start='2022-01-01', periods=365, freq='D')
    values = 100 + np.random.randn(365).cumsum()
    
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'feature1': np.random.randn(365),
        'feature2': np.random.randn(365)
    })
    
    return df


@pytest.fixture
def sample_results(sample_data):
    """Generate sample resample results."""
    data_prep = TimeSeriesDataPrep(
        data=sample_data,
        date_column='date',
        target_column='value',
        feature_columns=['feature1', 'feature2']
    )
    
    models = {
        'linear': LinearRegression(),
        'rf': RandomForestRegressor(n_estimators=10, random_state=42)
    }
    
    results = fit_resamples(
        data_prep=data_prep,
        models=models,
        initial_window=180,
        assess_period=30,
        skip_period=30,
        slice_limit=3
    )
    
    return results


@pytest.fixture
def sample_accuracy(sample_results):
    """Generate sample accuracy results."""
    return resample_accuracy(sample_results, metrics=['rmse', 'mae', 'mape'])


class TestDashboard:
    """Test interactive dashboard functionality."""
    
    def test_dashboard_creation(self, sample_results, sample_accuracy):
        """Test dashboard can be created."""
        dashboard = create_interactive_dashboard(
            resamples_df=sample_results,
            accuracy_df=sample_accuracy,
            title="Test Dashboard"
        )
        
        assert isinstance(dashboard, ResamplesDashboard)
        assert dashboard.title == "Test Dashboard"
        assert dashboard.app is not None
    
    def test_dashboard_without_accuracy(self, sample_results):
        """Test dashboard works without accuracy data."""
        dashboard = create_interactive_dashboard(
            resamples_df=sample_results,
            accuracy_df=None
        )
        
        assert isinstance(dashboard, ResamplesDashboard)
        assert dashboard.accuracy_df is None
    
    def test_dashboard_layout(self, sample_results):
        """Test dashboard layout is created properly."""
        dashboard = ResamplesDashboard(sample_results)
        
        # Check layout components exist
        assert dashboard.app.layout is not None
        
        # Layout should have been set up by __init__
        layout_str = str(dashboard.app.layout)
        assert 'model-selector' in layout_str
        assert 'split-selector' in layout_str
        assert 'date-range-picker' in layout_str


class TestModelComparison:
    """Test model comparison visualizations."""
    
    def test_heatmap_comparison(self, sample_accuracy):
        """Test heatmap comparison plot."""
        fig = plot_model_comparison_matrix(
            accuracy_df=sample_accuracy,
            plot_type='heatmap',
            title='Test Heatmap'
        )
        
        assert fig is not None
        assert fig.layout.title.text == 'Test Heatmap'
        assert len(fig.data) > 0
    
    def test_radar_comparison(self, sample_accuracy):
        """Test radar chart comparison."""
        fig = plot_model_comparison_matrix(
            accuracy_df=sample_accuracy,
            plot_type='radar',
            title='Test Radar'
        )
        
        assert fig is not None
        assert fig.layout.title.text == 'Test Radar'
        assert len(fig.data) == 2  # Two models
    
    def test_parallel_comparison(self, sample_accuracy):
        """Test parallel coordinates comparison."""
        fig = plot_model_comparison_matrix(
            accuracy_df=sample_accuracy,
            plot_type='parallel',
            title='Test Parallel'
        )
        
        assert fig is not None
        assert fig.layout.title.text == 'Test Parallel'
        assert len(fig.data) > 0
    
    def test_metric_filtering(self, sample_accuracy):
        """Test filtering by metrics."""
        fig = plot_model_comparison_matrix(
            accuracy_df=sample_accuracy,
            metrics=['rmse', 'mae'],
            plot_type='heatmap'
        )
        
        # Check that only selected metrics are included
        assert fig.layout.xaxis.ticktext == ('rmse', 'mae')
    
    def test_model_filtering(self, sample_accuracy):
        """Test filtering by models."""
        fig = plot_model_comparison_matrix(
            accuracy_df=sample_accuracy,
            models=['linear'],
            plot_type='heatmap'
        )
        
        # Check that only selected model is included
        assert fig.layout.yaxis.ticktext == ('linear',)
    
    def test_matplotlib_engine(self, sample_accuracy):
        """Test matplotlib engine for heatmap."""
        import matplotlib.pyplot as plt
        
        fig = plot_model_comparison_matrix(
            accuracy_df=sample_accuracy,
            plot_type='heatmap',
            engine='matplotlib'
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_invalid_plot_type(self, sample_accuracy):
        """Test error handling for invalid plot type."""
        with pytest.raises(ValueError, match="Unknown plot_type"):
            plot_model_comparison_matrix(
                accuracy_df=sample_accuracy,
                plot_type='invalid'
            )


class TestComparisonReport:
    """Test comparison report generation."""
    
    def test_report_generation(self, sample_accuracy):
        """Test basic report generation."""
        report = create_comparison_report(
            accuracy_df=sample_accuracy,
            include_plots=['heatmap']
        )
        
        assert 'figures' in report
        assert 'summary_stats' in report
        assert 'rankings' in report
        assert 'html' in report
        
        # Check figures
        assert 'heatmap' in report['figures']
        
        # Check rankings
        assert 'avg_rank' in report['rankings'].columns
        assert len(report['rankings']) == 2  # Two models
    
    def test_report_with_html_output(self, sample_accuracy):
        """Test report generation with HTML output."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            report = create_comparison_report(
                accuracy_df=sample_accuracy,
                output_path=temp_path,
                include_plots=['heatmap', 'radar']
            )
            
            # Check file was created
            assert os.path.exists(temp_path)
            
            # Check HTML content
            with open(temp_path, 'r') as f:
                html_content = f.read()
            
            assert '<html>' in html_content
            assert 'Model Rankings' in html_content
            assert 'Summary Statistics' in html_content
            assert report['html'] is not None
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_report_all_plots(self, sample_accuracy):
        """Test report with all plot types."""
        report = create_comparison_report(
            accuracy_df=sample_accuracy,
            include_plots=['heatmap', 'radar', 'parallel']
        )
        
        assert len(report['figures']) == 3
        assert all(plot in report['figures'] for plot in ['heatmap', 'radar', 'parallel'])
    
    def test_report_with_filtering(self, sample_accuracy):
        """Test report with metric and model filtering."""
        report = create_comparison_report(
            accuracy_df=sample_accuracy,
            metrics=['rmse'],
            models=['linear'],
            include_plots=['heatmap']
        )
        
        # Check that filtering was applied
        stats = report['summary_stats']
        assert all(idx[1] == 'rmse' for idx in stats.index)
        assert all(idx[0] == 'linear' for idx in stats.index)


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 