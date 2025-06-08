"""
Modeltime Resample for Python
=============================

A Python implementation of time series model resampling and evaluation,
inspired by the R modeltime.resample package.

This package provides tools for:
- Time series cross-validation
- Model performance evaluation across multiple time slices
- Parallel processing of large-scale time series models
- Advanced visualization and comparison tools
"""

__version__ = '0.2.0'

# Core functionality - using actual existing modules
from .core.splits import time_series_split, time_series_cv
from .core.modeling import fit_resamples

# Metrics
from .metrics.accuracy import resample_accuracy

# Plotting functionality
from .plot.resamples import plot_resamples
from .plot.plan import plot_time_series_cv_plan

# Convenience functions
from .convenience import evaluate_model, compare_models

# Advanced visualization
from .visualization.dashboard import (
    create_interactive_dashboard,
    EnhancedResamplesDashboard
)
from .visualization.comparison import (
    plot_model_comparison_matrix,
    create_comparison_report
)

# Import new classes that we'll need to create
try:
    from .core.data_prep import TimeSeriesDataPrep
except ImportError:
    TimeSeriesDataPrep = None

try:
    from .parallel import (
        ParallelResampler,
        fit_resamples_parallel,
        resample_accuracy_parallel
    )
except ImportError:
    ParallelResampler = None
    fit_resamples_parallel = None
    resample_accuracy_parallel = None

__all__ = [
    # Core
    'time_series_split',
    'time_series_cv',
    'fit_resamples',
    'resample_accuracy',
    # Plotting
    'plot_resamples',
    'plot_time_series_cv_plan',
    # Convenience
    'evaluate_model',
    'compare_models',
    # Advanced visualization
    'create_interactive_dashboard',
    'EnhancedResamplesDashboard',
    'plot_model_comparison_matrix',
    'create_comparison_report'
]

# Add optional imports if available
if TimeSeriesDataPrep is not None:
    __all__.append('TimeSeriesDataPrep')

if ParallelResampler is not None:
    __all__.extend(['ParallelResampler', 'fit_resamples_parallel', 'resample_accuracy_parallel']) 