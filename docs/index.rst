.. modeltime-resample-py documentation master file

Welcome to modeltime-resample-py
=================================

A Python package for time series cross-validation, resampling, model fitting, and evaluation, inspired by the R ``modeltime.resample`` and ``rsample`` packages.

.. image:: https://img.shields.io/pypi/v/modeltime-resample-py.svg
   :target: https://pypi.python.org/pypi/modeltime-resample-py
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/modeltime-resample-py.svg
   :target: https://pypi.python.org/pypi/modeltime-resample-py
   :alt: Python Versions

.. image:: https://readthedocs.org/projects/modeltime-resample-py/badge/?version=latest
   :target: https://modeltime-resample-py.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/your_username/modeltime-resample-py/workflows/CI/badge.svg
   :target: https://github.com/your_username/modeltime-resample-py/actions
   :alt: CI Status

Key Features
------------

* **Time Series Splitting**: Create train/test splits respecting temporal order
* **Cross-Validation**: Rolling and expanding window cross-validation strategies
* **Flexible Period Specification**: Use integers or time-based strings (e.g., '6 months')
* **Model Evaluation**: Fit models to resamples and calculate performance metrics
* **Visualization**: Plot CV plans and model predictions
* **Convenience Functions**: High-level API for common workflows

Installation
------------

.. code-block:: bash

   pip install modeltime-resample-py

Quick Example
-------------

.. code-block:: python

   import pandas as pd
   import numpy as np
   from sklearn.linear_model import LinearRegression
   from modeltime_resample_py import evaluate_model

   # Create sample time series
   dates = pd.date_range('2020-01-01', periods=365, freq='D')
   data = pd.Series(np.random.randn(365).cumsum(), index=dates, name='value')

   # Evaluate model with time series cross-validation
   model = LinearRegression()
   results = evaluate_model(
       data=data,
       model=model,
       initial='6 months',
       assess='1 month',
       metrics=['mae', 'rmse']
   )

   # View average performance
   print(results.groupby('metric_name')['metric_value'].mean())

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   user_guide/splitting
   user_guide/modeling
   user_guide/metrics
   user_guide/visualization
   user_guide/convenience

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/visualization
   api/metrics
   api/plotting
   api/utils
   api/exceptions

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_usage
   examples/model_comparison
   examples/custom_metrics
   examples/advanced_visualization

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   migration_from_r

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 