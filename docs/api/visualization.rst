Visualization
============

Advanced visualization utilities for interactive exploration and model comparison.

Interactive Dashboard
--------------------

.. automodule:: modeltime_resample_py.visualization.dashboard
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ResamplesDashboard
   :members:
   :undoc-members:
   :show-inheritance:

Model Comparison
---------------

.. automodule:: modeltime_resample_py.visualization.comparison
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Creating an Interactive Dashboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from modeltime_resample_py import (
       create_interactive_dashboard,
       fit_resamples,
       resample_accuracy
   )
   
   # Fit models and calculate accuracy
   results = fit_resamples(data_prep, models, ...)
   accuracy = resample_accuracy(results)
   
   # Create and launch dashboard
   dashboard = create_interactive_dashboard(
       resamples_df=results,
       accuracy_df=accuracy,
       title="Time Series Model Analysis"
   )
   
   # Run on localhost:8050
   dashboard.run(port=8050)

Model Comparison Matrix
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from modeltime_resample_py import plot_model_comparison_matrix
   
   # Create heatmap
   fig = plot_model_comparison_matrix(
       accuracy_df=accuracy,
       plot_type='heatmap',
       metrics=['rmse', 'mae', 'mape']
   )
   fig.show()
   
   # Create radar chart
   fig = plot_model_comparison_matrix(
       accuracy_df=accuracy,
       plot_type='radar'
   )
   fig.show()

Comprehensive Comparison Report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from modeltime_resample_py import create_comparison_report
   
   # Generate full HTML report
   report = create_comparison_report(
       accuracy_df=accuracy,
       output_path='model_comparison.html',
       include_plots=['heatmap', 'radar', 'parallel'],
       title='Time Series Model Comparison'
   )
   
   # Access report components
   print(report['rankings'])  # Model rankings
   print(report['summary_stats'])  # Summary statistics 