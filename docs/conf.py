"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'modeltime-resample-py'
copyright = f'{datetime.now().year}, Your Name'
author = 'Your Name'
release = '0.0.1'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'nbsphinx',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns to exclude
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
}

# Add any paths that contain custom static files
html_static_path = ['_static']

# Intersphinx mapping to other documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# nbsphinx settings for Jupyter notebooks
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True  # Continue even if notebooks have errors

# Suppress warnings about duplicate labels
suppress_warnings = ['autosectionlabel.*'] 