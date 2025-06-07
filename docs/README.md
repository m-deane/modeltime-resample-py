# Building the Documentation

This directory contains the source files for the modeltime-resample-py documentation.

## Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
# or
pip install -r docs/requirements.txt
```

## Building Locally

To build the documentation locally:

```bash
cd docs
make html
```

The built documentation will be in `docs/_build/html/`. Open `index.html` in your browser to view it.

## Other Formats

```bash
make latexpdf  # Build PDF
make epub      # Build EPUB
make clean     # Clean build files
```

## Live Reload During Development

For development with auto-reload:

```bash
pip install sphinx-autobuild
sphinx-autobuild . _build/html
```

Then navigate to http://localhost:8000

## Deploying to Read the Docs

1. Create an account at https://readthedocs.org
2. Import your GitHub repository
3. The `.readthedocs.yml` file will configure the build automatically
4. Documentation will be available at https://modeltime-resample-py.readthedocs.io

## Writing Documentation

- Use reStructuredText (.rst) format for documentation files
- Follow NumPy style for docstrings in Python code
- Place example notebooks in `docs/examples/`
- API documentation is auto-generated from docstrings

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── getting_started.md   # Getting started guide (Markdown)
├── api/                 # API reference (auto-generated)
├── user_guide/          # User guide sections
├── examples/            # Example notebooks and scripts
├── _static/             # Static files (CSS, images)
└── _templates/          # Custom templates
``` 