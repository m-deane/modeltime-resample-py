# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-13

### Added
- **Package Refactoring and Cleanup**
  - Removed unnecessary archived R package files from `_archive/` directory
  - Cleaned up redundant demo files from root directory
  - Consolidated examples into organized `examples/` directory
  - Standardized package metadata and configuration

### Changed
- **Package Configuration**
  - Updated `pyproject.toml` with proper metadata and URLs
  - Improved package description and author information
  - Updated README.md with corrected installation instructions and badges
  - Removed redundant `USAGE_EXAMPLES.md` file

### Removed
- Entire `_archive/` directory containing R package files
- Redundant demo files: `app.py`, `dashboard_example.py`, `demo_separate_plots.py`, etc.
- Temporary files and system files (`.DS_Store`)
- Duplicate example files in examples directory

### Added - High Priority Items
- **Restructured Package Organization**
  - Moved `modeling.py` to `core/modeling.py`
  - Created `metrics/` module for accuracy calculations
  - Separated plotting functions into individual modules
  - Added `utils/` module for validation utilities
  - Improved module organization with proper `__init__.py` files

- **CI/CD Pipeline**
  - Added GitHub Actions workflow for automated testing
  - Configured testing matrix for Python 3.8-3.11
  - Integrated code coverage reporting
  - Added linting (flake8), formatting (black), and type checking (mypy)

- **Documentation Structure**
  - Created getting started guide
  - Added comprehensive docstrings
  - Prepared structure for Sphinx documentation

### Added - Medium Priority Items
- **Convenience Functions**
  - `evaluate_model()`: High-level function for model evaluation
  - `compare_models()`: Compare multiple models with same CV splits
  - `quick_cv_split()`: Simple wrapper for single train/test split

- **Error Handling**
  - Custom exception hierarchy with base `ModelTimeError`
  - Specific exceptions for data, split, model, and metric errors
  - Better error messages with context

- **Project Configuration**
  - Enhanced `pyproject.toml` with complete metadata
  - Added development dependencies and optional extras
  - Configured tools (black, isort, mypy, pytest, coverage)
  - Added pre-commit hooks configuration

### Added - Supporting Files
- `.gitignore`: Comprehensive Python gitignore
- `LICENSE`: MIT license
- `MANIFEST.in`: Package distribution configuration
- `setup.py`: Minimal setup for backward compatibility
- `.pre-commit-config.yaml`: Code quality automation

### Changed
- Updated imports in main `__init__.py` to reflect new structure
- Exposed convenience functions and exceptions in public API
- Improved dependency specifications

### Technical Improvements
- Better separation of concerns
- Cleaner module interfaces
- More maintainable code structure
- Ready for PyPI distribution

## Future Enhancements (Low Priority)
The following items are prepared for future implementation:
- Parallel processing for `fit_resamples`
- Caching for repeated calculations
- Progress bars for long operations
- Residual diagnostic plots
- Model comparison plots
- Panel data support
- Advanced resampling methods (Monte Carlo CV)
- Better scikit-learn pipeline integration
- Deep learning framework support
- Dask support for large datasets 