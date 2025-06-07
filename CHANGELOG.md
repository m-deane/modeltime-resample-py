# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2024-XX-XX

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