"""Custom exceptions for modeltime_resample_py."""


class ModelTimeError(Exception):
    """Base exception for modeltime_resample_py."""
    pass


class DataValidationError(ModelTimeError):
    """Raised when input data is invalid."""
    pass


class DateValidationError(DataValidationError):
    """Raised when date/time index validation fails."""
    pass


class SplitValidationError(ModelTimeError):
    """Raised when CV splits are invalid."""
    pass


class ModelValidationError(ModelTimeError):
    """Raised when model specification is invalid."""
    pass


class MetricCalculationError(ModelTimeError):
    """Raised when metric calculation fails."""
    pass


class PlottingError(ModelTimeError):
    """Raised when plotting operations fail."""
    pass 