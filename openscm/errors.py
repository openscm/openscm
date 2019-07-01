"""
Errors/Exceptions defined and used in OpenSCM.
"""

from .core.units import (  # noqa: F401 # pylint: disable=unused-import
    DimensionalityError,
    UndefinedUnitError,
)


class AdapterNeedsModuleError(Exception):
    """Exception raised when an adapter needs a module that is not installed."""


class InsufficientDataError(ValueError):
    """
    Exception raised when not enough data is available to convert from one timeseries to
    another (e.g. when the target timeseries is outside the range of the source
    timeseries) or when data is too short (fewer than 3 data points).
    """


class ParameterError(Exception):
    """
    Exception relating to a parameter. Used as super class.
    """


class ParameterAggregationError(ParameterError):
    """
    Exception raised when a parameter is read from but has child parameters which cannot
    be aggregated (boolean or string parameters).
    """


class ParameterEmptyError(ParameterError):
    """
    Exception raised when trying to read when a parameter's value hasn't been set
    """


class ParameterReadError(ParameterError):
    """
    Exception raised when a parameter has been read from (raised, e.g., when attempting
    to create a child parameter).
    """


class ParameterReadonlyError(ParameterError):
    """
    Exception raised when a requested parameter is read-only.

    This can happen, for instance, if a parameter's parent parameter in the parameter
    hierarchy has already been requested as writable.
    """


class ParameterTypeError(ParameterError):
    """
    Exception raised when a parameter is of a different type than requested.
    """


class ParameterWrittenError(ParameterError):
    """
    Exception raised when a parameter has been written to (raised, e.g., when attempting
    to create a child parameter).
    """


class RegionAggregatedError(Exception):
    """
    Exception raised when a region has already been read from in a region-aggregated way.
    """


class TimeseriesPointsValuesMismatchError(IndexError):
    """
    Exception raised when the length of the values and of the time points of a
    timeseries mismatch (depending on the type of timeseries these must equal or deviate
    by one).
    """


class OutOfBoundsError(IndexError):
    """
    Error raised when the user attempts to step a model beyond its input data range.
    """


class OverwriteError(AssertionError):
    """
    Error raised when the user's action will overwrite existing data
    """


class ModelNotInitialisedError(Exception):
    """
    Exception raised when a model is being used before being initiliased
    """
