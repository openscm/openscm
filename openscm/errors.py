class AdapterNeedsModuleError(Exception):
    """Exception raised when an adapter needs a module that is not installed."""


class InsufficientDataError(ValueError):
    """
    Exception raised when not enough data overlap is available when converting from one
    timeframe to another (e.g. when the target timeframe is outside the range of the
    source timeframe) or when data is too short (less than 3 data points).
    """


class ParameterError(Exception):
    """
    Exception relating to a parameter. Used as super class.
    """


class ParameterReadonlyError(ParameterError):
    """
    Exception raised when a requested parameter is read-only.

    This can happen, for instance, if a parameter's parent parameter
    in the parameter hierarchy has already been requested as writable.
    """


class ParameterEmptyError(ParameterError):
    """
    Exception raised when trying to read when a parameter's value hasn't been set
    """


class ParameterTypeError(ParameterError):
    """
    Exception raised when a parameter is of a different type than
    requested (scalar or timeseries).
    """


class ParameterReadError(ParameterError):
    """
    Exception raised when a parameter has been read from (raised, e.g., when attempting
    to create a child parameter).
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
