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
    requested.
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


class OutOfBoundsError(IndexError):
    """Error raised when the user attempts to step a model beyond its input data range.
    """

    pass


class OverwriteError(AssertionError):
    """Error raised when the user's action will overwrite existing data
    """

    pass


class ArrayLengthError(ParameterError):
    """
    Exception raised when a user attempts to set an array parameter with an array of a
    different length to what has already been set.
    """


class ModelNotInitialisedError(Exception):
    """
    Exception raised when a model is being used before being initiliased
    """
