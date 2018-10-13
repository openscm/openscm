"""
Then OpenSCM low-level API includes the basic functionality to run a
particular simple climate model with OpenSCM as well as
setting/getting its parameter values. Mapping of parameter names and
units is done internally.

Parts of this API definition seems unpythonic as it is designed to
be easily implementable in several programming languages.
"""

from typing import Sequence, Tuple


class ParameterLengthError(Exception):
    """
    Exception raised when sequences in timeseries do not match run
    size.
    """


class ParameterReadonlyError(Exception):
    """
    Exception raised when a requested parameter is read-only.

    This can happen, for instance, if a parameter's parent parameter
    in the parameter hierarchy has already been requested as writable.
    """


class ParameterTypeError(Exception):
    """
    Exception raised when a parameter is of a different type than
    requested (scalar or timeseries).
    """


class ParameterView:
    """
    Generic view to a parameter (scalar or timeseries).

    Parameters
    ----------
    name
        Hierarchical name
    region
        Region (hierarchy)
    unit
        Unit

    Attributes
    ----------
    name
        Hierarchical name (read-only)
    region
        Region (hierarchy) (read-only)
    unit
        Unit (read-only)
    """

    def __init__(self, name: Tuple[str], region: Tuple[str], unit: str):
        self.name = name
        self.region = region
        self.unit = unit

    @property
    def is_empty(self) -> bool:
        """
        Check if parameter is empty, i.e. has not yet been written to.
        """
        raise NotImplementedError


class ParameterInfo(ParameterView):
    """
    Provides information about a parameter.

    Attributes
    ----------
    parameter_type
        Type (``"scalar"`` or ``"timeseries"``)
    """

    parameter_type: str


class ScalarView(ParameterView):
    """
    Read-only view of a scalar parameter.
    """

    def get(self) -> float:
        """
        Get current value of scalar parameter.
        """
        raise NotImplementedError


class WritableScalarView(ScalarView):
    """
    View of a scalar parameter whose value can be changed.
    """

    def set(self, value: float) -> None:
        """
        Set current value of scalar parameter.

        Parameters
        ----------
        value
            Value
        """
        raise NotImplementedError


class TimeseriesView(ParameterView):
    """
    Read-only :class:`ParameterView` of a timeseries.
    """

    def get(self) -> Sequence[float]:
        """
        Get values of the full timeseries.
        """
        raise NotImplementedError

    def get(self, index: int) -> float:
        """
        Get value at a particular time.

        Parameters
        ----------
        index
            Time step index

        Raises
        ------
        IndexError
            ``time`` is out of run time range.
        """
        raise NotImplementedError
        
    def length(self) -> int:
        """
        Get length of time series.
        """
        raise NotImplementedError


class WritableTimeseriesView(TimeseriesView):
    """
    View of a timeseries whose values can be changed.
    """

    def set_series(self, values: Sequence[float]) -> None:
        """
        Set value for whole time series.

        Parameters
        ----------
        values
            Values to set. The length of this sequence (list/1-D
            array/...) of ``float`` values must equal size.

        Raises
        ------
        ParameterLengthError
            Length of ``values`` does not equal size.
        """
        raise NotImplementedError

    def set(self, value: float, index: int) -> None:
        """
        Set value for a particular time in the time series.

        Parameters
        ----------
        value
            Value
        index
            Time step index

        Raises
        ------
        IndexError
            ``index`` is out of range.
        """
        raise NotImplementedError


class ParameterSet:
    """
    Collates a set of parameters.
    """

    def get_scalar_view(self, name: Tuple[str], region: Tuple[str], unit: str) -> ScalarView:
        """
        Get a read-only view to a scalar parameter.

        The parameter is created as a scalar if not viewed so far.

        Parameters
        ----------
        name
            Hierarchy name of the parameter
        region
            Region (hierarchy)
        unit
            Unit for the values in the view

        Raises
        ------
        ParameterTypeError
            Parameter is not scalar
        ValueError
            Name not given or invalid region
        """
        raise NotImplementedError

    def get_writable_scalar_view(self, name: Tuple[str], region: Tuple[str], unit: str) -> WritableScalarView:
        """
        Get a writable view to a scalar parameter.

        The parameter is created as a scalar if not viewed so far.

        Parameters
        ----------
        name
            Hierarchy name of the parameter
        region
            Region (hierarchy)
        unit
            Unit for the values in the view

        Raises
        ------
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        ParameterTypeError
            Parameter is not scalar
        ValueError
            Name not given or invalid region
        """
        raise NotImplementedError

    def get_timeseries_view(self, name: Tuple[str], region: Tuple[str], unit: str, start_time: int, period_length: int) -> TimeseriesView:
        """
        Get a read-only view to a timeseries parameter.

        The parameter is created as a timeseries if not viewed so far.
        The length of the returned ParameterView's timeseries is adjusted such
        that its last value corresponds to a time not exceeding the ``end_time``
        of the underlying run (i.e. ``Core`` object).

        Parameters
        ----------
        name
            Hierarchy name of the parameter
        region
            Region (hierarchy)
        unit
            Unit for the values in the view
        start_time
            Time for first point in timeseries (seconds since 1970-01-01 00:00:00)
        period_length
            Length of single time step in seconds

        Raises
        ------
        ParameterTypeError
            Parameter is not timeseries
        ValueError
            Name not given or invalid region
        """
        raise NotImplementedError

    def get_writable_timeseries_view(self, name: Tuple[str], region: Tuple[str], unit: str, start_time: int, period_length: int) -> WritableTimeseriesView:
        """
        Get a writable view to a timeseries parameter.

        The parameter is created as a timeseries if not viewed so far.

        Parameters
        ----------
        name
            Hierarchy name of the parameter
        region
            Region (hierarchy)
        unit
            Unit for the values in the view
        start_time
            Time for first point in timeseries (seconds since 1970-01-01 00:00:00)
        period_length
            Length of single time step in seconds

        Raises
        ------
        ParameterReadonlyError
            Parameter is read-only (e.g. because its parent has been written to)
        ParameterTypeError
            Parameter is not timeseries
        ValueError
            Name not given or invalid region
        """
        raise NotImplementedError

    def get_parameter_info(self, name: Tuple[str]) -> ParameterInfo:
        """
        Get information about a parameter.

        Parameters
        ----------
        name
            Hierarchy name of the parameter

        Raises
        ------
        ValueError
            Name not given

        Returns
        -------
        ParameterInfo
            Information about the parameter or ``None`` if the parameter has not been created yet.
        """
        raise NotImplementedError

    def has_parameter(self, name: Tuple[str]) -> bool:
        """
        Query if parameter set has a specific parameter.

        Parameters
        ----------
        name
            Hierarchy name of the parameter

        Raises
        ------
        ValueError
            Name not given
        """
        raise NotImplementedError


class Core:
    """
    OpenSCM core class.

    Represents a model run with a particular simple climate model.

    Parameters
    ----------
    model
        Name of the SCM to run
    start_time
        Beginning of the time range to run over (seconds since 1970-01-01 00:00:00)
    end_time
        End of the time range to run over (including; seconds since 1970-01-01 00:00:00)

    Raises
    ------
    KeyError
        No adapter for SCM named ``model`` found
    ValueError
        ``end_time`` before ``start_time``

    Attributes
    ----------
    end_time
        End of the time range to run over (read-only)
    model
        Name of the SCM to run (read-only)
    parameterset
        Set of parameters for the run (read-only)
    start_time
        Beginning of the time range to run over (read-only)
    """

    parameters: ParameterSet

    def __init__(self, model: str, start_time: int, end_time: int):
        self.model = model
        self.start_time = start_time
        self.end_time = end_time
        self.parameters = ParameterSet()

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        raise NotImplementedError

    def step(self) -> int:
        """
        Do a single time step.

        Returns
        -------
        int
            Current time (seconds since 1970-01-01 00:00:00)
        """
        raise NotImplementedError
