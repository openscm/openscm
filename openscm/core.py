"""
Then OpenSCM low-level API includes the basic functionality to run a
particular simple climate model with OpenSCM as well as
setting/getting its parameter values. Mapping of parameter names and
units is done internally.
"""

from typing import NewType, Sequence, Tuple

# TODO Make proper types where necessary
Region = str
Time = float
Unit = str


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
    unit
        Unit
    """

    def __init__(self, name: Tuple[str], unit: Unit):
        self.name = name
        self.unit = unit


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

    def set(value: float) -> None:
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

    def get() -> Sequence[float]:
        """
        Get values of the full timeseries.
        """
        raise NotImplementedError

    def get(time: Time) -> float:
        """
        Get value at a particular time.

        Parameters
        ----------
        time
            Time

        Raises
        ------
        IndexError
            ``time`` is out of run time range.
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
            array/...) of ``float`` values must equal run size.

        Raises
        ------
        ParameterLengthError
            Length of ``values`` does not equal run size.
        """
        raise NotImplementedError

    def set(self, value: float, time: Time) -> None:
        """
        Set value for a particular time in the time series.

        Parameters
        ----------
        value
            Value
        time
            Time

        Raises
        ------
        IndexError
            ``time`` is out of run time range.
        """
        raise NotImplementedError


class ParameterSet:
    """
    Collates a set of parameters.
    """

    def get_scalar_view(name: Tuple[str], region: Region, unit: Unit) -> ScalarView:
        """
        Get a read-only view to a scalar parameter.

        The parameter is created as a scalar if not viewed so far.

        Parameters
        ----------
        name
            Hierarchy name of the parameter
        region
            Region
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

    def get_writable_scalar_view(name: Tuple[str], region: Region, unit: Unit) -> WritableScalarView:
        """
        Get a writable view to a scalar parameter.

        The parameter is created as a scalar if not viewed so far.

        Parameters
        ----------
        name
            Hierarchy name of the parameter
        region
            Region
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

    def get_timeseries_view(name: Tuple[str], region: Region, unit: Unit) -> TimeseriesView:
        """
        Get a read-only view to a timeseries parameter.

        The parameter is created as a timeseries if not viewed so far.

        Parameters
        ----------
        name
            Hierarchy name of the parameter
        region
            Region
        unit
            Unit for the values in the view

        Raises
        ------
        ParameterTypeError
            Parameter is not timeseries
        ValueError
            Name not given or invalid region
        """
        raise NotImplementedError

    def get_writable_timeseries_view(name: Tuple[str], region: Region, unit: Unit) -> WritableTimeseriesView:
        """
        Get a writable view to a timeseries parameter.

        The parameter is created as a timeseries if not viewed so far.

        Parameters
        ----------
        name
            Hierarchy name of the parameter
        region
            Region
        unit
            Unit for the values in the view

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

    def get_parameter_info(name: Tuple[str]) -> ParameterInfo:
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


class Core:
    """
    OpenSCM core class.

    Represents a model run with a particular simple climate model.

    Parameters
    ----------
    model
        Name of the SCM to run
    start_time
        Beginning of the time range to run over
    end_time
        End of the time range to run over

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

    def __init__(self, model: str, start_time: Time, stop_time: Time):
        self.model = model
        self.start_time = start_time
        self.stop_time = stop_time
        self.parameters = ParameterSet()

    def run(self) -> None:
        """
        Run the model over the full time range.
        """
        raise NotImplementedError

    def step(self) -> Time:
        """
        Do a single time step.

        Returns
        -------
        Time
            Current time
        """
        raise NotImplementedError
