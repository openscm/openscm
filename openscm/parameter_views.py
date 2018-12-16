from typing import Sequence, Tuple
from .parameters import _Parameter, ParameterType

class ParameterView:
    """
    Generic view to a :ref:`parameter <parameters>` (scalar or timeseries).
    """

    _name: Tuple[str]
    """:ref:`Hierarchical name <parameter-hierarchy>`"""

    _region: Tuple[str]
    """Hierarchical region name"""

    _unit: str
    """Unit"""

    def __init__(self, name: Tuple[str], region: Tuple[str], unit: str):
        """
        Initialize.

        Parameters
        ----------
        name
            :ref:`Hierarchical name <parameter-hierarchy>`
        region
            Hierarchical region name
        unit
            Unit
        """
        self._name = name
        self._region = region
        self._unit = unit

    @property
    def name(self) -> Tuple[str]:
        """
        :ref:`Hierarchical name <parameter-hierarchy>`
        """
        return self._name

    @property
    def region(self) -> Tuple[str]:
        """
        Hierarchical region name
        """
        return self._region

    @property
    def unit(self) -> str:
        """
        Unit
        """
        return self._unit

    @property
    def is_empty(self) -> bool:
        """
        Check if parameter is empty, i.e. has not yet been written to.
        """
        raise NotImplementedError


class ParameterInfo(ParameterView):
    """
    Provides information about a :ref:`parameter <parameters>`.
    """

    _type: ParameterType
    """Parameter type"""

    def __init__(self, parameter: _Parameter):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter
        """
        self._type = parameter.parameter_type

    @property
    def parameter_type(self) -> ParameterType:
        """Parameter type"""
        return self._type


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

    def get_series(self) -> Sequence[float]:
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
