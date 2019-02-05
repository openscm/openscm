from typing import Sequence
from abc import ABCMeta, abstractmethod


from .parameters import _Parameter
from .timeframes import Timeframe, TimeframeConverter
from .units import UnitConverter
from .errors import ParameterEmptyError, ArrayLengthError


class ParameterView(metaclass=ABCMeta):
    """
    Generic view to a :ref:`parameter <parameters>` (scalar or timeseries).
    """

    _parameter: _Parameter
    """Parameter"""

    def __init__(self, parameter: _Parameter):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter
        """
        self._parameter = parameter

    @property
    def is_empty(self) -> bool:
        """
        Check if parameter is empty, i.e. has not yet been written to.
        """
        return not self._parameter._has_been_written_to

    def _attempt_aggregate_child_data(self) -> None:
        """
        Attempt to aggregate data in child parameters

        Raises
        ------
        ParameterEmptyError
            The parameter has no children to aggregate and hence is empty
        """
        if not self._parameter._children:
            raise ParameterEmptyError

        return self._sum_child_data_views(self._get_child_data_views())

    def _get_child_data_views(self) -> list:
        """
        Get a list of child views

        Returns
        -------
        list
            List of views into child parameters
        """
        views = []
        for cp in self._parameter._children.values():
            views.append(self._get_child_view(cp))

        return views

    # how do you type hints, return type is self...
    @abstractmethod
    def _get_child_view(self, child_parameter: _Parameter):
        """
        Get view of child data with desired units, timeframe etc.

        Parameters
        ----------
        child_parameter
            Parameter to get a view into

        Returns
        -------
        ParameterView
            View of child parameter
        """
        pass

    @abstractmethod
    def _sum_child_data_views(self, child_data_views: list):
        """
        Sum child data views

        Parameters
        ----------
        child_data_views
            List of child data views to sum

        Returns
        -------
        float, Sequence[float]
            Sum of child data
        """
        pass


class ScalarView(ParameterView):
    """
    Read-only view of a scalar parameter.
    """

    _unit_converter: UnitConverter
    """Unit converter"""

    def __init__(self, parameter: _Parameter, unit: str):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter
        unit
            Unit for the values in the view
        """
        super().__init__(parameter)
        self._unit_converter = UnitConverter(parameter._info._unit, unit)

    def get(self) -> float:
        """
        Get current value of scalar parameter.

        If the parameter has child parameters (aka ``_children`` is not empty),
        the returned value will be the sum of the values of all of the child
        parameters.
        """
        if self.is_empty:
            return self._attempt_aggregate_child_data()

        return self._unit_converter.convert_from(self._parameter._data)

    def _get_child_view(self, child_parameter: _Parameter) -> ParameterView:
        return type(self)(child_parameter, self._unit_converter._target)

    def _sum_child_data_views(self, child_data_views: list) -> float:
        data = 0
        for v in child_data_views:
            data += v.get()

        return data


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
        self._parameter._data = self._unit_converter.convert_to(value)


class BooleanView(ParameterView):
    """
    Read-only view of a boolean parameter.
    """

    def __init__(self, parameter: _Parameter):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter
        """
        super().__init__(parameter)

    def get(self) -> bool:
        """
        Get current value of boolean parameter.
        """
        return self._parameter._data


class WritableBooleanView(BooleanView):
    """
    View of a boolean parameter whose value can be changed.
    """

    def set(self, value: bool) -> None:
        """
        Set current value of boolean parameter.

        Parameters
        ----------
        value
            Value
        """
        self._parameter._data = value


class ArrayView(ParameterView):
    """
    Read-only view of an array parameter.
    """

    _unit_converter: UnitConverter
    """Unit converter"""

    def __init__(self, parameter: _Parameter, unit: str):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter
        unit
            Unit for the values in the view
        """
        super().__init__(parameter)
        self._unit_converter = UnitConverter(parameter._info._unit, unit)

    def get(self) -> float:
        """
        Get current value of an array parameter.
        """
        return self._unit_converter.convert_from(self._parameter._data)


class WritableArrayView(ArrayView):
    """
    View of an array parameter whose value can be changed.
    """

    def set(self, value: float) -> None:
        """
        Set current value of array parameter.

        Parameters
        ----------
        value
            Value
        """
        ld = self._parameter._data.size
        if (ld != 0) and (ld != value.size):
            raise ArrayLengthError
        self._parameter._data = self._unit_converter.convert_to(value)


class TimeseriesView(ParameterView):
    """
    Read-only :class:`ParameterView` of a timeseries.
    """

    _timeframe_converter: TimeframeConverter
    """Timeframe converter"""

    _unit_converter: UnitConverter
    """Unit converter"""

    def __init__(self, parameter: _Parameter, unit: str, timeframe: Timeframe):
        """
        Initialize.

        Parameters
        ----------
        parameter
            Parameter
        unit
            Unit for the values in the view
        timeframe
            Timeframe
        """
        super().__init__(parameter)
        self._unit_converter = UnitConverter(parameter._info._unit, unit)
        self._timeframe_converter = TimeframeConverter(
            parameter._info._timeframe, timeframe
        )

    def get_series(self) -> Sequence[float]:
        """
        Get values of the full timeseries.

        If the parameter has child parameters (aka ``_children`` is not empty),
        the returned value will be the sum of the values of all of the child
        parameters.
        """
        if self.is_empty:
            return self._attempt_aggregate_child_data()

        return self._timeframe_converter.convert_from(
            self._unit_converter.convert_from(self._parameter._data)
        )

    def _get_child_view(self, child_parameter: _Parameter) -> ParameterView:
        return type(self)(
            child_parameter,
            self._unit_converter._target,
            self._timeframe_converter._target,
        )

    def _sum_child_data_views(self, child_data_views: list) -> Sequence[float]:
        data = 0
        for v in child_data_views:
            data += v.get_series()

        return data

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

    @property
    def length(self) -> int:
        """
        Length of timeseries.
        """
        return self._timeframe_converter.get_target_len(len(self._parameter._data))


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
            Values to set.
        """
        self._parameter._data = self._timeframe_converter.convert_to(
            self._unit_converter.convert_to(values)
        )

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
