from typing import Sequence
from abc import ABCMeta, abstractmethod


from .parameters import _Parameter
from .timeframes import Timeframe, TimeframeConverter
from .units import UnitConverter
from .errors import ParameterEmptyError


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

    def _attempt_aggregate(self) -> None:
        """
        Attempt to aggregate data in child parameters

        Raises
        ------
        ParameterEmptyError
            The parameter has no children to aggregate and hence is empty
        """
        if not self._parameter._children:
            raise ParameterEmptyError

        self._parameter._data = self.sum_child_data(self._parameter)

    def sum_child_data(self, parameter: _Parameter):
        """
        Sum child data

        Parameters
        ----------
        parameter
            Parameter whose child data we want to sum

        Returns
        -------
        Sum of child parameter data
        """
        for _, cp in parameter._children.items():
            data_raw = cp._data if not cp._children else self.sum_child_data(cp)

            data_to_add = self._format_data(data_raw)

            try:
                data += data_to_add
            except NameError:
                data = data_to_add

        return data

    @abstractmethod
    def _format_data(self, data):
        """
        Format data e.g. convert to desired units, desired timeframe

        Parameters
        ----------
        data
            Data to format

        Returns
        -------
        Formatted data
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
            self._attempt_aggregate()

        return self._unit_converter.convert_from(self._parameter._data)

    def _format_data(self, data):
        return self._unit_converter.convert_from(data)


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
            self._attempt_aggregate()

        return self._timeframe_converter.convert_from(
            self._unit_converter.convert_from(self._parameter._data)
        )

    def _format_data(self, data):
        return self._timeframe_converter.convert_from(
            self._unit_converter.convert_from(data)
        )

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
