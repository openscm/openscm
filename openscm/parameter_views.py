from typing import Sequence


from .parameters import _Parameter
from .timeframes import Timeframe, TimeframeConverter
from .units import UnitConverter
from .errors import ParameterEmptyError


class ParameterView:
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


class ScalarView(ParameterView):
    """
    Read-only view of a scalar parameter.
    """

    _child_data_views: Sequence["ScalarView"]
    """List of views to the child parameters for aggregated reads"""

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

        def get_data_views_for_children_or_parameter(
            parameter: _Parameter
        ) -> Sequence["ScalarView"]:
            if parameter._children:
                return sum(
                    (
                        get_data_views_for_children_or_parameter(p)
                        for p in parameter._children.values()
                    ),
                    [],
                )
            return [ScalarView(parameter, self._unit_converter._target)]

        super().__init__(parameter)
        self._unit_converter = UnitConverter(parameter._info._unit, unit)
        if self._parameter._children:
            self._child_data_views = get_data_views_for_children_or_parameter(
                self._parameter
            )

    def get(self) -> float:
        """
        Get current value of scalar parameter.

        If the parameter has child parameters (i.e. ``_children`` is not empty),
        the returned value will be the sum of the values of all of the child
        parameters.

        Raises
        ------
        ParameterEmptyError
            Parameter is empty, i.e. has not yet been written to
        """
        if self._parameter._children:
            return sum(v.get() for v in self._child_data_views)
        elif self.is_empty:
            raise ParameterEmptyError

        return self._unit_converter.convert_from(self._parameter._data)


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

    _child_data_views: Sequence["TimeseriesView"]
    """List of views to the child parameters for aggregated reads"""

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

        def get_data_views_for_children_or_parameter(
            parameter: _Parameter
        ) -> Sequence["TimeseriesView"]:
            if parameter._children:
                return sum(
                    (
                        get_data_views_for_children_or_parameter(p)
                        for p in parameter._children.values()
                    ),
                    [],
                )
            return [
                TimeseriesView(
                    parameter,
                    self._unit_converter._target,
                    self._timeframe_converter._target,
                )
            ]

        super().__init__(parameter)
        self._unit_converter = UnitConverter(parameter._info._unit, unit)
        self._timeframe_converter = TimeframeConverter(
            parameter._info._timeframe, timeframe
        )
        if self._parameter._children:
            self._child_data_views = get_data_views_for_children_or_parameter(
                self._parameter
            )

    def get_series(self) -> Sequence[float]:
        """
        Get values of the full timeseries.

        If the parameter has child parameters (i.e. ``_children`` is not empty),
        the returned value will be the sum of the values of all of the child
        parameters.

        Raises
        ------
        ParameterEmptyError
            Parameter is empty, i.e. has not yet been written to
        """
        if self._parameter._children:
            return sum(v.get_series() for v in self._child_data_views)
        elif self.is_empty:
            raise ParameterEmptyError

        return self._timeframe_converter.convert_from(
            self._unit_converter.convert_from(self._parameter._data)
        )

    def get(self, index: int) -> float:
        """
        Get value at a particular time.

        TODO implement

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

        TODO implement.

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
