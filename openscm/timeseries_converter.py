"""
Different climate models often use different time frames for their input and output
data. This includes different 'meanings' of time steps (e.g. beginning vs middle of year) and
different lengths of the time steps (e.g. years vs months). Accordingly, OpenSCM
supports the conversion of timeseries data between such timeseries, which is handled in
this module. A thorough explaination of the procedure used is given in a dedicated
`Jupyter Notebook
<https://github.com/openclimatedata/openscm/blob/master/notebooks/timeseries.ipynb>`_.
"""

from enum import Enum
from typing import Any, Callable, Dict

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

from .errors import InsufficientDataError
from .parameters import ParameterType

# pylint: disable=too-many-arguments


class ExtrapolationType(Enum):
    """
    Extrapolation type.
    """

    NONE = -1
    CONSTANT = 0
    LINEAR = 1
    # TODO support CUBIC = 3


class InterpolationType(Enum):
    """
    Interpolation type.
    """

    LINEAR = 1
    # TODO support CUBIC = 3


def create_time_points(
    start_time: int, period_length: int, points_num: int, timeseries_type: ParameterType
) -> np.ndarray:
    """
    Create time points for an equi-distant time series.

    Parameters
    ----------
    start_time
        First time point of the timeseries (seconds since ``1970-01-01 00:00:00``)
    period_length
        Period length (in seconds)
    points_num
        Length of timeseries
    timeseries_type
        Timeseries type

    Returns
    -------
    np.ndarray
        Array of the timeseries time points
    """
    points_num_output = (
        (points_num + 1)  # +1 for averages as we need to give the full last interval
        if timeseries_type == ParameterType.AVERAGE_TIMESERIES
        else points_num
    )
    end_time_output = start_time + (points_num_output - 1) * period_length
    return np.linspace(start_time, end_time_output, points_num_output)


def _calc_interval_averages(
    continuous_representation: Callable[[float], float], target_intervals: np.ndarray
) -> np.ndarray:
    """
    Calculate the interval averages of a continuous function.

    Here interval average is calculated as the integral over the period divided by
    the period length.

    Parameters
    ----------
    continuous_representation
        Continuous function from which to calculate the interval averages. Should be
        calculated using
        :func:`openscm.timeseries_converter.TimeseriesConverter._calc_continuous_representation`.
    target_intervals
        Intervals to calculate the average of.

    Returns
    -------
    np.ndarray
        Array of the interval/period averages
    """
    # TODO: numerical integration here could be very expensive
    # TODO: update to include caching and/or analytic solutions depending on
    # TODO:     interpolation choice
    int_averages = [np.nan] * len(target_intervals[:-1])
    for i, l in enumerate(target_intervals[:-1]):
        u = target_intervals[i + 1]
        y, _ = integrate.quad(continuous_representation, l, u)
        int_averages[i] = y / (u - l)

    return np.array(int_averages)


def _calc_integral_preserving_linear_interpolation(values: np.ndarray) -> np.ndarray:
    """
    Calculate the "linearization" values of the array ``values`` which is assumed to
    represent averages over time periods. Values at the edges of the periods are
    taken as the average of adjacent periods, values at the period middles are taken
    such that the integral over a period is the same as for the input data.

    Parameters
    ----------
    values
        Timeseries values of period averages

    Returns
    -------
    np.ndarray
        Values of linearization (of length ``2 * len(values) + 1``)
    """
    edge_point_values = (values[1:] + values[:-1]) / 2
    middle_point_values = (
        4 * values[1:-1] - edge_point_values[:-1] - edge_point_values[1:]
    ) / 2
    # values = (
    #   1 / 2 * (edges_lower + middle_point_values) * 1 /2
    #   + 1 / 2 * (middle_point_values + edges_upper)
    # ) / 2
    first_edge_point_value = (
        2 * values[0] - edge_point_values[0]
    )  # values[0] = (first_edge_point_value + edge_point_values[0]) / 2
    last_edge_point_value = (
        2 * values[-1] - edge_point_values[-1]
    )  # values[-1] = (last_edge_point_value + edge_point_values[-1]) / 2
    return np.concatenate(
        (
            np.array(
                [
                    np.concatenate(([first_edge_point_value], edge_point_values)),
                    np.concatenate(([values[0]], middle_point_values, [values[-1]])),
                ]
            ).T.reshape(2 * len(values)),
            [last_edge_point_value],
        )
    )


class TimeseriesConverter:
    """
    Converts timeseries and their points between two timeseriess (each defined by a time
    of the first point and a period length).
    """

    _source: np.ndarray
    """Source timeseries time points"""

    _target: np.ndarray
    """Target timeseries time points"""

    _timeseries_type: ParameterType
    """Time series type"""

    _interpolation_type: InterpolationType
    """Interpolation type"""

    _extrapolation_type: ExtrapolationType
    """Extrapolation type"""

    def __init__(
        self,
        source_time_points: np.ndarray,
        target_time_points: np.ndarray,
        timeseries_type: ParameterType,
        interpolation_type: InterpolationType,
        extrapolation_type: ExtrapolationType,
    ):
        """
        Initialize.

        Parameters
        ----------
        source_time_points
            Source timeseries time points
        target_time_points
            Target timeseries time points
        timeseries_type
            Time series type
        interpolation_type
            Interpolation type
        extrapolation_type
            Extrapolation type
        """
        self._source = np.array(source_time_points, copy=True)
        self._target = np.array(target_time_points, copy=True)
        self._timeseries_type = timeseries_type
        self._interpolation_type = interpolation_type
        self._extrapolation_type = extrapolation_type

        if (
            source_time_points[0] > target_time_points[1]
        ):  # TODO Consider extrapolation type
            raise InsufficientDataError

    def _calc_continuous_representation(
        self, time_points: np.ndarray, values: np.ndarray
    ) -> Callable[[float], float]:
        """
        Calculate a "continuous" representation of a timeseries (see
        :func:`openscm.timeseries_converter._calc_integral_preserving_linear_interpolation`)
        with the time points ``time_points`` and values ``values``.

        Parameters
        ----------
        time_points
            Time points of the timeseries
        values
            Values of the timeseries

        Returns
        -------
        func
            Function that represents the interpolated timeseries. It takes a single
            argument, time ("x-value"), and returns a single float, the value of the
            interpolated timeseries at that point in time ("y-value").
        """
        if (self._timeseries_type == ParameterType.AVERAGE_TIMESERIES) and (
            self._interpolation_type == InterpolationType.LINEAR
        ):
            # our custom implementation of a mean preserving linear interpolation
            linearization_points = (
                np.concatenate(
                    (
                        # [time_points[0] - (time_points[1] - time_points[0]) / 2],
                        time_points,
                        (time_points[1:] + time_points[:-1]) / 2,
                        [0],
                    )
                )
                .reshape((2, len(time_points)))
                .T.flatten()[:-1]
            )
            linearization_values = _calc_integral_preserving_linear_interpolation(
                values
            )
            res_average = interpolate.interp1d(
                linearization_points,
                linearization_values,
                kind=self._get_scipy_interpolation_arg(),
                **self._get_scipy_extrapolation_args(values),
            )  # type: Callable[[float], float]
            return res_average

        if self._timeseries_type == ParameterType.POINT_TIMESERIES:
            res_point = interpolate.interp1d(
                time_points,
                values,
                kind=self._get_scipy_interpolation_arg(),
                **self._get_scipy_extrapolation_args(values),
            )  # type: Callable[[float], float]
            return res_point

        raise NotImplementedError

    def _get_scipy_extrapolation_args(self, values: np.ndarray) -> Dict[str, Any]:
        if self._extrapolation_type == ExtrapolationType.LINEAR:
            return {"fill_value": "extrapolate"}
        if self._extrapolation_type == ExtrapolationType.CONSTANT:
            return {"fill_value": (values[0], values[-1]), "bounds_error": False}
        # TODO: add cubic support
        return {}

    def _get_scipy_interpolation_arg(self) -> str:
        if self._interpolation_type == InterpolationType.LINEAR:
            return "linear"
        # TODO: add cubic support
        raise NotImplementedError

    def _convert(
        self,
        values: np.ndarray,
        source_time_points: np.ndarray,
        target_time_points: np.ndarray,
    ) -> np.ndarray:
        """
        Wrap ``self._convert_unsafe`` to provide proper error handling.

        ``self._convert_unsafe`` converts time period average timeseries data
        ``values`` for timeseries time points ``source_time_points`` to the time
        points ``target_time_points``.

        Parameters
        ----------
        values
            Array of data to convert
        source_time_points
            Source timeseries time points
        target_time_points
            Target timeseries time points

        Raises
        ------
        InsufficientDataError
            Length of the time series is too short to convert

        InsufficientDataError
            Target time points are outside the source time points and
            ``self._extrapolation_type`` is ``ExtrapolationType.None``

        Returns
        -------
        np.ndarray
            Converted time period average data for timeseries ``target``
        """
        if len(values) < 3:
            raise InsufficientDataError

        try:
            return self._convert_unsafe(values, source_time_points, target_time_points)
        except ValueError:
            error_msg = (
                "Target time points are outside the source time points, use an "
                "extrapolation type other than None"
            )
            raise InsufficientDataError(error_msg)

    def _convert_unsafe(
        self,
        values: np.ndarray,
        source_time_points: np.ndarray,
        target_time_points: np.ndarray,
    ) -> np.ndarray:
        """
        Convert time period average timeseries data ``values`` for timeseries time
        points ``source_time_points`` to the time points ``target_time_points``.

        Parameters
        ----------
        values
            Array of data to convert
        source_time_points
            Source timeseries time points
        target_time_points
            Target timeseries time points

        Raises
        ------
        NotImplementedError
            The timeseries type is not recognised

        Returns
        -------
        np.ndarray
            Converted time period average data for timeseries ``target``
        """
        if self._timeseries_type == ParameterType.AVERAGE_TIMESERIES:
            return _calc_interval_averages(
                self._calc_continuous_representation(source_time_points, values),
                target_time_points,
            )
        if self._timeseries_type == ParameterType.POINT_TIMESERIES:
            return self._calc_continuous_representation(source_time_points, values)(
                target_time_points
            )

        raise NotImplementedError

    def convert_from(self, values: np.ndarray) -> np.ndarray:
        """
        Convert value **from** source timeseries time points to target timeseries time
        points.

        Parameters
        ----------
        values
            Value

        Returns
        -------
        np.ndarray
            Converted array
        """
        return self._convert(values, self._source, self._target)

    def convert_to(self, values: np.ndarray) -> np.ndarray:
        """
        Convert value from target timeseries time points **to** source timeseries time
        points.

        Parameters
        ----------
        values
            Value

        Returns
        -------
        np.ndarray
            Converted array
        """
        return self._convert(values, self._target, self._source)

    @property
    def source_length(self) -> int:
        """
        Length of source timeseries
        """
        return len(self._source) - (
            1 if self._timeseries_type == ParameterType.AVERAGE_TIMESERIES else 0
        )

    @property
    def target_length(self) -> int:
        """
        Length of target timeseries
        """
        return len(self._target) - (
            1 if self._timeseries_type == ParameterType.AVERAGE_TIMESERIES else 0
        )
