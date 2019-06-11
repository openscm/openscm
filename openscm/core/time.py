"""
Different climate models often use different time frames for their input and output
data. This includes different 'meanings' of time steps (e.g. beginning vs middle of
year) and different lengths of the time steps (e.g. years vs months). Accordingly,
OpenSCM supports the conversion of timeseries data between such timeseries, which is
handled in this module. A thorough explaination of the procedure used is given in a
dedicated `Jupyter Notebook
<https://github.com/openclimatedata/openscm/blob/master/notebooks/timeseries.ipynb>`_.
"""

import datetime
from enum import Enum
from typing import Any, Callable, Dict, Sequence, Union

import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from dateutil import parser

from ..errors import InsufficientDataError
from .parameters import ParameterType

TARGET_TYPE = np.int64


class ExtrapolationType(Enum):
    """
    Extrapolation type.
    """

    NONE = -1
    CONSTANT = 0
    LINEAR = 1
    # TODO: support CUBIC = 3

    @classmethod
    def from_extrapolation_type(
        cls, extrapolation_type: Union["ExtrapolationType", str]
    ) -> "ExtrapolationType":
        """
        Get extrapolation type from :class:`ExtrapolationType` or string value.

        Parameters
        ----------
        extrapolation_type
            Value to convert to enum value (can be
            ``"none"``/:attr:`ExtrapolationType.NONE`,
            ``"constant"``/:attr:`ExtrapolationType.CONSTANT`, or
            ``"linear"``/:attr:`ExtrapolationType.LINEAR`)

        Returns
        -------
        ExtrapolationType
            Enum value
        """
        if isinstance(extrapolation_type, str):
            return cls[extrapolation_type.upper()]
        return extrapolation_type


class InterpolationType(Enum):
    """
    Interpolation type.
    """

    LINEAR = 1
    # TODO support CUBIC = 3

    @classmethod
    def from_interpolation_type(
        cls, interpolation_type: Union["InterpolationType", str]
    ) -> "InterpolationType":
        """
        Get interpolation type from :class:`InterpolationType` or string value.

        Parameters
        ----------
        interpolation_type
            Value to convert to enum value (can be
            ``"linear"``/:attr:`InterpolationType.LINEAR`)

        Returns
        -------
        InterpolationType
            Enum value
        """
        if isinstance(interpolation_type, str):
            return cls[interpolation_type.upper()]
        return interpolation_type


def _float_year_to_datetime(inp: float) -> np.datetime64:
    year = int(inp)
    fractional_part = inp - year
    return np.datetime64(  # pylint: disable=too-many-function-args
        year - 1970, "Y"
    ) + np.timedelta64(  # pylint: disable=too-many-function-args
        int(
            (
                datetime.datetime(year + 1, 1, 1) - datetime.datetime(year, 1, 1)
            ).total_seconds()
            * fractional_part
        ),
        "s",
    )


_ufunc_float_year_to_datetime = np.frompyfunc(_float_year_to_datetime, 1, 1)
_ufunc_str_to_datetime = np.frompyfunc(parser.parse, 1, 1)


def _parse_datetime(inp: np.ndarray) -> np.ndarray:
    try:
        return _ufunc_float_year_to_datetime(inp.astype(float))
    except (TypeError, ValueError):
        return _ufunc_str_to_datetime(inp)


def _format_datetime(dts: np.ndarray) -> np.ndarray:
    """
    Convert an array to an array of :class:`np.datetime64`.

    Parameters
    ----------
    dts
        Input to attempt to convert

    Returns
    -------
    :class:`np.ndarray` of :class:`np.datetime64`
        Converted array

    Raises
    ------
    ValueError
        If one of the values in :obj:`dts` cannot be converted to :class:`np.datetime64`
    """
    if len(dts) <= 0:  # pylint: disable=len-as-condition
        return np.array([], dtype="datetime64[s]")

    dtype = dts.dtype.type
    # if issubclass(dtype, np.object):
    dtype = np.dtype(type(dts[0])).type
    if issubclass(dtype, np.datetime64):
        return np.asarray(dts, dtype="datetime64[s]")
    if issubclass(dtype, np.floating):
        return _ufunc_float_year_to_datetime(dts).astype("datetime64[s]")
    if issubclass(dtype, np.integer):
        return (np.asarray(dts) - 1970).astype("datetime64[Y]").astype("datetime64[s]")
    if issubclass(dtype, str):
        return _parse_datetime(dts).astype("datetime64[s]")
    return np.asarray(dts, dtype="datetime64[s]")


class TimePoints:  # TODO: track type of timeseries
    """
    Handles time points by wrapping :class:`np.ndarray` of :class:`np.datetime64`..
    """

    _values: np.ndarray
    """Actual time points array"""

    def __init__(self, values: Sequence):
        """
        Initialize.

        Parameters
        ----------
        values
            Time points array to handle
        """
        self._values = _format_datetime(np.asarray(values))

    @property
    def values(self) -> np.ndarray:
        """
        Time points
        """
        return self._values

    def to_index(self) -> pd.Index:
        """
        Get time points as :class:`pd.Index`.

        Returns
        -------
        :class:`pd.Index`
            :class:`pd.Index` of :class:`np.dtype` :class:`object` with name ``"time"``
            made from the time points represented as :class:`datetime.datetime`.
        """
        return pd.Index(self._values.astype(object), dtype=object, name="time")

    def years(self) -> np.ndarray:
        """
        Get year of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Year of each time point
        """
        return np.vectorize(getattr)(self._values.astype(object), "year")

    def months(self) -> np.ndarray:
        """
        Get month of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Month of each time point
        """
        return np.vectorize(getattr)(self._values.astype(object), "month")

    def days(self) -> np.ndarray:
        """
        Get day of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Day of each time point
        """
        return np.vectorize(getattr)(self._values.astype(object), "day")

    def hours(self) -> np.ndarray:
        """
        Get hour of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Hour of each time point
        """
        return np.vectorize(getattr)(self._values.astype(object), "hour")

    def weekdays(self) -> np.ndarray:
        """
        Get weekday of each time point.

        Returns
        -------
        :obj:`np.array` of :obj:`int`
            Day of the week of each time point
        """
        return np.vectorize(datetime.datetime.weekday)(self._values.astype(object))


def create_time_points(  # TODO: replace by simpler function
    start_time: np.datetime64,
    period_length: np.timedelta64,
    points_num: int,
    timeseries_type: Union[ParameterType, str],
) -> np.ndarray:
    """
    Create time points for an equi-distant time series.

    Parameters
    ----------
    start_time
        First time point of the timeseries
    period_length
        Period length
    points_num
        Length of timeseries
    timeseries_type
        Timeseries type

    Returns
    -------
    :class:`np.ndarray` of :class:`np.datetime64`
        Array of the timeseries time points
    """
    timeseries_type = ParameterType.from_timeseries_type(timeseries_type)
    points_num_output = (
        (points_num + 1)  # +1 for averages as we need to give the full last interval
        if timeseries_type == ParameterType.AVERAGE_TIMESERIES
        else points_num
    )
    end_time_output = start_time + (points_num_output - 1) * period_length
    return np.linspace(
        start_time.astype("datetime64[s]").astype(float),
        end_time_output.astype("datetime64[s]").astype(float),
        points_num_output,
        dtype="datetime64[s]",
    )


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
    # TODO: update to include caching and/or analytic solutions depending on interpolation choice
    int_averages = [np.nan] * len(target_intervals[:-1])
    for i, l in enumerate(target_intervals[:-1]):
        u = target_intervals[i + 1]
        y, _ = integrate.quad(continuous_representation, l, u)
        int_averages[i] = y / (u - l)

    return np.array(int_averages)


def _calc_integral_preserving_linear_interpolation(values: np.ndarray) -> np.ndarray:
    """
    Calculate the "linearization" values of the array :obj:`values` which is assumed to
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

    def __init__(  # pylint: disable=too-many-arguments
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

        Raises
        ------
        InsufficientDataError
            Timeseries too short to extrapolate
        """
        self._source = np.array(source_time_points).astype(TARGET_TYPE, copy=True)
        self._target = np.array(target_time_points).astype(TARGET_TYPE, copy=True)
        self._timeseries_type = timeseries_type
        self._interpolation_type = interpolation_type
        self._extrapolation_type = extrapolation_type

        if self._source[0] > self._target[1]:  # TODO: consider extrapolation type
            raise InsufficientDataError

    def _calc_continuous_representation(
        # TODO: remove when NotImplementedError removed:
        # pylint: disable=missing-raises-doc
        self,
        time_points: np.ndarray,
        values: np.ndarray,
    ) -> Callable[[float], float]:
        """
        Calculate a "continuous" representation of a timeseries (see
        :func:`openscm.timeseries_converter._calc_integral_preserving_linear_interpolation`)
        with the time points :obj:`time_points` and values :obj:`values`.

        Parameters
        ----------
        time_points
            Time points of the timeseries
        values
            Values of the timeseries

        Returns
        -------
        Callable[[float], float]
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
        Wrap :func:`_convert_unsafe` to provide proper error handling.

        :func:`_convert_unsafe` converts time period average timeseries data
        :obj:`values` for timeseries time points :obj:`source_time_points` to the time
        points :obj:`target_time_points`.

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
            :attr:`_extrapolation_type` is :attr:`ExtrapolationType.NONE`

        Returns
        -------
        np.ndarray
            Converted time period average data for timeseries :obj:`values`
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
        Convert time period average timeseries data :obj:`values` for timeseries time
        points :obj:`source_time_points` to the time points :obj:`target_time_points`.

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
            Converted time period average data for timeseries :obj:`values`
        """
        if self._timeseries_type == ParameterType.AVERAGE_TIMESERIES:
            return _calc_interval_averages(
                self._calc_continuous_representation(
                    source_time_points.astype(TARGET_TYPE), values
                ),
                target_time_points.astype(TARGET_TYPE),
            )
        if self._timeseries_type == ParameterType.POINT_TIMESERIES:
            return self._calc_continuous_representation(
                source_time_points.astype(TARGET_TYPE), values
            )(target_time_points.astype(TARGET_TYPE))

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
