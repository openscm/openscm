"""
Different climate models often use different time frames for their input and output
data. This includes different 'meanings' of time steps (e.g. beginning vs middle of
year) and different lengths of the time steps (e.g. years vs months). Accordingly,
OpenSCM supports the conversion of timeseries data between such timeseries, which is
handled in this module. A thorough explaination of the procedure used is given in a
dedicated `Jupyter Notebook
<https://github.com/openclimatedata/openscm/blob/master/notebooks/timeseries.ipynb>`_.
"""

from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy.interpolate as interpolate

from ..errors import InsufficientDataError
from .parameters import ParameterType

"""numpy.dtype to do time conversions in"""
_TARGET_TYPE = np.int64


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


def _calc_linear_interval_averages(
    continuous_representation: Callable[[float], float],
    linearization_points: np.ndarray,
    target_intervals: np.ndarray,
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
    linearization_points
        Intervals used in the source data.
    target_intervals
        Intervals to calculate the average of.

    Returns
    -------
    np.ndarray
        Array of the interval/period averages
    """
    int_averages = [np.nan] * len(target_intervals[:-1])
    for i, l in enumerate(target_intervals[:-1]):
        u = target_intervals[i + 1]
        y = 0
        kink_points = np.concatenate(
            [
                [l],
                linearization_points[
                    np.where(
                        np.logical_and(
                            linearization_points > l, linearization_points < u
                        )
                    )
                ],
                [u],
            ]
        )

        for j, kink_point in enumerate(kink_points[:-1]):
            next_kink_point = kink_points[j + 1]
            left_edge = continuous_representation(kink_point)
            right_edge = continuous_representation(next_kink_point)
            dx = next_kink_point - kink_point
            y += (left_edge + right_edge) * dx / 2

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


@lru_cache()
def _calc_integral_preserving_linearization_points(time_points: Tuple) -> np.ndarray:
    time_points_array = np.array(time_points)
    return (
        np.concatenate(
            (
                # [time_points[0] - (time_points[1] - time_points[0]) / 2],
                time_points_array,
                (time_points_array[1:] + time_points_array[:-1]) / 2,
                [0],
            )
        )
        .reshape((2, len(time_points)))
        .T.flatten()[:-1]
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

    _timeseries_type_source: ParameterType
    """Timeseries type of the source"""

    _timeseries_type_target: ParameterType
    """Timeseries type of the target"""

    _interpolation_type: InterpolationType
    """Interpolation type"""

    _extrapolation_type: ExtrapolationType
    """Extrapolation type"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        source_time_points: np.ndarray,
        target_time_points: np.ndarray,
        timeseries_type_source: ParameterType,
        interpolation_type: InterpolationType,
        extrapolation_type: ExtrapolationType,
        timeseries_type_target: Optional[ParameterType] = None,
    ):
        """
        Initialize.

        Parameters
        ----------
        source_time_points
            Source timeseries time points
        target_time_points
            Target timeseries time points
        timeseries_type_source
            Timeseries type of the source
        interpolation_type
            Interpolation type
        extrapolation_type
            Extrapolation type
        timeseries_type_target
            Timeseries type of the target

        Raises
        ------
        InsufficientDataError
            Timeseries too short to extrapolate
        """
        self._source = np.array(source_time_points).astype(_TARGET_TYPE, copy=True)
        self._target = np.array(target_time_points).astype(_TARGET_TYPE, copy=True)
        self._timeseries_type_source = timeseries_type_source
        self._timeseries_type_target = (
            timeseries_type_target
            if timeseries_type_target is not None
            else timeseries_type_source
        )

        self._interpolation_type = interpolation_type
        self._extrapolation_type = extrapolation_type
        if not self.points_are_compatible(self._source, self._target):
            error_msg = (
                "Target time points are outside the source time points, use an "
                "extrapolation type other than None"
            )
            raise InsufficientDataError(error_msg)
        self.__inverter = None

    def points_are_compatible(
        self, source: Union[list, np.ndarray], target: Union[list, np.ndarray]
    ) -> bool:
        """
        Are the two sets of time points compatible i.e. can I convert between the two?

        Parameters
        ----------
        source
            Source timeseries time points
        target
            Target timeseries time points

        Returns
        -------
        bool
            Can I convert between the time points?
        """
        if self._extrapolation_type == ExtrapolationType.NONE:

            def _wrong_type(inp):
                return not isinstance(inp, np.ndarray) or inp.dtype != _TARGET_TYPE

            if _wrong_type(source):
                source = np.array(source).astype(_TARGET_TYPE)
            if _wrong_type(target):
                target = np.array(target).astype(_TARGET_TYPE)

            if source[0] > target[0] or source[-1] < target[-1]:
                return False

        return True

    def _calc_continuous_representation(
        self, time_points: np.ndarray, values: np.ndarray
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

        Raises
        ------
        NotImplementedError
            A conversion which has not yet been implemented is requested.
        """
        if (self._timeseries_type_source == ParameterType.AVERAGE_TIMESERIES) and (
            self._interpolation_type == InterpolationType.LINEAR
        ):
            # our custom implementation of an integral preserving linear interpolation
            linearization_points = _calc_integral_preserving_linearization_points(
                tuple(time_points.tolist())
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

        if self._timeseries_type_source == ParameterType.POINT_TIMESERIES:
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
        ValueError
            Timeseries conversion fails

        Returns
        -------
        np.ndarray
            Converted time period average data for timeseries :obj:`values`
        """
        if len(values) < 3:
            raise InsufficientDataError

        try:
            return self._convert_unsafe(values, source_time_points, target_time_points)
        except ValueError:  # pragma: no cover # emergency valve
            print("Timeseries conversion failed")
            raise

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
        if self._timeseries_type_source == ParameterType.AVERAGE_TIMESERIES:
            return self._convert_unsafe_average(
                values, source_time_points, target_time_points
            )

        if self._timeseries_type_source == ParameterType.POINT_TIMESERIES:
            return self._convert_unsafe_point(
                values, source_time_points, target_time_points
            )

        raise NotImplementedError

    def _convert_unsafe_average(
        self,
        values: np.ndarray,
        source_time_points: np.ndarray,
        target_time_points: np.ndarray,
    ) -> np.ndarray:
        if self._interpolation_type != InterpolationType.LINEAR:
            raise NotImplementedError  # pragma: no cover # emergency valve
        if self._timeseries_type_target == ParameterType.AVERAGE_TIMESERIES:
            return _calc_linear_interval_averages(
                self._calc_continuous_representation(
                    source_time_points.astype(_TARGET_TYPE), values
                ),
                _calc_integral_preserving_linearization_points(
                    tuple(source_time_points.astype(_TARGET_TYPE).tolist())
                ),
                target_time_points.astype(_TARGET_TYPE),
            )

        return self._calc_continuous_representation(
            source_time_points.astype(_TARGET_TYPE), values
        )(target_time_points.astype(_TARGET_TYPE))

    def _convert_unsafe_point(
        self,
        values: np.ndarray,
        source_time_points: np.ndarray,
        target_time_points: np.ndarray,
    ) -> np.ndarray:
        if self._timeseries_type_target == ParameterType.POINT_TIMESERIES:
            return self._calc_continuous_representation(
                source_time_points.astype(_TARGET_TYPE), values
            )(target_time_points.astype(_TARGET_TYPE))

        return _calc_linear_interval_averages(
            self._calc_continuous_representation(
                source_time_points.astype(_TARGET_TYPE), values
            ),
            source_time_points,
            target_time_points.astype(_TARGET_TYPE),
        )

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
        return self._inverter.convert_from(values)

    @property
    def _inverter(self):
        if self.__inverter is None:
            self.__inverter = self.__class__(
                self._target,
                self._source,
                self._timeseries_type_target,
                self._interpolation_type,
                self._extrapolation_type,
                self._timeseries_type_source,
            )
        return self.__inverter

    @property
    def source_length(self) -> int:
        """
        Length of source timeseries
        """
        return len(self._source) - (
            1 if self._timeseries_type_source == ParameterType.AVERAGE_TIMESERIES else 0
        )

    @property
    def target_length(self) -> int:
        """
        Length of target timeseries
        """
        return len(self._target) - (
            1 if self._timeseries_type_source == ParameterType.AVERAGE_TIMESERIES else 0
        )
