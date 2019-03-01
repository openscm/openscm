"""
Different climate models often use different time frames for their input and output
data. This includes different 'meanings' of time steps (e.g. beginning vs middle of year) and
different lengths of the time steps (e.g. years vs months). Accordingly, OpenSCM
supports the conversion of timeseries data between such timeseriess, which is handled in
this module. A thorough explaination of the procedure used is given in a dedicated
`Jupyter Notebook
<https://github.com/openclimatedata/openscm/blob/master/notebooks/timeseriess.ipynb>`_.
"""

from copy import copy
from typing import NamedTuple, Tuple, Callable


import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate


class InsufficientDataError(ValueError):
    """
    Exception raised when not enough data overlap is available when converting from one
    timeseries to another (e.g. when the target timeseries is outside the range of the
    source timeseries) or when data is too short (less than 3 data points).
    """


class Timeseries:
    """
    Convenience class representing a timeseries consisting of a start time and a period
    length.
    """

    start_time: int
    """Start time (seconds since :literal:`1970-01-01 00:00:00`)"""

    period_length: int
    """Period length in seconds"""

    def __init__(self, start_time, period_length):
        """
        Initialize.

        Parameters
        ----------
        start_time
            Start time (seconds since ``1970-01-01 00:00:00``)
        period_length
            Period length (in seconds)
        """
        self.start_time = start_time
        self.period_length = period_length

    def __repr__(self) -> str:
        """
        Get string representation.

        Returns
        -------
        str
            String representation
        """
        return "<openscm.timeseries.Timeseries(start_time={}, period_length={})>".format(
            self.start_time, self.period_length
        )

    def get_points(self, count: int) -> np.ndarray:
        """
        Get the ``count`` first points in time corresponding to the timeseries.

        Parameters
        ----------
        count
            Number of time points to return

        Returns
        -------
        np.ndarray
            Array of time points
        """
        return np.linspace(self.start_time, self.get_stop_time(count - 1), count)

    def get_stop_time(self, count: int) -> int:
        """
        Get the point in time at which a timeseries of ``count`` points stops according
        to this timeseries.

        Parameters
        ----------
        count
            Number of time points

        Returns
        -------
        int
            Time point (seconds since ``1970-01-01 00:00:00``)
        """
        return self.start_time + count * self.period_length

    def get_length_until(self, stop_time: int) -> int:
        """
        Get the number of time points in this timeseries until ``stop_time`` (including).

		Note that this excludes the start time as a point so
		``self.get_length_until(self.start_time)`` will be ``0``.

        Parameters
        ----------
        stop_time
            Stop time(seconds since ``1970-01-01 00:00:00``)

        Returns
        -------
        int
            Number of time points
        """
        return (stop_time + 1 - self.start_time) // self.period_length


def _calc_linearization_values(values: np.ndarray) -> np.ndarray:
    """
    Calculate the "linearization" values of the array ``values`` which is assumed to
    represent averages over time periods. Values at the edges of the periods are taken
    as the average of adjacent periods, values at the period middles are taken such that
    the integral over a period is the same as for the input data.

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
    ) / 2  # values = 1 / 2 * (edges_lower + middle_point_values) / 2 + 1 / 2 * (middle_point_values + edges_upper) / 2
    first_edge_point_value = (
        2 * values[0] - edge_point_values[0]
    )  # values[0] = (first_edge_point_value + edge_point_values[0] ) / 2
    last_edge_point_value = (
        2 * values[-1] - edge_point_values[-1]
    )  # values[-1] = (last_edge_point_value + edge_point_values[-1] ) / 2
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


def _calc_continuous(
    values: np.ndarray, timeseries: Timeseries
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the "continuous" time points and values (see
    :func:`openscm.timeseries._calc_linearization_values`) of the array ``values``
    according to timeseries ``timeseries``.

    Parameters
    ----------
    values
        Timeseries values of period averages
    timeseries
        Timeseries

    Returns
    -------
    func
        Function which, represents the interpolated timeseries. It takes a single argument,
        time ("x-value"), and returns a single float, the value of the interpolated
        timeseries at that point in time ("y-value").
    """
    stop_time = timeseries.start_time + len(values) * timeseries.period_length
    linearization_points = np.linspace(
        timeseries.start_time, stop_time, 2 * len(values) + 1
    )
    linearization_values = _calc_linearization_values(values)
    # I'm not happy about extrapolation by default but only way to make things
    # work at the moment
    return sp.interpolate.interp1d(
    	linearization_points,
    	linearization_values,
    	fill_value="extrapolate"
    )


class _Interpolation(NamedTuple):
    """
    Collection of interpolation points and associated information.
    """

    data_len: int
    """
    Length of source data array for which these interpolation points have been
    calculated
    """

    first_interval_length: int
    """
    Length of the first interval/period (can be smaller than ``period_length``
    when not enough data overlap is available)
    """

    last_interval_length: int
    """
    Length of the last interval/period (can be smaller than ``period_length``
    when not enough data overlap is available)
    """

    points: np.ndarray
    """Array of interpolation time points"""

    period_length: int
    """
    Length of the middle intervals/periods
    """

    target_indices: np.ndarray
    """
    Array of indices in ``interpolation_points`` that correspond to edges of target
    periods
    """


def _calc_interpolation_points(
    source_len: int, source: Timeseries, target: Timeseries
) -> _Interpolation:
    """
    Calculate the "interpolation" time points that correspond to edges of the target
    periods and the sampling points for the linearization of source values (i.e. edges
    and middle points of the source periods).

    Parameters
    ----------
    source_len
        Length of source data values
    source
        Source timeseries
    target
        Target timeseries

    Returns
    -------
    _Interpolation
        Object wrapping the actual interpolation points and additional information
    """
    linearization_points_len = 2 * source_len + 1
    # For in-between steps periods in the source timeseries need to be halved. To stick
    # calculations to integer calculations, rather than devide source period length by 2
    # double all other times/period lengths and devide at the end:
    source = Timeseries(2 * source.start_time, 1 * source.period_length)
    target = Timeseries(2 * target.start_time, 2 * target.period_length)

    if target.start_time >= source.start_time:
        skip_len = 1 + (target.start_time - source.start_time) // source.period_length
        linearization_points_len -= skip_len
        source.start_time += skip_len * source.period_length
        first_target_point = target.start_time
        first_interval_length = target.period_length
    else:
        first_target_point = target.start_time + target.period_length
        first_interval_length = source.start_time - target.start_time

    source_stop_time = source.get_stop_time(linearization_points_len - 1)
    target_len = (source_stop_time - first_target_point) // target.period_length
    target_stop_time = target.get_stop_time(target_len)

    if source_stop_time > target_stop_time:
        last_interval_length = source_stop_time - target_stop_time
    else:
        last_interval_length = target.period_length

    interpolation_points, indices = np.unique(
        np.concatenate(
            (
                target.get_points(target_len + 1),
                source.get_points(linearization_points_len),
            )
        ),
        return_index=True,
    )
    target_indices = np.where(indices <= target_len)[0]

    if source_stop_time <= target_stop_time:
        target_indices = target_indices[:-1]

    return _Interpolation(
        data_len=source_len,
        first_interval_length=first_interval_length // 2,
        last_interval_length=last_interval_length // 2,
        period_length=target.period_length // 2,
        points=interpolation_points / 2,
        target_indices=target_indices,
    )


def _calc_interval_averages(
    continuous: Callable[[float], float], target_intervals: np.ndarray
) -> np.ndarray:
    """
    Calculate the interval averages of a continuous function.

    Here interval average is calculated as the integral over the period divided by
    the period length.

    Parameters
    ----------
    continuous
        Continuous function from which to calculate the interval averages. Should be
        calculated using :func:`openscm.timeseries._calc_continuous`.
    target_intervals
        Intervals to calculate the average of.

    Returns
    -------
    np.ndarray
        Array of the interval/period averages
    """
    averages = np.zeros_like(target_intervals[:-1])
    for i, l in enumerate(target_intervals[:-1]):
        u = target_intervals[i + 1]
        y, _ = integrate.quad(continuous, l, u)
        averages[i] = y / (u - l)

    return averages


def _convert(
    values: np.ndarray,
    source: Timeseries,
    target: Timeseries,
    interpolation: _Interpolation = None,
) -> np.ndarray:
    """
    Convert time period average data ``values`` for timeseries ``source`` to the
    timeseries ``target``.

    Parameters
    ----------
    values
        Array of data to convert
    source
        Source timeseries
    target
        Target timeseries
    interpolation
        Interpolation data. Used for caching and is newly calculated when not given,
        i.e. ``None`` (default).

    Returns
    -------
    np.ndarray
        Converted time period average data for timeseries ``target``
    """
    continuous = _calc_continuous(values, source)

    target_len = target.get_length_until(source.get_stop_time(len(values)))
    target_times = target.get_points(target_len)
    target_intervals = np.concatenate(
        [target_times, [target_times[-1] + target.period_length]]
    )

    return _calc_interval_averages(continuous, target_intervals)


def _convert_cached(
    values: np.ndarray,
    source: Timeseries,
    target: Timeseries,
    interpolation: _Interpolation,
) -> Tuple[np.ndarray, _Interpolation]:
    """
    Convert time period average data ``values`` for timeseries ``source`` to the
    timeseries ``target`` using and updating cache.

    Parameters
    ----------
    values
        Array of data to convert
    source
        Source timeseries
    target
        Target timeseries
    interpolation
        Interpolation data (as resulting from
        :func:`openscm.timeseriess._calc_interpolation_points`)

    Returns
    -------
    result
        Converted time period average data for timeseries ``target``
    interpolation
        (Possibly) updated interpolation
    """
    values_len = len(values)
    if values_len < 3:
        raise InsufficientDataError
    if interpolation is None or values_len != interpolation.data_len:
        interpolation = _calc_interpolation_points(values_len, source, target)
    return _convert(values, source, target, interpolation), interpolation


class TimeseriesConverter:
    """
    Converts timeseries and their points between two timeseriess (each defined by a time
    of the first point and a period length).
    """

    _source: Timeseries
    """Source timeseries"""

    _target: Timeseries
    """Target timeseries"""

    _convert_from_interpolation: _Interpolation = None
    """
    Cached interpolation data as resulting from
    :func:`openscm.timeseriess._calc_interpolation_points` for conversion from source to
    target
    """

    _convert_to_interpolation: _Interpolation = None
    """
    Cached interpolation data as resulting from
    :func:`openscm.timeseriess._calc_interpolation_points` for conversion from target to
    source
    """

    def __init__(self, source: Timeseries, target: Timeseries):
        """
        Initialize.

        Parameters
        ----------
        source
            Source timeseries
        target
            Target timeseries
        """
        self._source = copy(source)
        self._target = copy(target)

        if source.start_time > target.start_time + target.period_length:
            raise InsufficientDataError

    def convert_from(self, values: np.ndarray) -> np.ndarray:
        """
        Convert value **from** source timeseries to target timeseries.

        Parameters
        ----------
        values
            Value

        Returns
        -------
        np.ndarray
            Converted array
        """
        result, self._convert_from_interpolation = _convert_cached(
            values, self._source, self._target, self._convert_from_interpolation
        )
        return result

    def convert_to(self, values: np.ndarray) -> np.ndarray:
        """
        Convert value from target timeseries **to** source timeseries.

        Parameters
        ----------
        values
            Value

        Returns
        -------
        np.ndarray
            Converted array
        """
        result, self._convert_to_interpolation = _convert_cached(
            values, self._target, self._source, self._convert_to_interpolation
        )
        return result

    def get_source_len(self, target_len: int) -> int:
        """
        Get length of timeseries in source timeseries.

        Parameters
        ----------
        target_len
            Length of timeseries in target timeseries.
        """
        return self._source.get_length_until(self._target.get_stop_time(target_len))

    def get_target_len(self, source_len: int) -> int:
        """
        Get length of timeseries in target timeseries.

        Parameters
        ----------
        source_len
            Length of timeseries in source timeseries.
        """
        return self._target.get_length_until(self._source.get_stop_time(source_len))

    @property
    def source(self) -> Timeseries:
        """
        Source timeseries
        """
        return self._source

    @property
    def target(self) -> Timeseries:
        """
        Target timeseries
        """
        return self._target
