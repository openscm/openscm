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
from typing import Tuple, Callable


import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate


class InsufficientDataError(ValueError):
    """
    Exception raised when not enough data is available to convert from one
    timeseries to another (e.g. when the target timeseries is outside the range of the
    source timeseries) or when data is too short (fewer than 3 data points).
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
    return interpolate.interp1d(
        linearization_points, linearization_values, fill_value="extrapolate"
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
    int_averages = [np.nan] * len(target_intervals[:-1])
    for i, l in enumerate(target_intervals[:-1]):
        u = target_intervals[i + 1]
        y, _ = integrate.quad(continuous, l, u)
        int_averages[i] = y / (u - l)

    return np.array(int_averages)


def _convert(values: np.ndarray, source: Timeseries, target: Timeseries) -> np.ndarray:
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
    values: np.ndarray, source: Timeseries, target: Timeseries
) -> Tuple[np.ndarray]:
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

    Returns
    -------
    result
        Converted time period average data for timeseries ``target``
    """
    values_len = len(values)
    if values_len < 3:
        raise InsufficientDataError
    return _convert(values, source, target)


class TimeseriesConverter:
    """
    Converts timeseries and their points between two timeseriess (each defined by a time
    of the first point and a period length).
    """

    _source: Timeseries
    """Source timeseries"""

    _target: Timeseries
    """Target timeseries"""

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
        result = _convert_cached(values, self._source, self._target)
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
        result = _convert_cached(values, self._target, self._source)
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
