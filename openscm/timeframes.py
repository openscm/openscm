from copy import copy
import numpy as np
from typing import Tuple


class Timeframe:
    """
    TODO
    """

    start_time: int
    """Start time (seconds since :literal:`1970-01-01 00:00:00`)"""

    period_length: int
    """Period length in seconds"""

    def __init__(self, start_time, period_length):
        """
        TODO
        """
        self.start_time = start_time
        self.period_length = period_length

    def get_points(self, count: int) -> np.ndarray:
        """
        TODO
        """
        return np.linspace(self.start_time, self.get_stop_time(count), count)

    def get_stop_time(self, count: int) -> int:
        """
        TODO
        """
        return self.start_time + (count - 1) * self.period_length

    def get_length_until(self, stop_time: int) -> int:
        """
        TODO
        """
        return (
            stop_time - self.start_time + self.period_length - 1
        ) // self.period_length


def _running_mean(values: np.ndarray, n: int) -> np.ndarray:
    """
    TODO
    """
    c = np.cumsum(np.concatenate(([0], values)))
    return (c[n:] - c[:-n]) / n


def _calc_linearization_values(values: np.ndarray) -> np.ndarray:
    """
    TODO
    """
    edge_points = _running_mean(values, 2)
    middle_points = 3 * values[1:-1] / 2 - (values[0:-2] + values[2:]) / 4
    first_edge_point = 2 * values[0] - edge_points[0]
    last_edge_point = 2 * values[-1] - edge_points[-1]
    return np.concatenate(
        (
            np.array(
                [
                    np.concatenate(([first_edge_point], edge_points)),
                    np.concatenate(([values[0]], middle_points, [values[-1]])),
                ]
            ).T.reshape(2 * len(values)),
            [last_edge_point],
        )
    )


def _calc_linearization(
    values: np.ndarray, timeframe: Timeframe
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO
    """
    stop_time = timeframe.start_time + len(values) * timeframe.period_length
    linearization_points = np.linspace(
        timeframe.start_time, stop_time, 2 * len(values) + 1
    )
    linearization_values = _calc_linearization_values(values)
    return linearization_points, linearization_values


def _calc_interpolation_points(
    source_len: int, source: Timeframe, target: Timeframe
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    TODO
    """
    linearization_points_len = 2 * source_len + 1
    source = Timeframe(2 * source.start_time, 1 * source.period_length)
    target = Timeframe(2 * target.start_time, 2 * target.period_length)

    if target.start_time >= source.start_time:
        skip_len = 1 + (target.start_time - source.start_time) // source.period_length
        linearization_points_len -= skip_len
        source.start_time += skip_len * source.period_length
        first_target_point = target.start_time
        first_interval_length = target.period_length
    else:
        first_target_point = target.start_time + target.period_length
        first_interval_length = source.start_time - target.start_time

    source_stop_time = source.get_stop_time(linearization_points_len)
    target_len = 1 + (source_stop_time - first_target_point) // target.period_length
    target_stop_time = target.get_stop_time(target_len)

    if source_stop_time > target_stop_time:
        last_interval_length = source_stop_time - target_stop_time
    else:
        last_interval_length = target.period_length

    interpolation_points, indices = np.unique(
        np.concatenate(
            (target.get_points(target_len), source.get_points(linearization_points_len))
        ),
        return_index=True,
    )
    target_indices = np.where(indices < target_len)[0]

    if source_stop_time <= target_stop_time:
        target_indices = target_indices[:-1]

    return (
        interpolation_points / 2,
        target_indices,
        first_interval_length // 2,
        last_interval_length // 2,
    )


def _calc_interval_averages(
    interpolation_points: np.ndarray,
    interpolation_values: np.ndarray,
    target_indices: np.ndarray,
    first_interval_length: int,
    period_length: int,
    last_interval_length: int,
) -> np.ndarray:
    """
    TODO
    """
    interval_sums = (
        np.add.reduceat(
            (interpolation_values[1:] + interpolation_values[:-1])
            * (interpolation_points[1:] - interpolation_points[:-1]),
            np.concatenate(([0], target_indices))
            if target_indices[0] != 0
            else target_indices,
        )
        / 2
    )
    return np.concatenate(
        (
            [interval_sums[0] / first_interval_length],
            interval_sums[1:-1] / period_length,
            [interval_sums[-1] / last_interval_length],
        )
    )


def _convert(
    values: np.ndarray, source: Timeframe, target: Timeframe, interpolation_cache=None
) -> np.ndarray:
    """
    TODO
    """
    if interpolation_cache is None:
        interpolation_cache = _calc_interpolation_points(len(values), source, target)
    interpolation_points, target_indices, first_interval_length, last_interval_length = (
        interpolation_cache
    )

    linearization_points, linearization_values = _calc_linearization(values, source)

    interpolation_values = np.interp(
        interpolation_points, linearization_points, linearization_values
    )

    return _calc_interval_averages(
        interpolation_points,
        interpolation_values,
        target_indices,
        first_interval_length,
        target.period_length,
        last_interval_length,
    )


class TimeframeConverter:
    """
    Converts timeseries and their points between two timeframes (each defined by a time
    of the first point and a period length).
    """

    _source: Timeframe
    """Source timeframe"""

    _target: Timeframe
    """Target timeframe"""

    def __init__(self, source: Timeframe, target: Timeframe):
        """
        Initialize.

        Parameters
        ----------
        source
            Source timeframe
        target
            Target timeframe
        """
        self._source = copy(source)
        self._target = copy(target)

        if source.start_time > target.start_time + target.period_length:
            raise Exception("TODO Not enough information about first point")

    def convert_from(self, values):
        """
        Convert value **from** source timeframe to target timeframe.

        Parameters
        ----------
        values
            Value
        """
        raise NotImplementedError

    def convert_to(self, values):
        """
        Convert value from target timeframe **to** source timeframe.

        Parameters
        ----------
        values
            Value
        """
        raise NotImplementedError
