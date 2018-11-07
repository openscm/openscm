import numpy as np


def _running_mean(v: np.ndarray, n: int) -> np.ndarray:
    """
    TODO
    """
    c = np.cumsum(np.concatenate(([0], v)))
    return (c[n:] - c[:-n]) / n


def _calc_linearization_values(v: np.ndarray) -> np.ndarray:
    """
    TODO
    """
    edge_points = _running_mean(v, 2)
    middle_points = 3 * v[1:-1] / 2 - (v[0:-2] + v[2:]) / 4
    first_edge_point = 2 * v[0] - edge_points[0]
    last_edge_point = 2 * v[-1] - edge_points[-1]
    return np.concatenate(
        (
            np.array(
                [
                    np.concatenate(([first_edge_point], edge_points)),
                    np.concatenate(([v[0]], middle_points, [v[-1]])),
                ]
            ).T.reshape(2 * len(v)),
            [last_edge_point],
        )
    )


def _calc_linearization(
    source_values: np.ndarray, source_start_time: int, source_period_length: int
) -> (np.ndarray, np.ndarray):
    """
    TODO
    """
    source_stop_time = source_start_time + len(source_values) * source_period_length
    linearization_points = np.linspace(
        source_start_time, source_stop_time, 2 * len(source_values) + 1
    )
    linearization_values = _calc_linearization_values(source_values)
    return linearization_points, linearization_values


def _calc_interpolation_points(
    source_len: int,
    source_start_time: int,
    source_period_length: int,
    target_start_time: int,
    target_period_length: int,
) -> (np.ndarray, np.ndarray, int, int):
    """
    TODO
    """
    source_len *= 2
    source_start_time *= 2
    target_start_time *= 2
    target_period_length *= 2

    if target_start_time >= source_start_time:
        skip_len = 1 + (target_start_time - source_start_time) // source_period_length
        source_len -= skip_len
        source_start_time += skip_len * source_period_length
        first_target_point = target_start_time
        first_interval_length = target_period_length
    else:
        first_target_point = target_start_time + target_period_length
        first_interval_length = source_start_time - target_start_time

    source_stop_time = source_start_time + source_len * source_period_length
    linearization_points = np.linspace(
        source_start_time, source_stop_time, source_len + 1
    )

    target_len = (source_stop_time - first_target_point) // target_period_length
    target_stop_time = first_target_point + target_len * target_period_length
    target_points = np.linspace(first_target_point, target_stop_time, target_len + 1)

    interpolation_points, indices = np.unique(
        np.concatenate((target_points, linearization_points)), return_index=True
    )
    target_indices = np.where(indices < target_len + 1)[0]

    if source_stop_time > target_stop_time:
        last_interval_length = source_stop_time - target_stop_time
    else:
        last_interval_length = target_period_length
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
    target_period_length: int,
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
            interval_sums[1:-1] / target_period_length,
            [interval_sums[-1] / last_interval_length],
        )
    )


def _convert(
    source_values: np.ndarray,
    source_start_time: int,
    source_period_length: int,
    target_start_time: int,
    target_period_length: int,
    interpolation_cache=None,
) -> np.ndarray:
    """
    TODO
    """
    if interpolation_cache is None:
        interpolation_cache = _calc_interpolation_points(
            len(source_values),
            source_start_time,
            source_period_length,
            target_start_time,
            target_period_length,
        )
    interpolation_points, target_indices, first_interval_length, last_interval_length = (
        interpolation_cache
    )

    linearization_points, linearization_values = _calc_linearization(
        source_values, source_start_time, source_period_length
    )

    interpolation_values = np.interp(
        interpolation_points, linearization_points, linearization_values
    )

    return _calc_interval_averages(
        interpolation_points,
        interpolation_values,
        target_indices,
        first_interval_length,
        target_period_length,
        last_interval_length,
    )


class TimeframeConverter:
    """
    Converts timeseries and their points between two timeframes (each defined by a time
    of the first point and a period length).
    """

    _source_start_time: int
    """Start time of source timeframe (seconds since 1970-01-01 00:00:00)"""

    _source_period_length: int
    """Period length of source timeframe in seconds"""

    _target_start_time: int
    """Start time of target timeframe (seconds since 1970-01-01 00:00:00)"""

    _target_period_length: int
    """Period length of target timeframe in seconds"""

    def __init__(
        self,
        source_start_time: int,
        source_period_length: int,
        target_start_time: int,
        target_period_length: int,
    ):
        """
        Initialize.

        Parameters
        ----------
        source_start_time
           Start time of source timeframe (seconds since 1970-01-01 00:00:00)
        source_period_length
            Period length of source timeframe in seconds
        target_start_time
            Start time of target timeframe (seconds since 1970-01-01 00:00:00)
        target_period_length
            Period length of target timeframe in seconds
        """
        self._source_start_time = source_start_time
        self._source_period_length = source_period_length
        self._target_start_time = target_start_time
        self._target_period_length = target_period_length

        if source_start_time > target_start_time + target_period_length:
            raise Exception("TODO Not enough information about first point")

    def convert_from(self, v):
        """
        Convert value **from** source timeframe to target timeframe.

        Parameters
        ----------
        value
            Value
        """
        raise NotImplementedError

    def convert_to(self, v):
        """
        Convert value from target timeframe **to** source timeframe.

        Parameters
        ----------
        value
            Value
        """
        raise NotImplementedError
