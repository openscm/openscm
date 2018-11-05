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
