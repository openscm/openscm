"""
Imports and classes required to ensure compatibility with Pyam is intelligently handled
"""
import datetime

from dateutil import parser

try:
    from pyam import IamDataFrame

    # mypy can't work out try-except block forces IamDataFrame to be ok here
    class LongDatetimeIamDataFrame(IamDataFrame):  # type: ignore
        """
        Custom implementation of ``pyam.IamDataFrame`` which handles long datetime data.

        This custom implementation allows the data frame to handle times outside
        panda's limited time range of 1677-09-22 00:12:43.145225 to
        2262-04-11 23:47:16.854775807, see `this discussion
        <https://stackoverflow.com/a/37226672>`_.
        """

        def _format_datetime_col(self):
            if self.data["time"].apply(lambda x: isinstance(x, str)).any():

                def convert_str_to_datetime(inp):
                    try:
                        return parser.parse(inp)
                    except ValueError:
                        return inp

                self.data["time"] = self.data["time"].apply(convert_str_to_datetime)

            not_datetime = [
                not isinstance(x, datetime.datetime) for x in self.data["time"]
            ]
            if any(not_datetime):
                bad_values = self.data[not_datetime]["time"]
                error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(
                    bad_values
                )
                raise ValueError(error_msg)


except ImportError:
    # mypy can't work out try-except block sets typing
    IamDataFrame = None
    LongDatetimeIamDataFrame = None  # type: ignore

try:
    from matplotlib.axes import Axes  # pylint: disable=unused-import
except ImportError:
    Axes = None
