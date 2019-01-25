"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
from datetime import datetime


from pyam import IamDataFrame


from .core import Core


class OpenSCM(Core):
    """
    High-level OpenSCM class.

    Represents model runs with a particular simple climate model.
    """

    pass

class OpenSCMDataFrame(IamDataFrame):
    def _format_datetime_col(self, df):
        not_datetime = [not isinstance(x, datetime) for x in df["time"]]
        if any(not_datetime):
            bad_values = df[not_datetime]["time"]
            error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(bad_values)
            raise ValueError(error_msg)

        return df
