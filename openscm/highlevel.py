"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
from datetime import datetime
from dateutil import parser


from pyam import IamDataFrame


from .core import Core


class OpenSCM(Core):
    """
    High-level OpenSCM class.

    Represents model runs with a particular simple climate model.
    """

    pass

class OpenSCMDataFrameBase(IamDataFrame):
    """This base is the class other libraries can subclass

    Having such a subclass avoids a potential circularity where e.g. openscm imports OpenSCMDataFrame as well as Pymagicc, but Pymagicc wants to import OpenSCMDataFrame and hence to try and import OpenSCMDataFrame you have to import OpenSCMDataFrame itself (hence the circularity).
    """
    def _format_datetime_col(self):
        if isinstance(self.data["time"].iloc[0], str):
            def convert_str_to_datetime(inp):
                return parser.parse(inp)

            self.data["time"] = self.data["time"].apply(convert_str_to_datetime)

        not_datetime = [not isinstance(x, datetime) for x in self.data["time"]]
        if any(not_datetime):
            bad_values = self.data[not_datetime]["time"]
            error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(bad_values)
            raise ValueError(error_msg)


class OpenSCMDataFrame(OpenSCMDataFrameBase):
    """OpenSCM's custom DataFrame implementation.

    The OpenSCMDataFrame wraps around `pyam`'s IamDataFrame, which itself wraps around Pandas.

    The OpenSCMDataFrame provides a number of diagnostic features (including validation of
    data, completeness of variables provided, running of simple climate models) as
    well as a number of visualization and plotting tools.
    """
    pass
