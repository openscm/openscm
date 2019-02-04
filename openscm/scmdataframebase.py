from datetime import datetime
from dateutil import parser


from pyam import IamDataFrame


class ScmDataFrameBase(IamDataFrame):
    """This base is the class other libraries can subclass

    Having such a subclass avoids a potential circularity where e.g. openscm imports ScmDataFrame as well as Pymagicc, but Pymagicc wants to import ScmDataFrame and hence to try and import ScmDataFrame you have to import ScmDataFrame itself (hence the circularity).
    """

    def _format_datetime_col(self):
        if isinstance(self.data["time"].iloc[0], str):

            def convert_str_to_datetime(inp):
                return parser.parse(inp)

            self.data["time"] = self.data["time"].apply(convert_str_to_datetime)

        not_datetime = [not isinstance(x, datetime) for x in self.data["time"]]
        if any(not_datetime):
            bad_values = self.data[not_datetime]["time"]
            error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(
                bad_values
            )
            raise ValueError(error_msg)
