"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
import datetime


from .core import Core, ParameterSet
from .scmdataframebase import ScmDataFrameBase, DATA_HIERARCHY_SEPARATOR, df_append  # pylint: disable=unused-import
from .constants import ONE_YEAR
from .utils import convert_datetime_to_openscm_time


class OpenSCM(Core):
    """
    High-level OpenSCM class.

    Represents model runs with a particular simple climate model.
    """

    pass


class ScmDataFrame(ScmDataFrameBase):
    """OpenSCM's custom DataFrame implementation.

    The ScmDataFrame implements a subset of the functionality provided by `pyam`'s
    IamDataFrame, but is focused on the providing a performant way of storing
    time series data and the metadata associated with those time series.

    For users who wish to take advantage of all of Pyam's functionality, please cast
    your data frame to an IamDataFrame first with `to_iamdataframe()`. Note: this
    operation can be relatively computationally expensive for large data sets.
    """

    pass


def convert_scmdataframe_to_parameterset(scmdf: ScmDataFrame) -> ParameterSet:
    parameter_set = ParameterSet()
    for (variable, region), df in scmdf.data.groupby(["variable", "region"]):
        df = df.sort_values("time")
        variable_openscm = tuple(variable.split(DATA_HIERARCHY_SEPARATOR))

        region_openscm = tuple(region.split(DATA_HIERARCHY_SEPARATOR))
        assert (
            region_openscm[0] == "World"
        ), "have not considered cases other than the RCPs yet"

        unit = df.unit.unique()
        assert (
            len(unit) == 1
        ), (
            "emissions timeseries should all be in one unit"
        )  # TODO: remove this restriction
        unit = unit[0]

        syr = df.time.min().year
        assert (
            syr == 1765
        ), (
            "have not considered cases other than the RCPs yet"
        )  # TODO: remove this restriction
        eyr = df.time.max().year
        assert (
            eyr == 2500
        ), (
            "have not considered cases other than the RCPs yet"
        )  # TODO: remove this restriction
        assert (
            len(df) == 736
        ), (
            "have not considered cases other than the RCPs read in by pymagicc yet"
        )  # TODO: remove this restriction
        tstep = ONE_YEAR.to(
            "s"
        ).magnitude  # having passed all above, can safely assume this [TODO: remove this assumption]

        emms_view = parameter_set.get_writable_timeseries_view(
            variable_openscm,
            region_openscm,
            unit,
            convert_datetime_to_openscm_time(datetime.datetime(syr, 1, 1, 0, 0, 0)),
            tstep,
        )
        emms_view.set_series(df["value"].values)

    return parameter_set
