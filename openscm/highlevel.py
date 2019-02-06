"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
import datetime


import numpy as np
import pandas as pd


from .core import Core, ParameterSet
from .scmdataframebase import ScmDataFrameBase, DATA_HIERARCHY_SEPARATOR, df_append  # pylint: disable=unused-import
from .constants import ONE_YEAR
from .utils import convert_datetime_to_openscm_time
from .parameters import ParameterType


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


def convert_parameterset_to_scmdataframe(
    paraset: ParameterSet,
    model: str = "unspecified",
    scenario: str = "unspecified",
    climate_model: str = "unspecified",
    start: int = 0,
    period_length: int = ONE_YEAR.to("s").magnitude,
    no_timesteps: int = 10,
) -> ScmDataFrame:
    metadata = {
        "climate_model": climate_model,
        "scenario": scenario,
        "model": model,
    }

    def get_metadata(para):
        md = {}
        if para._children:
            for _, child_para in para._children.items():
                md.update(get_metadata(child_para))
        is_time_data = para._info._type == ParameterType.TIMESERIES
        if (para._info._type is None) or is_time_data:
            return metadata

        raise NotImplementedError
        values = para._data
        variable = value.info.name
        if isinstance(values, float):
            metadata["{} ({})".format(variable, para.info.unit)] = values

        return metadata

    for key, value in paraset._root._parameters.items():
        metadata.update(get_metadata(value))

    def get_dataframes(para, metadata):
        df = []
        if para._children:
            for _, child_para in para._children.items():
                df.append(get_dataframes(child_para, metadata))
        if not para._info._type == ParameterType.TIMESERIES:
            return df

        import pdb
        pdb.set_trace()
        values = para._data
        variable = DATA_HIERARCHY_SEPARATOR.join(para.full_name)
        region = DATA_HIERARCHY_SEPARATOR.join(para.info.region)
        unit = para.info.unit
        time =
        tdf = {
            **metadata,
            "variable": [variable] * no_timesteps,
            "unit": [unit] * no_timesteps,
            "region": [region] * no_timesteps,
            "time": time,
        }
        tdf["value"] = values

        return pd.DataFrame(tdf)

    dataframes = []
    for key, value in paraset._root._parameters.items():
        dfs = get_dataframes(value, metadata)
        if dfs:
            dataframes.append(dfs)


    result = ScmDataFrame(pd.concat(dataframes))

    return result
