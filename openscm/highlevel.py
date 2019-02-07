"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
import datetime


import numpy as np
import pandas as pd


from .core import Core
from .scmdataframebase import (
    ScmDataFrameBase,
    DATA_HIERARCHY_SEPARATOR,
    df_append,
)  # pylint: disable=unused-import
from .constants import ONE_YEAR_IN_S_INTEGER
from .utils import convert_datetime_to_openscm_time, convert_openscm_time_to_datetime
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


def convert_scmdataframe_to_core(
    scmdf: ScmDataFrame, climate_model: str = "unspecified"
) -> Core:
    # TODO: move to method of scmdataframe
    tsdf = scmdf.timeseries()

    # columns are times when you call scmdataframe.timeseries()
    stime = tsdf.columns.min()
    etime = tsdf.columns.max()

    st = convert_datetime_to_openscm_time(stime)
    et = convert_datetime_to_openscm_time(etime)
    core = Core(climate_model, st, et)

    syr = stime.year
    eyr = etime.year
    assert (
        syr == 1765
    ), (
        "have not considered cases other than the RCPs yet"
    )  # TODO: remove this restriction
    eyr = scmdf["time"].max().year
    assert (
        eyr == 2500
    ), (
        "have not considered cases other than the RCPs yet"
    )  # TODO: remove this restriction
    assert (
        len(scmdf["time"].unique()) == 736
    ), (
        "have not considered cases other than the RCPs read in by pymagicc yet"
    )  # TODO: remove this restriction
    tstep = (
        ONE_YEAR_IN_S_INTEGER
    )  # having passed all above, can safely assume this [TODO: remove this assumption]

    variable_idx = scmdf.timeseries().index.names.index("variable")
    region_idx = scmdf.timeseries().index.names.index("region")
    unit_idx = scmdf.timeseries().index.names.index("unit")

    assert len(scmdf["scenario"].unique()) == 1, "haven't thought about this yet"
    assert len(scmdf["model"].unique()) == 1, "haven't thought about this yet"
    assert len(scmdf["climate_model"].unique()) == 1, "haven't thought about this yet"

    for i in tsdf.index:
        variable = i[variable_idx]
        region = i[region_idx]
        unit = i[unit_idx]

        variable_openscm = tuple(variable.split(DATA_HIERARCHY_SEPARATOR))

        region_openscm = tuple(region.split(DATA_HIERARCHY_SEPARATOR))
        assert (
            region_openscm[0] == "World"
        ), "have not considered cases other than the RCPs yet"

        emms_view = core.parameters.get_writable_timeseries_view(
            variable_openscm,
            region_openscm,
            unit,
            convert_datetime_to_openscm_time(datetime.datetime(syr, 1, 1, 0, 0, 0)),
            tstep,
        )
        emms_view.set_series(tsdf.loc[i, :].values)

    return core


def convert_core_to_scmdataframe(
    core: Core,
    period_length: int = ONE_YEAR_IN_S_INTEGER,
    model: str = "unspecified",
    scenario: str = "unspecified",
    climate_model: str = "unspecified",
) -> ScmDataFrame:
    def get_metadata(c, para):
        md = {}
        if para._children:
            for _, child_para in para._children.items():
                md.update(get_metadata(c, child_para))
        is_time_data = para._info._type == ParameterType.TIMESERIES
        if (para._info._type is None) or is_time_data:
            return metadata

        raise NotImplementedError
        values = para._data
        variable = value.info.name
        if isinstance(values, float):
            metadata["{} ({})".format(variable, para.info.unit)] = values

        return metadata

    def get_scmdataframe_timeseries_columns(core_in, metadata_in):
        def get_ts_ch(core_here, para_here, ts_in, time_in, ch_in):
            if para_here._children:
                for _, child_para in para_here._children.items():
                    ts_in, time_in, ch_in = get_ts_ch(
                        core_here, child_para, ts_in, time_in, ch_in
                    )
            if not para_here._info._type == ParameterType.TIMESERIES:
                return ts_in, time_in, ch_in

            unit = para_here.info.unit
            tview = core.parameters.get_timeseries_view(
                para_here.full_name,
                para_here.info.region,
                unit,
                core_here.start_time,
                period_length,
            )
            values = tview.get_series()
            time = np.array(
                [convert_openscm_time_to_datetime(int(t)) for t in tview.get_times()]
            )
            if time_in is None:
                time_in = time
            else:
                assert (time_in == time).all()

            ts_in.append(values)
            ch_in["unit"].append(unit)
            ch_in["variable"].append(DATA_HIERARCHY_SEPARATOR.join(para_here.full_name))
            ch_in["region"].append(DATA_HIERARCHY_SEPARATOR.join(para_here.info.region))

            return ts_in, time_in, ch_in

        ts = []
        time_axis = None
        column_headers = {"variable": [], "region": [], "unit": []}
        for key, value in core_in.parameters._root._parameters.items():
            ts, time_axis, column_headers = get_ts_ch(
                core_in, value, ts, time_axis, column_headers
            )

        return (
            pd.DataFrame(np.vstack(ts).T, pd.Index(time_axis)),
            {**metadata, **column_headers},
        )

    metadata = {
        "climate_model": [climate_model],
        "scenario": [scenario],
        "model": [model],
    }

    for key, value in core.parameters._root._parameters.items():
        metadata.update(get_metadata(core, value))

    timeseries, columns = get_scmdataframe_timeseries_columns(core, metadata)
    # convert timeseries to dataframe with time index here
    return ScmDataFrame(timeseries, columns=columns)
