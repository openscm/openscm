"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
import datetime

import numpy as np
import pandas as pd
from dateutil import parser

from openscm.core import Core, ParameterSet
from openscm.parameters import ParameterType
from openscm.units import UnitConverter
from openscm.utils import convert_datetime_to_openscm_time, convert_openscm_time_to_datetime
from .base import ScmDataFrameBase

try:
    from pyam import IamDataFrame
except ImportError:
    IamDataFrame = None

DATA_HIERARCHY_SEPARATOR = "|"
"""str: String used to define different levels in our data hierarchies.

For example, "Emissions|CO2|Energy|Coal".

We copy this straight from pyam
"""

ONE_YEAR_IN_S_INTEGER = int(round(UnitConverter('year', 's').convert_from(1)))


class ScmDataFrame(ScmDataFrameBase):
    """OpenSCM's custom DataFrame implementation.

    The ScmDataFrame implements a subset of the functionality provided by `pyam`'s
    IamDataFrame, but is focused on the providing a performant way of storing
    time series data and the metadata associated with those time series.

    For users who wish to take advantage of all of Pyam's functionality, please cast
    your data frame to an IamDataFrame first with `to_iamdataframe()`. Note: this
    operation can be relatively computationally expensive for large data sets.
    """

    def append(self, other: ScmDataFrameBase, inplace=False, **kwargs):
        """Appends additional timeseries from a castable object to the current dataframe

        See ``df_append``

        Parameters
        ----------
        other: openscm.scmdataframe.ScmDataFrame or something which can be cast to ScmDataFrameBase
        """
        if not isinstance(other, ScmDataFrameBase):
            other = self.__class__(other, **kwargs)

        return df_append([self, other], inplace=inplace)

    def to_iamdataframe(self):
        """Convert to  IamDataFrame instance

        Returns
        -------
        An pyam.IamDataFrame instance containing the same data
        """
        if IamDataFrame is None:
            raise ImportError('pyam is not installed. Features involving IamDataFrame are unavailable')

        class LongIamDataFrame(IamDataFrame):
            """This baseclass is a custom implementation of the IamDataFrame which handles datetime data which spans longer than pd.to_datetime
            can handle
            """

            def _format_datetime_col(self):
                if isinstance(self.data["time"].iloc[0], str):
                    def convert_str_to_datetime(inp):
                        return parser.parse(inp)

                    self.data["time"] = self.data["time"].apply(convert_str_to_datetime)

                not_datetime = [not isinstance(x, datetime.datetime) for x in self.data["time"]]
                if any(not_datetime):
                    bad_values = self.data[not_datetime]["time"]
                    error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(
                        bad_values
                    )
                    raise ValueError(error_msg)

        return LongIamDataFrame(self.timeseries())

    def to_csv(self, path: str, **kwargs):
        """Write timeseries data to a csv file

        Parameters
        ----------
        path: string
            file path
        """
        self.to_iamdataframe().to_csv(path, **kwargs)

    def line_plot(self, x: str = "time", y: str = "value", **kwargs):
        """Helper to generate line plots of timeseries

        See ``pyam.IamDataFrame.line_plot`` for more information

        """
        return self.to_iamdataframe().line_plot(x, y, **kwargs)

    def scatter(self, x: str, y: str, **kwargs):
        """Plot a scatter chart using metadata columns

        see pyam.plotting.scatter() for all available options
        """
        self.to_iamdataframe().scatter(x, y, **kwargs)

    def region_plot(self, **kwargs):
        """Plot regional data for a single model, scenario, variable, and year

        see ``pyam.plotting.region_plot()`` for all available options
        """
        return self.to_iamdataframe().region_plot(**kwargs)

    def pivot_table(self, index, columns, **kwargs):
        """Returns a pivot table

        see ``pyam.core.IamDataFrame.pivot_table()`` for all available options
        """
        return self.to_iamdataframe().pivot_table(index, columns, **kwargs)


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
    # TODO: remove this restriction
    assert syr == 1765, "have not considered cases other than the RCPs yet"
    eyr = scmdf["time"].max().year
    # TODO: remove this restriction
    assert eyr == 2500, "have not considered cases other than the RCPs yet"
    # TODO: remove this restriction
    assert len(scmdf["time"].unique()) == 736, \
        "have not considered cases other than the RCPs read in by pymagicc yet"
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

        variable = value.info.name
        if para._info._type == ParameterType.BOOLEAN:
            values = para._data
            label = "{}".format(variable)
        elif para._info._type == ParameterType.ARRAY:
            values = tuple(para._data)  # pandas indexes must be hashable
            label = "{} ({})".format(variable, para.info.unit)
        else:
            values = para._data
            label = "{} ({})".format(variable, para.info.unit)

        metadata[label] = [values]
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


def convert_config_dict_to_parameter_set(config):
    assert isinstance(config, dict)
    parameters = ParameterSet()
    for key, (region, value) in config.items():
        view = parameters.get_writable_scalar_view(key, region, str(value.units))
        view.set(value.magnitude)

    return parameters


def df_append(dfs, inplace=False):
    """
    Append together many dataframes into a single ScmDataFrame
    When appending many dataframes it may be more efficient to call this routine once with a list of ScmDataFrames, then using
    `ScmDataFrame.append`. If timeseries with duplicate metadata are found, the timeseries are appended. For duplicate timeseries,
    values fallings on the same timestep are averaged.
    Parameters
    ----------
    dfs: list of ScmDataFrameBase object, string or pd.DataFrame.
    The dataframes to append. Values will be attempted to be cast to non ScmDataFrameBase.
    inplace : bool
    If True, then the operation updates the first item in dfs
    Returns
    -------
    ScmDataFrameBase-like object containing the merged data. The resultant class will be determined by the type of the first object
    in dfs
    """
    dfs = [
        df if isinstance(df, ScmDataFrameBase) else ScmDataFrameBase(df) for df in dfs
    ]
    joint_dfs = [df.copy() for df in dfs]
    joint_meta = []
    for df in joint_dfs:
        joint_meta += df.meta.columns.tolist()

    joint_meta = set(joint_meta)

    # should probably solve this https://github.com/pandas-dev/pandas/issues/3729
    na_fill_value = -999
    for i, _ in enumerate(joint_dfs):
        for col in joint_meta:
            if col not in joint_dfs[i].meta:
                joint_dfs[i].set_meta(na_fill_value, name=col)

    # we want to put data into timeseries format and pass into format_ts instead of format_data
    data = pd.concat(
        [d.timeseries().reorder_levels(joint_meta) for d in joint_dfs], sort=False
    )

    data = data.reset_index()
    data[list(joint_meta)] = data[joint_meta].replace(
        to_replace=np.nan, value=na_fill_value
    )
    data = data.set_index(list(joint_meta))

    data = data.groupby(data.index.names).mean()

    if not inplace:
        ret = dfs[0].copy()
    else:
        ret = dfs[0]

    ret._data = data.reset_index(drop=True).T
    ret._data.index = ret._data.index.astype("object")
    ret._data.index.name = "time"
    ret._data = ret._data.astype(float)

    ret._meta = (
        data.index.to_frame()
        .reset_index(drop=True)
        .replace(to_replace=na_fill_value, value=np.nan)
    )
    ret._sort_meta_cols()

    if not inplace:
        return ret
