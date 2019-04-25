"""
The OpenSCM high-level API provides high-level functionality around
single model runs.  This includes reading/writing input and output
data, easy setting of parameters and stochastic ensemble runs.
"""
# import numpy as np
# import pandas as pd

# from openscm.core import Core, ParameterSet
# from openscm.parameters import ParameterType
from openscm.units import UnitConverter
# from openscm.utils import convert_openscm_time_to_datetime

from .base import ScmDataFrameBase, df_append  # noqa: F401

ONE_YEAR_IN_S_INTEGER = int(round(UnitConverter("year", "s").convert_from(1)))


class ScmDataFrame(ScmDataFrameBase):
    """
    OpenSCM's custom DataFrame implementation.

    The ScmDataFrame implements a subset of the functionality provided by `pyam`'s
    IamDataFrame, but is focused on the providing a performant way of storing
    time series data and the metadata associated with those time series.

    For users who wish to take advantage of all of Pyam's functionality, please cast
    your data frame to an IamDataFrame first with `to_iamdataframe()`. Note: this
    operation can be relatively computationally expensive for large data sets.
    """


# TODO: decide what to do about this legacy code
# def convert_core_to_scmdataframe(
#     core: Core,
#     period_length: int = ONE_YEAR_IN_S_INTEGER,
#     model: str = "unspecified",
#     scenario: str = "unspecified",
#     climate_model: str = "unspecified",
# ) -> ScmDataFrame:
#     def get_metadata(c, para):
#         md = {}
#         if para._children:
#             for _, child_para in para._children.items():
#                 md.update(get_metadata(c, child_para))
#         is_time_data = para._info._type == ParameterType.TIMESERIES
#         if (para._info._type is None) or is_time_data:
#             return metadata

#         variable = value.info.name
#         if para._info._type == ParameterType.BOOLEAN:
#             values = para._data
#             label = "{}".format(variable)
#         elif para._info._type == ParameterType.ARRAY:
#             values = tuple(para._data)  # pandas indexes must be hashable
#             label = "{} ({})".format(variable, para.info.unit)
#         else:
#             values = para._data
#             label = "{} ({})".format(variable, para.info.unit)

#         metadata[label] = [values]
#         return metadata

#     def get_scmdataframe_timeseries_columns(core_in, metadata_in):
#         def get_ts_ch(core_here, para_here, ts_in, time_in, ch_in):
#             if para_here._children:
#                 for _, child_para in para_here._children.items():
#                     ts_in, time_in, ch_in = get_ts_ch(
#                         core_here, child_para, ts_in, time_in, ch_in
#                     )
#             if not para_here._info._type == ParameterType.TIMESERIES:
#                 return ts_in, time_in, ch_in

#             unit = para_here.info.unit
#             tview = core.parameters.get_timeseries_view(
#                 para_here.full_name,
#                 para_here.info.region,
#                 unit,
#                 core_here.start_time,
#                 period_length,
#             )
#             values = tview.get_series()
#             time = np.array(
#                 [convert_openscm_time_to_datetime(int(t)) for t in tview.get_times()]
#             )
#             if time_in is None:
#                 time_in = time
#             else:
#                 if not (time_in == time).all():
#                     raise AssertionError("Time axes do not match")

#             ts_in.append(values)
#             ch_in["unit"].append(unit)
#             ch_in["variable"].append(
#                 ScmDataFrame.DATA_HIERARCHY_SEPARATOR.join(para_here.full_name)
#             )
#             ch_in["region"].append(
#                 ScmDataFrame.DATA_HIERARCHY_SEPARATOR.join(para_here.info.region)
#             )

#             return ts_in, time_in, ch_in

#         ts = []
#         time_axis = None
#         column_headers = {"variable": [], "region": [], "unit": []}
#         for key, value in core_in.parameters._root._parameters.items():
#             ts, time_axis, column_headers = get_ts_ch(
#                 core_in, value, ts, time_axis, column_headers
#             )

#         return (
#             pd.DataFrame(np.vstack(ts).T, pd.Index(time_axis)),
#             {**metadata, **column_headers},
#         )

#     metadata = {
#         "climate_model": [climate_model],
#         "scenario": [scenario],
#         "model": [model],
#     }

#     for key, value in core.parameters._root._parameters.items():
#         metadata.update(get_metadata(core, value))

#     timeseries, columns = get_scmdataframe_timeseries_columns(core, metadata)
#     # convert timeseries to dataframe with time index here
#     return ScmDataFrame(timeseries, columns=columns)


# def convert_config_dict_to_parameter_set(config):
#     parameters = ParameterSet()
#     for key, (region, value) in config.items():
#         view = parameters.get_writable_scalar_view(key, region, str(value.units))
#         view.set(value.magnitude)

#     return parameters
