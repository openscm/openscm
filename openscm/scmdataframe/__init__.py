"""
ScmDataFrame provides a high level analysis tool for simple climate
model relevant data. It provides a simple interface for reading/writing,
subsetting and visualising model data. ScmDataFrames are able to hold
multiple model runs which aids in analysis of ensembles of model runs.
"""
from typing import Dict, List, Tuple

import numpy as np

from openscm.core import Core
from openscm.parameters import ParameterInfo, ParameterType
from openscm.utils import convert_openscm_time_to_datetime

from .base import ScmDataFrameBase, df_append  # noqa: F401


class ScmDataFrame(ScmDataFrameBase):
    """
    OpenSCM's custom DataFrame implementation.

    The ScmDataFrame implements a subset of the functionality provided by
    `pyam <https://github.com/IAMconsortium/pyam>`_'s
    IamDataFrame, but is focused on providing a performant way of storing
    time series data and the metadata associated with those time series.

    For users who wish to take advantage of all of Pyam's functionality, please cast
    your ScmDataFrame to an IamDataFrame first with `to_iamdataframe()`. Note: this
    operation can be computationally expensive for large data sets because
    IamDataFrames stored data in long/tidy form internally rather than ScmDataFrames'
    more compact internal format.
    """


def convert_core_to_scmdataframe(  # pylint: disable=missing-return-doc
    core: Core,
    time_points: List[int],
    model: str = "unspecified",
    scenario: str = "unspecified",
    climate_model: str = "unspecified",
) -> ScmDataFrame:
    """
    Get an ScmDataFrame from a Core object

    An ScmDataFrame is a view with a common time index for all time series. All metadata in Core must be represented as Generic
    parameters with in the `World` region.

    Parameters
    ----------
    core
        Core object containing time series and optional metadata.
    time_points
        List of OpenSCM time values to which all timeseries will be interpolated.
    model
        Default value for the model metadata value. This value is only used if the `model` parameter is not found.
    scenario
        Default value for the scenario metadata value. This value is only used if the `scenario` parameter is not found.
    climate_model
        Default value for the climate_model metadata value. This value is only used if the `climate_model` parameter is not found.

    Raises
    ------
    ValueError
        If a generic parameter cannot be mapped to an ScmDataFrame meta table. This happens if the parameter has a region which is
        not `('World',)`

    Returns
    -------
    :obj:`ScmDataFrame`
    """
    time_points = np.asarray(time_points)

    def walk_parameters(  # type: ignore
        c: Core, para, past=()
    ) -> Dict[Tuple, ParameterInfo]:
        md = {}
        full_para_name = past + (para.info.name,)
        if para._children:  # pylint: disable=protected-access
            for (
                _,
                child_para,
            ) in para._children.items():  # pylint: disable=protected-access
                md.update(walk_parameters(c, child_para, past=full_para_name))
            return md

        md[(full_para_name, para.info.region)] = para.info
        return md

    def parameter_name_to_scm(t):
        return ScmDataFrame.data_hierarchy_separator.join(t)

    metadata: Dict[str, List] = {
        "climate_model": [climate_model],
        "scenario": [scenario],
        "model": [model],
        "variable": [],
        "region": [],
        "unit": [],
    }
    data = []

    root_params: Dict[Tuple, ParameterInfo] = {}
    for (
        _,
        value,
    ) in core.parameters._root._parameters.items():  # pylint: disable=protected-access
        root_params.update(walk_parameters(core, value))

    for param_name, region in root_params:
        p_info = root_params[param_name, region]

        # All meta values are stored as generic value (AKA no units)
        if p_info.parameter_type == ParameterType.GENERIC:
            if region != ("World",):  # pragma: no cover
                raise ValueError(
                    "Only generic types with Region=World can be extracted"
                )
            metadata[parameter_name_to_scm(param_name)] = [
                core.parameters.get_generic_view(param_name, region).get()
            ]
        else:
            ts = core.parameters.get_timeseries_view(  # type: ignore
                param_name, region, p_info.unit, time_points, p_info.parameter_type
            )
            data.append(ts.get())
            metadata["variable"].append(parameter_name_to_scm(param_name))
            metadata["region"].append(parameter_name_to_scm(region))
            metadata["unit"].append(p_info.unit)

    # convert timeseries to dataframe with time index here
    return ScmDataFrame(
        np.atleast_2d(data).T,
        columns=metadata,
        index=[convert_openscm_time_to_datetime(t) for t in time_points],
    )
