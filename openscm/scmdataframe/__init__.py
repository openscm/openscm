"""
ScmDataFrame provides a high level analysis tool for simple climate model relevant
data. It provides a simple interface for reading/writing, subsetting and visualising
model data. ScmDataFrames are able to hold multiple model runs which aids in analysis of
ensembles of model runs.
"""
from typing import Dict, List, Tuple

import numpy as np

from ..core.parameters import ParameterInfo, ParameterType, _Parameter
from ..core.parameterset import ParameterSet
from .base import ScmDataFrameBase, df_append  # noqa: F401


class ScmDataFrame(ScmDataFrameBase):
    """
    OpenSCM's custom DataFrame implementation.

    The ScmDataFrame implements a subset of the functionality provided by `pyam
    <https://github.com/IAMconsortium/pyam>`_'s IamDataFrame, but is focused on
    providing a performant way of storing time series data and the metadata associated
    with those time series.

    For users who wish to take advantage of all of Pyam's functionality, please cast
    your ScmDataFrame to an IamDataFrame first with :func:`to_iamdataframe`. Note: this
    operation can be computationally expensive for large data sets because IamDataFrames
    stored data in long/tidy form internally rather than ScmDataFrames' more compact
    internal format.
    """


def convert_openscm_to_scmdataframe(  # pylint: disable=too-many-locals
    parameterset: ParameterSet,
    time_points: np.ndarray,
    model: str = "unspecified",
    scenario: str = "unspecified",
    climate_model: str = "unspecified",
) -> ScmDataFrame:
    """
    Get an :class:`ScmDataFrame` from a :class:`ParameterSet`.

    An ScmDataFrame is a view with a common time index for all time series. All metadata
    in the ParameterSet must be represented as Generic parameters with in the `World`
    region.

    TODO: overhaul this function and move to an appropriate location

    Parameters
    ----------
    parameterset
        :class:`ParameterSet` containing time series and optional metadata.
    time_points
        Time points onto which all timeseries will be interpolated.
    model
        Default value for the model metadata value. This value is only used if the
        :obj:`model` parameter is not found.
    scenario
        Default value for the scenario metadata value. This value is only used if the
        :obj:`scenario` parameter is not found.
    climate_model
        Default value for the climate_model metadata value. This value is only used if
        the :obj:`climate_model` parameter is not found.

    Raises
    ------
    ValueError
        If a generic parameter cannot be mapped to an ScmDataFrame meta table. This
        happens if the parameter has a region which is not ``('World',)``.

    Returns
    -------
    :class:`ScmDataFrame`
        :class:`ScmDataFrame` containing the data from :obj:`parameterset`
    """
    time_points = np.asarray(
        time_points, dtype="datetime64[s]"
    )  # TODO: check this can handle many different input types
    time_points_average = np.copy(time_points)
    delta_t = time_points_average[-1] - time_points_average[-2]
    time_points_average = np.concatenate(
        [time_points_average, [time_points_average[-1] + delta_t]]
    )

    def walk_parameters(
        para: _Parameter, past: Tuple[str, ...] = ()
    ) -> Dict[Tuple, ParameterInfo]:
        md = {}
        full_para_name = past + (para.name,)
        if para.children:
            for (_, child_para) in para.children.items():
                md.update(walk_parameters(child_para, past=full_para_name))
            return md

        md[(full_para_name, para.region.full_name)] = ParameterInfo(para)
        return md

    def parameter_name_to_scm(t):
        return ScmDataFrame.data_hierarchy_separator.join(t)

    def parameter_type_to_scm(t):
        return "average" if t == ParameterType.AVERAGE_TIMESERIES else "point"

    metadata: Dict[str, List] = {
        "climate_model": [climate_model],  # TODO: auto-fill
        "scenario": [scenario],
        "model": [model],
        "variable": [],
        "region": [],
        "unit": [],
        "parameter_type": [],
    }
    data = []

    root_params: Dict[Tuple, ParameterInfo] = {}
    for (
        _,
        value,
    ) in parameterset._root._parameters.items():  # pylint: disable=protected-access
        root_params.update(walk_parameters(value))
    for (
        _,
        r,
    ) in parameterset._root._children.items():  # pylint: disable=protected-access
        for _, value in r._parameters.items():  # pylint: disable=protected-access
            root_params.update(walk_parameters(value))

    for (param_name, region), p_info in root_params.items():
        # All meta values are stored as generic value (AKA no units)
        # TODO: fix this
        if p_info.parameter_type == ParameterType.GENERIC:
            if region != ("World",):  # TODO: fix this
                raise ValueError(
                    "Only generic types with Region=World can be extracted"
                )
            metadata[parameter_name_to_scm(param_name)] = [
                parameterset.generic(param_name, region=region).value
            ]
        elif p_info.parameter_type == ParameterType.SCALAR:
            meta_key = "{} ({})".format(parameter_name_to_scm(param_name), p_info.unit)
            meta_value = parameterset.scalar(param_name, unit=str(p_info.unit)).value
            metadata[meta_key] = [meta_value]
        else:
            para_type = p_info.parameter_type
            print(para_type)
            tp = (
                time_points
                if para_type == ParameterType.POINT_TIMESERIES
                else time_points_average
            )

            ts = parameterset.timeseries(  # type: ignore
                param_name, p_info.unit, tp, region=region, timeseries_type=para_type
            )
            data.append(ts.values)
            metadata["variable"].append(parameter_name_to_scm(param_name))
            metadata["region"].append(parameter_name_to_scm(region))
            metadata["unit"].append(p_info.unit)
            metadata["parameter_type"].append(parameter_type_to_scm(para_type))

    # convert timeseries to dataframe with time index here
    return ScmDataFrame(np.atleast_2d(data).T, columns=metadata, index=time_points)
