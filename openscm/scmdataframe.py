"""
ScmDataFrame provides a high level analysis tool for simple climate model relevant
data. It provides a simple interface for reading/writing, subsetting and visualising
model data. ScmDataFrames are able to hold multiple model runs which aids in analysis of
ensembles of model runs.
"""
from __future__ import annotations

import datetime
import re
import warnings
from typing import Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from scmdata import ScmDataFrame, df_append

from openscm.core.parameters import (
    ParameterInfo,
    ParameterType,
    _Parameter,
    guess_parameter_type,
)
from openscm.core.parameterset import ParameterSet
from openscm.core.time import ExtrapolationType, InterpolationType, TimeseriesConverter
from openscm.errors import ParameterEmptyError


class OpenScmDataFrame(ScmDataFrame):
    """
    OpenSCM's custom ScmDataFrame implementation.

    The `scmdata <https://github.com/IAMconsortium/pyam>`_'s ScmDataFrame implements a subset of the
    functionality provided by `pyam <https://github.com/IAMconsortium/pyam>`_'s IamDataFrame, but is
    focused on providing a performant way of storing time series data and the metadata associated
    with those time series.

    This class adds a column named `parameter_type` and a custom interpolator which allows
    for integral preserving interpolation.

    For users who wish to take advantage of all of Pyam's functionality, please cast
    your OpenScmDataFrame to an IamDataFrame first with :func:`to_iamdataframe`. Note: this
    operation can be computationally expensive for large data sets because IamDataFrames
    stored data in long/tidy form internally rather than ScmDataFrames' more compact
    internal format.
    """

    def to_parameterset(
        self, parameterset: Optional[ParameterSet] = None
    ) -> ParameterSet:
        """
        Add parameters in this :class:`ScmDataFrameBase` to a :class:`ParameterSet`.

        It can only be transformed if all timeseries have the same metadata. This is
        typically the case if all data comes from a single scenario/model input
        dataset. If that is not the case, further filtering is needed to reduce to a
        dataframe with identical metadata.

        Parameters
        ----------
        parameterset
            ParameterSet to add this :class:`ScmDataFrameBase`'s parameters to. A new
            :class:`ParameterSet` is created if this is ``None``.

        Returns
        -------
        ParameterSet
            :class:`ParameterSet` containing the data in ``self`` (equals
            :obj:`parameterset` if not ``None``)

        Raises
        ------
        ValueError
            Not all timeseries have the same metadata or :obj:`climate_model` is given
            and does not equal "unspecified"
        """
        # pylint: disable=too-many-locals

        meta_values = self._meta.drop(
            ["variable", "region", "unit", "parameter_type"], axis=1, errors="ignore"
        ).drop_duplicates()
        if len(meta_values) > 1:
            raise ValueError("Not all timeseries have identical metadata")
        meta_values = meta_values.squeeze()

        if meta_values.get("climate_model", "unspecified") != "unspecified":
            raise ValueError(
                "Only input data can be converted to a ParameterSet. Remove climate_model first."
            )

        if parameterset is None:
            parameterset = ParameterSet()

        for i in self._data:
            vals = self._data[i]
            metadata = self._meta.loc[i]
            variable = metadata.pop("variable")
            region = metadata.pop("region")
            unit = metadata.pop("unit")
            try:
                timeseries_type = ParameterType.from_timeseries_type(
                    metadata.pop("parameter_type")
                )
            except KeyError:
                timeseries_type = guess_parameter_type(variable, unit)

            time_points = self.time_points
            if timeseries_type == ParameterType.AVERAGE_TIMESERIES:
                delta_t = time_points[-1] - time_points[-2]
                time_points = np.concatenate((time_points, [time_points[-1] + delta_t]))

            parameterset.timeseries(
                variable,
                unit,
                time_points=time_points,
                region=region,
                timeseries_type=timeseries_type,
            ).values = vals.values

        unit_regexp = re.compile(r".*\(.*\)")
        for k, v in meta_values.iteritems():
            if unit_regexp.match(k):
                para_name = k.split("(")[0].strip()
                para_unit = k.split("(")[1].split(")")[0].strip()
                parameterset.scalar(para_name, para_unit).value = v
            else:
                parameterset.generic(k).value = v

        return parameterset

    def interpolate(  # pylint: disable=too-many-locals
        self,
        target_times: Union[np.ndarray, List[Union[datetime.datetime, int]]],
        interpolation_type: Union[InterpolationType, str] = InterpolationType.LINEAR,
        extrapolation_type: Union[ExtrapolationType, str] = ExtrapolationType.CONSTANT,
    ) -> OpenScmDataFrame:
        """
        Interpolate the dataframe onto a new time frame.

        Uses :class:`openscm.timeseries_converter.TimeseriesConverter` internally. For
        each time series a :class:`ParameterType` is guessed from the variable name. To
        override the guessed parameter type, specify a "parameter_type" meta column
        before calling interpolate. The guessed parameter types are returned in meta.

        Parameters
        ----------
        target_times
            Time grid onto which to interpolate

        interpolation_type
            How to interpolate the data between timepoints

        extrapolation_type
            If and how to extrapolate the data beyond the data in
            :func:`self.timeseries()`

        Returns
        -------
        :obj:`OpenScmDataFrame`
            A new :class:`OpenScmDataFrame` containing the data interpolated onto the
            :obj:`target_times` grid
        """
        # pylint: disable=protected-access

        target_times = np.asarray(target_times, dtype="datetime64[s]")

        # Need to keep an object index or pandas will not be able to handle a wide
        # time range
        timeseries_index = pd.Index(
            target_times.astype(object), dtype="object", name="time"
        )

        res = self.copy()

        # Add in a parameter_type column if it doesn't exist
        if "parameter_type" not in res._meta:
            res._meta["parameter_type"] = None
            res._sort_meta_cols()

        def guess(r):
            if r.parameter_type is None:
                warnings.warn(
                    "`parameter_type` metadata not available. Guessing parameter types where unavailable."
                )
                parameter_type = guess_parameter_type(r.variable, r.unit)
                r.parameter_type = (
                    "average"
                    if parameter_type == ParameterType.AVERAGE_TIMESERIES
                    else "point"
                )
            return r

        res._meta.apply(guess, axis=1)

        # Resize dataframe to new index length
        old_data = res._data
        res._data = pd.DataFrame(index=timeseries_index, columns=res._data.columns)

        for parameter_type, grp in res._meta.groupby("parameter_type"):
            p_type = ParameterType.from_timeseries_type(parameter_type)
            time_points = self.time_points

            if p_type == ParameterType.AVERAGE_TIMESERIES:
                # With an average time series we are making the assumption that the last value is the
                # average value between t[-1] and (t[-1] - t[-2]). This will ensure that both the
                # point and average timeseries can use the same time grid.
                delta_t = target_times[-1] - target_times[-2]
                target_times = np.concatenate(
                    (target_times, [target_times[-1] + delta_t])
                )

                delta_t = time_points[-1] - time_points[-2]
                time_points = np.concatenate((time_points, [time_points[-1] + delta_t]))

            timeseries_converter = TimeseriesConverter(
                time_points,
                target_times,
                p_type,
                InterpolationType.from_interpolation_type(interpolation_type),
                ExtrapolationType.from_extrapolation_type(extrapolation_type),
            )

            res._data[grp.index] = old_data[
                grp.index
            ].apply(  # pylint: disable=protected-access
                lambda col: pd.Series(
                    timeseries_converter.convert_from(  # pylint: disable=cell-var-from-loop
                        col.values
                    ),
                    index=timeseries_index,
                )
            )

            # Convert from ParameterType to str
            parameter_type_str = (
                "average" if p_type == ParameterType.AVERAGE_TIMESERIES else "point"
            )
            res._meta.loc[grp.index] = res._meta.loc[grp.index].assign(
                parameter_type=parameter_type_str
            )

        res["time"] = timeseries_index
        return res


def convert_openscm_to_openscmdataframe(  # pylint: disable=too-many-locals
    parameterset: ParameterSet,
    time_points: np.ndarray,
    model: str = "unspecified",
    scenario: str = "unspecified",
    climate_model: str = "unspecified",
) -> OpenScmDataFrame:
    """
    Get an :class:`OpenScmDataFrame` from a :class:`ParameterSet`.

    An OpenScmDataFrame is a view with a common time index for all time series. All metadata
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
    :class:`OpenScmDataFrame`
        :class:`OpenScmDataFrame` containing the data from :obj:`parameterset`
    """
    time_points = np.asarray(time_points, dtype="datetime64[s]")
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
        return OpenScmDataFrame.data_hierarchy_separator.join(t)

    metadata: Dict[str, List[Union[float, str]]] = {
        "climate_model": [climate_model],
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
            if region != ("World",):
                raise ValueError(
                    "Only generic types with Region==World can be extracted"
                )
            value = parameterset.generic(param_name, region=region).value
            if isinstance(value, list):
                value = tuple(value)

            meta_key = parameter_name_to_scm(param_name)
            if meta_key in ["model", "scenario", "climate_model"]:
                continue  # should always come from kwargs
            metadata[meta_key] = [value]
        elif p_info.parameter_type == ParameterType.SCALAR:
            if region != ("World",):
                raise ValueError(
                    "Only scalar types with Region==World can be extracted"
                )
            meta_key = "{} ({})".format(parameter_name_to_scm(param_name), p_info.unit)

            try:
                meta_value = parameterset.scalar(
                    param_name, unit=cast(str, p_info.unit)
                ).value
            except ParameterEmptyError:  # hack hack hack
                continue
            metadata[meta_key] = [meta_value]
        else:
            para_type = cast(ParameterType, p_info.parameter_type)
            tp = (
                time_points
                if para_type == ParameterType.POINT_TIMESERIES
                else time_points_average
            )

            ts = parameterset.timeseries(
                param_name,
                cast(str, p_info.unit),
                time_points=tp,
                region=region,
                timeseries_type=para_type,
            )
            try:
                data.append(ts.values)
            except ParameterEmptyError:
                print("Empty {}".format(param_name))
                continue
            metadata["variable"].append(parameter_name_to_scm(param_name))
            metadata["region"].append(parameter_name_to_scm(region))
            metadata["unit"].append(cast(str, p_info.unit))
            metadata["parameter_type"].append(
                ParameterType.timeseries_type_to_string(para_type)
            )

    # convert timeseries to dataframe with time index here
    return OpenScmDataFrame(np.atleast_2d(data).T, columns=metadata, index=time_points)
