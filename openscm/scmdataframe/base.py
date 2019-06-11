"""
Base and utilities for OpenSCM's custom DataFrame implementation.
"""
from __future__ import annotations

import copy
import datetime
import os
import re
import warnings
from logging import getLogger
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dateutil import parser

from ..core.parameters import HIERARCHY_SEPARATOR
from ..core.parameterset import ParameterSet
from ..core.time import (
    ExtrapolationType,
    InterpolationType,
    ParameterType,
    TimePoints,
    TimeseriesConverter,
)
from ..core.units import UnitConverter
from .filters import (
    datetime_match,
    day_match,
    hour_match,
    month_match,
    pattern_match,
    years_match,
)
from .offsets import generate_range, to_offset
from .parameter_type import guess_parameter_type
from .pyam_compat import Axes, IamDataFrame, LongDatetimeIamDataFrame

_logger = getLogger(__name__)

REQUIRED_COLS: List[str] = ["model", "scenario", "region", "variable", "unit"]
"""Minimum metadata columns required by an ScmDataFrame"""


# pylint doesn't recognise return statements if they include ','
def _read_file(  # pylint: disable=missing-return-doc
    fnames: str, *args: Any, **kwargs: Any
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data to initialize :class:`ScmDataFrameBase` from a file.

    Parameters
    ----------
    *args
        Passed to :func:`_read_pandas`.
    **kwargs
        Passed to :func:`_read_pandas`.

    Returns
    -------
    :obj:`pd.DataFrame`, :obj:`pd.DataFrame`
        First dataframe is the data. Second dataframe is metadata
    """
    _logger.info("Reading %s", fnames)

    return _format_data(_read_pandas(fnames, *args, **kwargs))


def _read_pandas(fname: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Read a file and return a :class:`pd.DataFrame`.

    Parameters
    ----------
    fname
        Path from which to read data
    *args
        Passed to :func:`pd.read_csv` if :obj:`fname` ends with '.csv', otherwise passed
        to :func:`pd.read_excel`.
    **kwargs
        Passed to :func:`pd.read_csv` if :obj:`fname` ends with '.csv', otherwise passed
        to :func:`pd.read_excel`.

    Returns
    -------
    :obj:`pd.DataFrame`
        Read data

    Raises
    ------
    OSError
        Path specified by :obj:`fname` does not exist
    """
    if not os.path.exists(fname):
        raise OSError("no data file `{}` found!".format(fname))
    if fname.endswith("csv"):
        df = pd.read_csv(fname, *args, **kwargs)
    else:
        xl = pd.ExcelFile(fname)
        if len(xl.sheet_names) > 1 and "sheet_name" not in kwargs:
            kwargs["sheet_name"] = "data"
        df = pd.read_excel(fname, *args, **kwargs)

    return df


# pylint doesn't recognise return statements if they include ','
def _format_data(  # pylint: disable=missing-return-doc
    df: Union[pd.DataFrame, pd.Series]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data to initialize :class:`ScmDataFrameBase` from :class:`pd.DataFrame` or
    :class:`pd.Series`.

    See docstring of :func:`ScmDataFrameBase.__init__` for details.

    Parameters
    ----------
    df
        Data to format.

    Returns
    -------
    :obj:`pd.DataFrame`, :obj:`pd.DataFrame`
        First dataframe is the data. Second dataframe is metadata.

    Raises
    ------
    ValueError
        Not all required metadata columns are present or the time axis cannot be
        understood
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # all lower case
    str_cols = [c for c in df.columns if isinstance(c, str)]
    df.rename(columns={c: str(c).lower() for c in str_cols}, inplace=True)

    # reset the index if meaningful entries are included there
    if list(df.index.names) != [None]:
        df.reset_index(inplace=True)

    if not set(REQUIRED_COLS).issubset(set(df.columns)):
        missing = list(set(REQUIRED_COLS) - set(df.columns))
        raise ValueError("missing required columns `{}`!".format(missing))

    # check whether data in wide or long format
    if "value" in df.columns:
        df, meta = _format_long_data(df)
    else:
        df, meta = _format_wide_data(df)

    # sort data
    df.sort_index(inplace=True)

    return df, meta


def _format_long_data(df):
    # check if time column is given as `year` (int) or `time` (datetime)
    cols = set(df.columns)
    if "year" in cols and "time" not in cols:
        time_col = "year"
    elif "time" in cols and "year" not in cols:
        time_col = "time"
    else:
        msg = "invalid time format, must have either `year` or `time`!"
        raise ValueError(msg)

    extra_cols = list(set(cols) - set(REQUIRED_COLS + [time_col, "value"]))
    df = df.pivot_table(columns=REQUIRED_COLS + extra_cols, index=time_col).value
    meta = df.columns.to_frame(index=None)
    df.columns = meta.index

    return df, meta


def _format_wide_data(df):
    orig = df.copy()

    cols = set(df.columns) - set(REQUIRED_COLS)
    time_cols, extra_cols = False, []
    for i in cols:
        # if in wide format, check if columns are years (int) or datetime
        if isinstance(i, datetime.datetime):
            time_cols = True
        else:
            try:
                float(i)
                time_cols = True
            except (ValueError, TypeError):
                try:
                    try:
                        # most common format
                        datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        # this is super slow so avoid if possible
                        parser.parse(str(i))  # if no ValueError, this is datetime
                    time_cols = True
                except ValueError:
                    extra_cols.append(i)  # some other string

    if not time_cols:
        msg = "invalid column format, must contain some time (int, float or datetime) columns!"
        raise ValueError(msg)

    df = df.drop(REQUIRED_COLS + extra_cols, axis="columns").T
    df.index.name = "time"
    meta = orig[REQUIRED_COLS + extra_cols].set_index(df.columns)

    return df, meta


def _from_ts(
    df: Any, index: Any = None, **columns: Union[str, bool, float, int, List]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data to initialize :class:`ScmDataFrameBase` from wide timeseries.

    See docstring of :func:`ScmDataFrameBase.__init__` for details.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        First dataframe is the data. Second dataframe is metadata

    Raises
    ------
    ValueError
        Not all required columns are present
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if index is not None:
        if isinstance(index, np.ndarray):
            df.index = TimePoints(index).to_index()
        else:
            df.index = index

    # format columns to lower-case and check that all required columns exist
    if not set(REQUIRED_COLS).issubset(columns.keys()):
        missing = list(set(REQUIRED_COLS) - set(columns.keys()))
        raise ValueError("missing required columns `{}`!".format(missing))

    df.index.name = "time"

    num_ts = len(df.columns)
    for c_name, col in columns.items():
        col_list = (
            [col] if isinstance(col, str) or not isinstance(col, Iterable) else col
        )

        if len(col_list) == num_ts:
            continue
        if len(col_list) != 1:
            error_msg = (
                "Length of column '{}' is incorrect. It should be length "
                "1 or {}".format(c_name, num_ts)
            )
            raise ValueError(error_msg)
        columns[c_name] = col_list * num_ts

    meta = pd.DataFrame(columns, index=df.columns)

    return df, meta


class ScmDataFrameBase:  # pylint: disable=too-many-public-methods
    """
    Base of OpenSCM's custom DataFrame implementation.

    This base is the class other libraries can subclass. Having such a subclass avoids a
    potential circularity where e.g. OpenSCM imports ScmDataFrame as well as Pymagicc,
    but Pymagicc wants to import ScmDataFrame too. Hence, importing ScmDataFrame
    requires importing ScmDataFrame, causing a circularity.
    """

    _data: pd.DataFrame
    """Timeseries data"""

    _meta: pd.DataFrame
    """Meta data"""

    _time_points: TimePoints
    """Time points"""

    data_hierarchy_separator = HIERARCHY_SEPARATOR
    """
    str: String used to define different levels in our data hierarchies.

    By default we follow pyam and use "|". In such a case, emissions of CO2 for energy
    from coal would be "Emissions|CO2|Energy|Coal".
    """

    def __init__(
        self,
        data: Union[
            ScmDataFrameBase, IamDataFrame, pd.DataFrame, pd.Series, np.ndarray, str
        ],
        index: Any = None,
        columns: Optional[Dict[str, list]] = None,
        **kwargs: Any
    ):
        """
        Initialize.

        Parameters
        ----------
        data
            A pd.DataFrame or data file with IAMC-format data columns, or a numpy array
            of timeseries data if :obj:`columns` is specified. If a string is passed,
            data will be attempted to be read from file.

        index
            Only used if :obj:`columns` is not ``None``. If :obj:`index` is not
            ``None``, too, then this value sets the time index of the
            :class:`ScmDataFrameBase` instance. If :obj:`index` is ``None`` and
            :obj:`columns` is not ``None``, the index is taken from :obj:`data`.

        columns
            If None, ScmDataFrameBase will attempt to infer the values from the source.
            Otherwise, use this dict to write the metadata for each timeseries in data.
            For each metadata key (e.g. "model", "scenario"), an array of values (one
            per time series) is expected. Alternatively, providing a list of length 1
            applies the same value to all timeseries in data. For example, if you had
            three timeseries from 'rcp26' for 3 different models 'model', 'model2' and
            'model3', the column dict would look like either 'col_1' or 'col_2':

            .. code:: python

                >>> col_1 = {
                    "scenario": ["rcp26"],
                    "model": ["model1", "model2", "model3"],
                    "region": ["unspecified"],
                    "variable": ["unspecified"],
                    "unit": ["unspecified"]
                }
                >>> col_2 = {
                    "scenario": ["rcp26", "rcp26", "rcp26"],
                    "model": ["model1", "model2", "model3"],
                    "region": ["unspecified"],
                    "variable": ["unspecified"],
                    "unit": ["unspecified"]
                }
                >>> assert pd.testing.assert_frame_equal(
                    ScmDataFrameBase(d, columns=col_1).meta,
                    ScmDataFrameBase(d, columns=col_2).meta
                )

        **kwargs:
            Additional parameters passed to :func:`pyam.core._read_file` to read files

        Raises
        ------
        ValueError
            If metadata for ['model', 'scenario', 'region', 'variable', 'unit'] is not
            found. A :class:`ValueError` is also raised if you try to load from multiple
            files at once. If you wish to do this, please use :func:`df_append` instead.

        TypeError
            Timeseries cannot be read from :obj:`data`
        """
        if columns is not None:
            (_df, _meta) = _from_ts(data, index=index, **columns)
        elif isinstance(data, ScmDataFrameBase):
            # turn off mypy type checking here as ScmDataFrameBase isn't defined
            # when mypy does type checking
            (_df, _meta) = (
                data._data.copy(),  # pylint: disable=protected-access
                data._meta.copy(),  # pylint: disable=protected-access
            )
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            (_df, _meta) = _format_data(data.copy())
        elif (IamDataFrame is not None) and isinstance(data, IamDataFrame):
            (_df, _meta) = _format_data(data.data.copy())
        else:
            if not isinstance(data, str):
                if isinstance(data, list) and isinstance(data[0], str):
                    raise ValueError(
                        "Initialising from multiple files not supported, "
                        "use `openscm.ScmDataFrame.append()`"
                    )
                error_msg = "Cannot load {} from {}".format(type(self), type(data))
                raise TypeError(error_msg)

            (_df, _meta) = _read_file(data, **kwargs)

        self._time_points = TimePoints(_df.index.values)
        _df.index = self._time_points.to_index()
        _df = _df.astype(float)

        self._data, self._meta = (_df, _meta)
        self._sort_meta_cols()

    def copy(self) -> ScmDataFrameBase:
        """
        Return a :func:`copy.deepcopy` of self.

        Returns
        -------
        :obj:`ScmDataFrameBase`
            :func:`copy.deepcopy` of ``self``
        """
        return copy.deepcopy(self)

    def _sort_meta_cols(self):
        # First columns are from REQUIRED_COLS and the remainder of the columns are alphabetically sorted
        self._meta = self._meta[
            REQUIRED_COLS + sorted(list(set(self._meta.columns) - set(REQUIRED_COLS)))
        ]

    def __len__(self) -> int:
        """
        Get the number of timeseries.
        """
        return len(self._meta)

    def __getitem__(self, key: Any) -> Any:
        """
        Get item of self with helpful direct access.

        Provides direct access to "time", "year" as well as the columns in :attr:`meta`.
        If key is anything else, the key will be applied to :attr:`_data`.
        """
        _key_check = (
            [key] if isinstance(key, str) or not isinstance(key, Iterable) else key
        )
        if key == "time":
            return pd.Series(self._time_points.to_index(), dtype="object")
        if key == "year":
            return pd.Series(self._time_points.years())
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__getitem__(key)

        raise KeyError("I don't know what to do with key: {}".format(key))

    def __setitem__(  # pylint: disable=inconsistent-return-statements
        self, key: Any, value: Any
    ) -> Any:
        """
        Set item of self with helpful direct access.

        Provides direct access to set "time" as well as the columns in :attr:`meta`.
        """
        if key == "time":
            # TODO: double check if this will actually do what we want
            self._time_points = TimePoints(value)
            self._data.index = self._time_points.to_index()
            return value
        return self.set_meta(value, name=key)

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
                time_points,
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

    @property
    def time_points(self) -> np.ndarray:
        """
        Time points of the data
        """
        return self._time_points.values

    def timeseries(self, meta: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Return the data in wide format (same as the timeseries method of
        :class:`pyam.IamDataFrame`).

        Parameters
        ----------
        meta
            The list of meta columns that will be included in the output's MultiIndex.
            If None (default), then all metadata will be used.

        Returns
        -------
        :obj:`pd.DataFrame`
            DataFrame with datetimes as columns and timeseries as rows. Metadata is in
            the index.

        Raises
        ------
        ValueError
            If the metadata are not unique between timeseries
        """
        d = self._data.copy()
        meta_subset = self._meta if meta is None else self._meta[meta]
        if meta_subset.duplicated().any():
            raise ValueError("Duplicated meta values")

        d.columns = pd.MultiIndex.from_arrays(
            meta_subset.values.T, names=meta_subset.columns
        )

        return d.T

    @property
    def values(self) -> np.ndarray:
        """
        Timeseries values without metadata

        Calls :func:`timeseries`
        """
        return self.timeseries().values

    @property
    def meta(self) -> pd.DataFrame:
        """
        Metadata
        """
        return self._meta.copy()

    def filter(
        self,
        keep: bool = True,
        inplace: bool = False,
        has_nan: bool = True,
        **kwargs: Any
    ) -> Optional[ScmDataFrameBase]:
        """
        Return a filtered ScmDataFrame (i.e., a subset of the data).

        Parameters
        ----------
        keep
            If True, keep all timeseries satisfying the filters, otherwise drop all the
            timeseries satisfying the filters

        inplace
            If True, do operation inplace and return None

        has_nan
            If ``True``, convert all nan values in :obj:`meta_col` to empty string
            before applying filters. This means that "" and "*" will match rows with
            :class:`np.nan`. If ``False``, the conversion is not applied and so a search
            in a string column which contains ;class:`np.nan` will result in a
            :class:`TypeError`.

        **kwargs
            Argument names are keys with which to filter, values are used to do the
            filtering. Filtering can be done on:

            - all metadata columns with strings, "*" can be used as a wildcard in search
              strings

            - 'level': the maximum "depth" of IAM variables (number of hierarchy levels,
              excluding the strings given in the 'variable' argument)

            - 'time': takes a :class:`datetime.datetime` or list of
              :class:`datetime.datetime`'s
              TODO: default to np.datetime64

            - 'year', 'month', 'day', hour': takes an :class:`int` or list of
              :class:`int`'s ('month' and 'day' also accept :class:`str` or list of
              :class:`str`)

            If ``regexp=True`` is included in :obj:`kwargs` then the pseudo-regexp
            syntax in :obj:`pattern_match` is disabled.

        Returns
        -------
        :obj:`ScmDataFrameBase`
            If not :obj:`inplace`, return a new instance with the filtered data.

        Raises
        ------
        AssertionError
            Data and meta become unaligned
        """
        _keep_ts, _keep_cols = self._apply_filters(kwargs, has_nan)
        idx = _keep_ts[:, np.newaxis] & _keep_cols
        if not idx.shape == self._data.shape:
            raise AssertionError(
                "Index shape does not match data shape"
            )  # pragma: no cover  # don't think it's possible to get here...
        idx = idx if keep else ~idx

        ret = self.copy() if not inplace else self
        d = ret._data.where(idx)  # pylint: disable=protected-access
        ret._data = d.dropna(  # pylint: disable=protected-access
            axis=1, how="all"
        ).dropna(axis=0, how="all")
        ret._meta = ret._meta[  # pylint: disable=protected-access
            (~d.isna()).sum(axis=0) > 0
        ]
        ret["time"] = ret._data.index.values  # pylint: disable=protected-access

        if not (
            len(ret._data.columns) == len(ret._meta)  # pylint: disable=protected-access
        ):
            raise AssertionError(
                "Data and meta have become unaligned"
            )  # pragma: no cover  # don't think it's possible to get here...

        if not ret._meta.shape[0]:  # pylint: disable=protected-access
            _logger.warning("Filtered ScmDataFrame is empty!")

        if not inplace:
            return ret

        return None

    # pylint doesn't recognise ',' in returns type definition
    def _apply_filters(  # pylint: disable=missing-return-doc
        self, filters: Dict, has_nan: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine rows to keep in data for given set of filters.

        Parameters
        ----------
        filters
            Dictionary of filters ``({col: values}})``; uses a pseudo-regexp syntax by
            default but if ``filters["regexp"]`` is ``True``, regexp is used directly.

        has_nan
            If `True``, convert all nan values in :obj:`meta_col` to empty string before
            applying filters. This means that "" and "*" will match rows with
            :class:`np.nan`. If ``False``, the conversion is not applied and so a search
            in a string column which contains :class:`np.nan` will result in a
            :class:`TypeError`.

        Returns
        -------
        :obj:`np.ndarray` of :class:`bool`, :obj:`np.ndarray` of :class:`bool`
            Two boolean :class:`np.ndarray`'s. The first contains the columns to keep
            (i.e. which time points to keep). The second contains the rows to keep (i.e.
            which metadata matched the filters).

        Raises
        ------
        ValueError
            Filtering cannot be performed on requested column
        """
        regexp = filters.pop("regexp", False)
        keep_ts = np.array([True] * len(self._data))
        keep_meta = np.array([True] * len(self.meta))

        # filter by columns and list of values
        for col, values in filters.items():
            if col == "variable":
                level = filters["level"] if "level" in filters else None
                keep_meta &= pattern_match(
                    self.meta[col],
                    values,
                    level,
                    regexp,
                    has_nan=has_nan,
                    separator=self.data_hierarchy_separator,
                ).values
            elif col in self.meta.columns:
                keep_meta &= pattern_match(
                    self.meta[col],
                    values,
                    regexp=regexp,
                    has_nan=has_nan,
                    separator=self.data_hierarchy_separator,
                ).values
            elif col == "year":
                keep_ts &= years_match(self._time_points.years(), values)

            elif col == "month":
                keep_ts &= month_match(self._time_points.months(), values)

            elif col == "day":
                keep_ts &= self._day_match(values)

            elif col == "hour":
                keep_ts &= hour_match(self._time_points.hours(), values)

            elif col == "time":
                keep_ts &= datetime_match(self._time_points.values, values)

            elif col == "level":
                if "variable" not in filters.keys():
                    keep_meta &= pattern_match(
                        self.meta["variable"],
                        "*",
                        values,
                        regexp=regexp,
                        has_nan=has_nan,
                        separator=self.data_hierarchy_separator,
                    ).values
                # else do nothing as level handled in variable filtering

            else:
                raise ValueError("filter by `{}` not supported".format(col))

        return keep_ts, keep_meta

    def _day_match(self, values):
        if isinstance(values, str):
            wday = True
        elif isinstance(values, list) and isinstance(values[0], str):
            wday = True
        else:
            wday = False

        if wday:
            days = self._time_points.weekdays()
        else:  # ints or list of ints
            days = self._time_points.days()

        return day_match(days, values)

    def head(self, *args, **kwargs):
        """
        Return head of :func:`self.timeseries()`.

        Parameters
        ----------
        *args
            Passed to :func:`self.timeseries().head()`

        **kwargs
            Passed to :func:`self.timeseries().head()`

        Returns
        -------
        :obj:`pd.DataFrame`
            Tail of :func:`self.timeseries()`
        """
        return self.timeseries().head(*args, **kwargs)

    def tail(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return tail of :func:`self.timeseries()`.

        Parameters
        ----------
        *args
            Passed to :func:`self.timeseries().tail()`

        **kwargs
            Passed to :func:`self.timeseries().tail()`

        Returns
        -------
        :obj:`pd.DataFrame`
            Tail of :func:`self.timeseries()`
        """
        return self.timeseries().tail(*args, **kwargs)

    def rename(
        self, mapping: Dict[str, Dict[str, str]], inplace: bool = False
    ) -> Optional[ScmDataFrameBase]:
        """
        Rename and aggregate column entries using :func:`groupby.sum()` on values. When
        renaming models or scenarios, the uniqueness of the index must be maintained,
        and the function will raise an error otherwise.

        Parameters
        ----------
        mapping
            For each column where entries should be renamed, provide current
            name and target name

            .. code:: python

                {<column name>: {<current_name_1>: <target_name_1>,
                                 <current_name_2>: <target_name_2>}}

        inplace
            If ``True``, do operation inplace and return ``None``

        Returns
        -------
        :obj:`ScmDataFrameBase`
            If :obj:`inplace` is ``True``, return a new :class:`ScmDataFrameBase`
            instance

        Raises
        ------
        ValueError
            Column is not in meta or renaming will cause non-unique metadata
        """
        ret = copy.deepcopy(self) if not inplace else self
        for col, _mapping in mapping.items():
            if col not in self.meta.columns:
                raise ValueError("Renaming by {} not supported!".format(col))
            ret._meta[col] = ret._meta[col].replace(  # pylint: disable=protected-access
                _mapping
            )
            if ret._meta.duplicated().any():  # pylint: disable=protected-access
                raise ValueError("Renaming to non-unique metadata for {}!".format(col))

        if not inplace:
            return ret

        return None

    def set_meta(
        self,
        meta: Union[pd.Series, list, int, float, str],
        name: Optional[str] = None,
        index: Optional[Union[pd.DataFrame, pd.Series, pd.Index, pd.MultiIndex]] = None,
    ) -> None:
        """
        Set metadata information.

        TODO: re-write this to make it more sane and add type annotations

        Parameters
        ----------
        meta
            Column to be added to metadata

        name
            Meta column name (defaults to :obj:`meta.name`)

        index
            The index to which the metadata is to be applied

        Raises
        ------
        ValueError
            No name can be determined from inputs or index cannot be coerced to
            :class:`pd.MultiIndex`
        """
        # check that name is valid and doesn't conflict with data columns
        # mypy doesn't recognise if
        if (name or (hasattr(meta, "name") and meta.name)) in [  # type: ignore
            None,
            False,
        ]:
            raise ValueError("Must pass a name or use a named pd.Series")
        # mpy doesn't recognise if checks
        name = name or meta.name  # type: ignore

        # check if meta has a valid index and use it for further workflow
        # mypy doesn't recognise if
        if hasattr(meta, "index") and hasattr(meta.index, "names"):  # type: ignore
            index = meta.index  # type: ignore # mypy doesn't recognise if
        if index is None:
            self._meta[name] = meta
            return

        # turn dataframe to index if index arg is a DataFrame
        if isinstance(index, pd.DataFrame):
            index = index.set_index(
                index.columns.intersection(self._meta.columns).to_list()
            ).index
        if not isinstance(index, (pd.MultiIndex, pd.Index)):
            raise ValueError("index cannot be coerced to pd.MultiIndex")

        meta = pd.Series(meta, index=index, name=name)

        df = self.meta.reset_index()
        if all(index.names):
            df = df.set_index(index.names)
        self._meta = (
            pd.merge(df, meta, left_index=True, right_index=True, how="outer")
            .reset_index()
            .set_index("index")
        )
        # Edge case of using a different index on meta
        if "level_0" in self._meta:
            self._meta.drop("level_0", axis=1, inplace=True)
        self._sort_meta_cols()

    def interpolate(  # pylint: disable=too-many-locals
        self,
        target_times: Union[np.ndarray, List[Union[datetime.datetime, int]]],
        interpolation_type: Union[InterpolationType, str] = InterpolationType.LINEAR,
        extrapolation_type: Union[ExtrapolationType, str] = ExtrapolationType.CONSTANT,
    ) -> ScmDataFrameBase:
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
        :obj:`ScmDataFrameBase`
            A new :class:`ScmDataFrameBase` containing the data interpolated onto the
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

    def resample(self, rule: str = "AS", **kwargs: Any) -> ScmDataFrameBase:
        """
        Resample the time index of the timeseries data onto a custom grid.

        This helper function allows for values to be easily interpolated onto annual or
        monthly timesteps using the rules='AS' or 'MS' respectively. Internally, the
        interpolate function performs the regridding.

        Parameters
        ----------
        rule
            See the pandas `user guide
            <http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
            for a list of options. Note that Business-related offsets such as
            "BusinessDay" are not supported.

        **kwargs
            Other arguments to pass through to :func:`interpolate`

        Returns
        -------
        :obj:`ScmDataFrameBase`
            New :class:`ScmDataFrameBase` instance on a new time index

        Examples
        --------
        Resample a dataframe to annual values

        >>> scm_df = ScmDataFrame(
        ...     pd.Series([1, 2, 10], index=(2000, 2001, 2009)),
        ...     columns={
        ...         "model": ["a_iam"],
        ...         "scenario": ["a_scenario"],
        ...         "region": ["World"],
        ...         "variable": ["Primary Energy"],
        ...         "unit": ["EJ/y"],
        ...     }
        ... )
        >>> scm_df.timeseries().T
        model             a_iam
        scenario     a_scenario
        region            World
        variable Primary Energy
        unit               EJ/y
        year
        2000                  1
        2010                 10

        An annual timeseries can be the created by interpolating to the start of years
        using the rule 'AS'.

        >>> res = scm_df.resample('AS')
        >>> res.timeseries().T
        model                        a_iam
        scenario                a_scenario
        region                       World
        variable            Primary Energy
        unit                          EJ/y
        time
        2000-01-01 00:00:00       1.000000
        2001-01-01 00:00:00       2.001825
        2002-01-01 00:00:00       3.000912
        2003-01-01 00:00:00       4.000000
        2004-01-01 00:00:00       4.999088
        2005-01-01 00:00:00       6.000912
        2006-01-01 00:00:00       7.000000
        2007-01-01 00:00:00       7.999088
        2008-01-01 00:00:00       8.998175
        2009-01-01 00:00:00      10.00000

        >>> m_df = scm_df.resample('MS')
        >>> m_df.timeseries().T
        model                        a_iam
        scenario                a_scenario
        region                       World
        variable            Primary Energy
        unit                          EJ/y
        time
        2000-01-01 00:00:00       1.000000
        2000-02-01 00:00:00       1.084854
        2000-03-01 00:00:00       1.164234
        2000-04-01 00:00:00       1.249088
        2000-05-01 00:00:00       1.331204
        2000-06-01 00:00:00       1.416058
        2000-07-01 00:00:00       1.498175
        2000-08-01 00:00:00       1.583029
        2000-09-01 00:00:00       1.667883
                                    ...
        2008-05-01 00:00:00       9.329380
        2008-06-01 00:00:00       9.414234
        2008-07-01 00:00:00       9.496350
        2008-08-01 00:00:00       9.581204
        2008-09-01 00:00:00       9.666058
        2008-10-01 00:00:00       9.748175
        2008-11-01 00:00:00       9.833029
        2008-12-01 00:00:00       9.915146
        2009-01-01 00:00:00      10.000000
        [109 rows x 1 columns]


        Note that the values do not fall exactly on integer values as not all years are
        exactly the same length.

        References
        ----------
        See the pandas documentation for
        `resample <http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html>`
        for more information about possible arguments.
        """
        orig_dts = self["time"]
        target_dts = generate_range(
            orig_dts.iloc[0], orig_dts.iloc[-1], to_offset(rule)
        )
        return self.interpolate(list(target_dts), **kwargs)

    def process_over(
        self, cols: Union[str, List[str]], operation: str, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Process the data over the input columns.

        Parameters
        ----------
        cols
            Columns to perform the operation on. The timeseries will be grouped by all
            other columns in :attr:`meta`.

        operation : ['median', 'mean', 'quantile']
            The operation to perform. This uses the equivalent pandas function. Note
            that quantile means the value of the data at a given point in the cumulative
            distribution of values at each point in the timeseries, for each timeseries
            once the groupby is applied. As a result, using ``q=0.5`` is is the same as
            taking the median and not the same as taking the mean/average.

        **kwargs
            Keyword arguments to pass to the pandas operation

        Returns
        -------
        :obj:`pd.DataFrame`
            The quantiles of the timeseries, grouped by all columns in :attr:`meta`
            other than :obj:`cols`

        Raises
        ------
        ValueError
            If the operation is not one of ['median', 'mean', 'quantile']
        """
        cols = [cols] if isinstance(cols, str) else cols
        ts = self.timeseries()
        group_cols = list(set(ts.index.names) - set(cols))
        grouper = ts.groupby(group_cols)

        if operation == "median":
            return grouper.median(**kwargs)
        if operation == "mean":
            return grouper.mean(**kwargs)
        if operation == "quantile":
            return grouper.quantile(**kwargs)

        raise ValueError("operation must be one of ['median', 'mean', 'quantile']")

    def convert_unit(
        self,
        unit: str,
        context: Optional[str] = None,
        inplace: bool = False,
        **kwargs: Any
    ) -> Optional[ScmDataFrameBase]:
        """
        Convert the units of a selection of timeseries.

        Uses :class:`openscm.units.UnitConverter` to perform the conversion.

        Parameters
        ----------
        unit
            Unit to convert to. This must be recognised by
            :class:`~openscm.units.UnitConverter`.

        context
            Context to use for the conversion i.e. which metric to apply when performing
            CO2-equivalent calculations. If ``None``, no metric will be applied and
            CO2-equivalent calculations will raise :class:`DimensionalityError`.

        inplace
            If ``True``, the operation is performed inplace, updating the underlying
            data. Otherwise a new :class:`ScmDataFrameBase` instance is returned.

        **kwargs
            Extra arguments which are passed to :func:`~ScmDataFrameBase.filter` to
            limit the timeseries which are attempted to be converted. Defaults to
            selecting the entire ScmDataFrame, which will likely fail.

        Returns
        -------
        :obj:`ScmDataFrameBase`
            If :obj:`inplace` is not ``False``, a new :class:`ScmDataFrameBase` instance
            with the converted units.
        """
        # pylint: disable=protected-access
        ret = self if inplace else self.copy()

        if "unit_context" not in ret._meta:
            ret._meta["unit_context"] = None
            ret._sort_meta_cols()

        to_convert = ret.filter(**kwargs)
        for orig_unit, grp in to_convert._meta.groupby("unit"):  # type: ignore
            uc = UnitConverter(orig_unit, unit, context=context)
            ret._data[grp.index] = ret._data[grp.index].apply(uc.convert_from)
            # TODO: check if unit_context has changed
            ret._meta.loc[grp.index] = ret._meta.loc[grp.index].assign(
                unit=unit, unit_context=context
            )

        if not inplace:
            return ret
        return None

    def relative_to_ref_period_mean(
        self, append_str: Optional[str] = None, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Return the timeseries relative to a given reference period mean.

        The reference period mean is subtracted from all values in the input timeseries.

        Parameters
        ----------
        append_str
            String to append to the name of all the variables in the resulting DataFrame
            to indicate that they are relevant to a given reference period. E.g. `'rel.
            to 1961-1990'`. If None, this will be autofilled with the keys and ranges of
            :obj:`kwargs`.

        **kwargs
            Arguments to pass to :func:`filter` to determine the data to be included in
            the reference time period. See the docs of :func:`filter` for valid options.

        Returns
        -------
        :obj:`pd.DataFrame`
            DataFrame containing the timeseries, adjusted to the reference period mean
        """
        ts = self.timeseries()
        # mypy confused by `inplace` default
        ref_period_mean = (
            self.filter(**kwargs).timeseries().mean(axis="columns")  # type: ignore
        )

        res = ts.sub(ref_period_mean, axis="rows")
        res.reset_index(inplace=True)

        if append_str is None:
            append_str = ";".join(
                ["{}: {} - {}".format(k, v[0], v[-1]) for k, v in kwargs.items()]
            )
            append_str = "(ref. period {})".format(append_str)

        res["variable"] = res["variable"].apply(lambda x: "{} {}".format(x, append_str))

        return res.set_index(ts.index.names)

    def append(
        self,
        other: Union[
            ScmDataFrameBase, IamDataFrame, pd.DataFrame, pd.Series, np.ndarray, str
        ],
        inplace: bool = False,
        duplicate_msg: Union[str, bool] = "warn",
        **kwargs: Any
    ) -> Optional[ScmDataFrameBase]:
        """
        Append additional data to the current dataframe.

        For details, see :func:`df_append`.

        Parameters
        ----------
        other
            Data (in format which can be cast to :class:`ScmDataFrameBase`) to append

        inplace
            If ``True``, append data in place and return ``None``. Otherwise, return a
            new :class:`ScmDataFrameBase` instance with the appended data.

        duplicate_msg
            If "warn", raise a warning if duplicate data is detected. If "return",
            return the joint dataframe (including duplicate timeseries) so the user can
            inspect further. If ``False``, take the average and do not raise a warning.

        **kwargs
            Keywords to pass to :func:`ScmDataFrameBase.__init__` when reading
            :obj:`other`

        Returns
        -------
        :obj:`ScmDataFrameBase`
            If not :obj:`inplace`, return a new :class:`ScmDataFrameBase` instance
            containing the result of the append.
        """
        if not isinstance(other, ScmDataFrameBase):
            other = self.__class__(other, **kwargs)

        return df_append([self, other], inplace=inplace, duplicate_msg=duplicate_msg)

    def to_iamdataframe(self) -> LongDatetimeIamDataFrame:
        """
        Convert to a :class:`LongDatetimeIamDataFrame` instance.

        :class:`LongDatetimeIamDataFrame` is a subclass of :class:`pyam.IamDataFrame`.
        We use :class:`LongDatetimeIamDataFrame` to ensure all times can be handled, see
        docstring of :class:`LongDatetimeIamDataFrame` for details.

        Returns
        -------
        :class:`LongDatetimeIamDataFrame`
            :class:`LongDatetimeIamDataFrame` instance containing the same data.

        Raises
        ------
        ImportError
            If `pyam <https://github.com/IAMconsortium/pyam>`_ is not installed
        """
        if LongDatetimeIamDataFrame is None:
            raise ImportError(
                "pyam is not installed. Features involving IamDataFrame are unavailable"
            )

        return LongDatetimeIamDataFrame(self.timeseries())

    def to_csv(self, path: str, **kwargs: Any) -> None:
        """
        Write timeseries data to a csv file

        Parameters
        ----------
        path
            Path to write the file into
        """
        self.to_iamdataframe().to_csv(path, **kwargs)

    def line_plot(self, x: str = "time", y: str = "value", **kwargs: Any) -> Axes:
        """
        Plot a line chart.

        See :func:`pyam.IamDataFrame.line_plot` for more information.
        """
        return self.to_iamdataframe().line_plot(x, y, **kwargs)  # pragma: no cover

    def scatter(self, x: str, y: str, **kwargs: Any) -> Axes:
        """
        Plot a scatter chart using metadata columns.

        See :func:`pyam.plotting.scatter` for details.
        """
        self.to_iamdataframe().scatter(x, y, **kwargs)  # pragma: no cover

    def region_plot(self, **kwargs: Any) -> Axes:
        """
        Plot regional data for a single model, scenario, variable, and year.

        See :class:`pyam.plotting.region_plot` for details.
        """
        return self.to_iamdataframe().region_plot(**kwargs)  # pragma: no cover

    def pivot_table(
        self,
        index: Union[str, List[str]],
        columns: Union[str, List[str]],
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Pivot the underlying data series.

        See :func:`pyam.core.IamDataFrame.pivot_table` for details.
        """
        return self.to_iamdataframe().pivot_table(
            index, columns, **kwargs
        )  # pragma: no cover


def df_append(
    dfs: List[
        Union[ScmDataFrameBase, IamDataFrame, pd.DataFrame, pd.Series, np.ndarray, str]
    ],
    inplace: bool = False,
    duplicate_msg: Union[str, bool] = "warn",
) -> Optional[ScmDataFrameBase]:
    """
    Append together many objects.

    When appending many objects, it may be more efficient to call this routine once with
    a list of ScmDataFrames, than using :func:`ScmDataFrame.append` multiple times. If
    timeseries with duplicate metadata are found, the timeseries are appended and values
    falling on the same timestep are averaged (this behaviour can be adjusted with the
    :obj:`duplicate_msg` arguments).

    Parameters
    ----------
    dfs
        The dataframes to append. Values will be attempted to be cast to
        :class:`ScmDataFrameBase`.

    inplace
        If ``True``, then the operation updates the first item in :obj:`dfs` and returns
        ``None``.

    duplicate_msg
        If "warn", raise a warning if duplicate data is detected. If "return", return
        the joint dataframe (including duplicate timeseries) so the user can inspect
        further. If ``False``, take the average and do not raise a warning.

    Returns
    -------
    :obj:`ScmDataFrameBase`
        If not :obj:`inplace`, the return value is the object containing the merged
        data. The resultant class will be determined by the type of the first object. If
        ``duplicate_msg == "return"``, a `pd.DataFrame` will be returned instead.

    Raises
    ------
    TypeError
        If :obj:`inplace` is ``True`` but the first element in :obj:`dfs` is not an
        instance of :class:`ScmDataFrameBase`

    ValueError
        :obj:`duplicate_msg` option is not recognised.
    """
    scm_dfs = [
        df if isinstance(df, ScmDataFrameBase) else ScmDataFrameBase(df) for df in dfs
    ]
    joint_dfs = [d.copy() for d in scm_dfs]
    joint_meta = []  # type: List[str]
    for df in joint_dfs:
        joint_meta += df.meta.columns.tolist()

    joint_meta_set = set(joint_meta)

    # should probably solve this https://github.com/pandas-dev/pandas/issues/3729
    na_fill_value = -999
    for i, _ in enumerate(joint_dfs):
        for col in joint_meta_set:
            if col not in joint_dfs[i].meta:
                joint_dfs[i].set_meta(na_fill_value, name=col)

    # we want to put data into timeseries format and pass into format_ts instead of
    # _format_data
    data = pd.concat(
        [d.timeseries().reorder_levels(joint_meta_set) for d in joint_dfs], sort=False
    )

    data = data.reset_index()
    data[list(joint_meta_set)] = data[joint_meta_set].replace(
        to_replace=np.nan, value=na_fill_value
    )
    data = data.set_index(list(joint_meta_set))
    if duplicate_msg and data.index.duplicated().any():
        warn_handle_res = _handle_potential_duplicates_in_append(data, duplicate_msg)
        if warn_handle_res is not None:
            return warn_handle_res  # type: ignore  # special case

    data = data.groupby(data.index.names).mean()

    if inplace:
        if not isinstance(dfs[0], ScmDataFrameBase):
            raise TypeError("Can only append inplace to an ScmDataFrameBase")
        ret = dfs[0]
    else:
        ret = scm_dfs[0].copy()

    ret._data = data.reset_index(drop=True).T  # pylint: disable=protected-access
    ret._data = ret._data.sort_index()  # pylint: disable=protected-access
    ret["time"] = ret._data.index.values  # pylint: disable=protected-access
    ret._data = ret._data.astype(float)  # pylint: disable=protected-access

    ret._meta = (  # pylint: disable=protected-access
        data.index.to_frame()
        .reset_index(drop=True)
        .replace(to_replace=na_fill_value, value=np.nan)
    )
    ret._sort_meta_cols()  # pylint: disable=protected-access

    if not inplace:
        return ret

    return None


def _handle_potential_duplicates_in_append(data, duplicate_msg):
    # If only one number contributes to each of the timeseries, we're not looking at
    # duplicates so can return.
    contributing_values = (~data.isnull()).astype(int).groupby(data.index.names).sum()
    duplicates = (contributing_values > 1).any().any()
    if not duplicates:
        return None

    if duplicate_msg == "warn":
        warn_msg = (
            "Duplicate time points detected, the output will be the average of "
            "the duplicates. Set `duplicate_msg='return'` to examine the joint "
            "timeseries (the duplicates can be found by looking at "
            "`res[res.index.duplicated(keep=False)].sort_index()`. Set "
            "`duplicate_msg=False` to silence this message."
        )
        warnings.warn(warn_msg)
        return None

    if duplicate_msg == "return":
        warnings.warn("returning a `pd.DataFrame`, not an `ScmDataFrame`")
        return data

    raise ValueError("Unrecognised value for duplicate_msg")
