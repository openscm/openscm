from __future__ import annotations

import copy
import datetime
import os
from logging import getLogger
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from dateutil import parser

from openscm.core import Core
from openscm.timeseries_converter import (
    ExtrapolationType,
    InterpolationType,
    ParameterType,
    TimeseriesConverter,
)
from openscm.utils import (
    convert_datetime_to_openscm_time,
    convert_openscm_time_to_datetime,
    is_floatlike,
)

from .filters import (
    datetime_match,
    day_match,
    hour_match,
    is_str,
    month_match,
    pattern_match,
    years_match,
)
from .offsets import generate_range, to_offset
from .timeindex import TimeIndex

try:
    from pyam import IamDataFrame
    from matplotlib.axes import Axes  # matplotlib is dependency of pyam
except ImportError:
    IamDataFrame = None
    Axes = None


logger = getLogger(__name__)

REQUIRED_COLS = ["model", "scenario", "region", "variable", "unit"]


def read_files(fnames, *args, **kwargs):
    """Read data from a snapshot file saved in the standard IAMC format
    or a table with year/value columns
    """
    if not is_str(fnames):
        raise ValueError(
            "reading multiple files not supported, "
            "please use `openscm.ScmDataFrame.append()`"
        )
    logger.info("Reading `{}`".format(fnames))
    return format_data(read_pandas(fnames, *args, **kwargs))


def read_pandas(fname, *args, **kwargs):
    """Read a file and return a pd.DataFrame"""
    if not os.path.exists(fname):
        raise ValueError("no data file `{}` found!".format(fname))
    if fname.endswith("csv"):
        df = pd.read_csv(fname, *args, **kwargs)
    else:
        xl = pd.ExcelFile(fname)
        if len(xl.sheet_names) > 1 and "sheet_name" not in kwargs:
            kwargs["sheet_name"] = "data"
        df = pd.read_excel(fname, *args, **kwargs)
    return df


def format_data(df):
    """Convert an imported dataframe and check all required columns"""
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # all lower case
    str_cols = [c for c in df.columns if is_str(c)]
    df.rename(columns={c: str(c).lower() for c in str_cols}, inplace=True)

    # reset the index if meaningful entries are included there
    if not list(df.index.names) == [None]:
        # why is this so slow?
        df.reset_index(inplace=True)

    # format columns to lower-case and check that all required columns exist
    if not set(REQUIRED_COLS).issubset(set(df.columns)):
        missing = list(set(REQUIRED_COLS) - set(df.columns))
        raise ValueError("missing required columns `{}`!".format(missing))

    orig = df

    # check whether data in wide format (IAMC) or long format (`value` column)
    if "value" in df.columns:
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
    else:
        # if in wide format, check if columns are years (int) or datetime
        cols = set(df.columns) - set(REQUIRED_COLS)
        time_cols, extra_cols = False, []
        for i in cols:
            if is_floatlike(i) or isinstance(i, datetime.datetime):
                time_cols = True
            else:
                try:
                    try:
                        # most common format
                        datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
                        time_cols = True
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

    # cast value columns to numeric, drop NaN's, sort data
    # df.dropna(inplace=True, how="all")
    df.sort_index(inplace=True)

    return df, meta


def from_ts(df, index=None, **columns):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if index is not None:
        df.index = index

    # format columns to lower-case and check that all required columns exist
    if not set(REQUIRED_COLS).issubset(columns.keys()):
        missing = list(set(REQUIRED_COLS) - set(columns.keys()))
        raise ValueError("missing required columns `{}`!".format(missing))

    df.index.name = "time"

    num_ts = len(df.columns)
    for c_name in columns:
        col = columns[c_name]

        if len(col) == num_ts:
            continue
        if len(col) != 1:
            raise ValueError(
                "Length of column {} is incorrect. It should be length 1 or {}".format(
                    c_name, num_ts
                )
            )
        columns[c_name] = col * num_ts

    meta = pd.DataFrame(columns, index=df.columns)
    return df, meta


class ScmDataFrameBase(object):
    """This base is the class other libraries can subclass

    Having such a subclass avoids a potential circularity where e.g. openscm imports
    ScmDataFrame as well as Pymagicc, but Pymagicc wants to import ScmDataFrame and
    hence to try and import ScmDataFrame you have to import ScmDataFrame itself (hence
    the circularity).
    """

    DATA_HIERARCHY_SEPARATOR = "|"
    """str: String used to define different levels in our data hierarchies.

    For example, "Emissions|CO2|Energy|Coal".

    We copy this straight from pyam
    """

    def __init__(
        self,
        data: Union[ScmDataFrameBase, IamDataFrame, pd.DataFrame, np.ndarray, str],
        columns: Union[Dict[str, list], None] = None,
        climate_model: str = "unspecified",
        **kwargs: Any
    ):
        """Initialize an instance of an ScmDataFrameBase

        Parameters
        ----------
        data: ScmDataFrameBase, IamDataFrame, pd.DataFrame, np.ndarray or string
            A pd.DataFrame or data file with IAMC-format data columns, or a numpy array of timeseries data if `columns` is specified.
            If a string is passed, data will be attempted to be read from file.

        columns: dict
            If None, ScmDataFrameBase will attempt to infer the values from the source.
            Otherwise, use this dict to write the metadata for each timeseries in data. For each metadata key (e.g. "model", "scenario"), an array of values (one per time series) is expected.
            Alternatively, providing an array of length 1 applies the same value to all timeseries in data. For example, if you had three
            timeseries from 'rcp26' for 3 different models 'model', 'model2' and 'model3', the column dict would look like either `col_1` or `col_2`:

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

            Metadata for ['model', 'scenario', 'region', 'variable', 'unit'] is required, otherwise a ValueError will be raised.

        kwargs:
            Additional parameters passed to `pyam.core.read_files` to read nonstandard files
        """
        if columns is not None:
            (_df, _meta) = from_ts(data, **columns)
        elif isinstance(data, ScmDataFrameBase):
            # turn off mypy type checking here as ScmDataFrameBase isn't defined
            # when mypy does type checking
            (_df, _meta) = (data._data.copy(), data._meta.copy())  # type: ignore
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            (_df, _meta) = format_data(data.copy())
        else:
            if isinstance(data, IamDataFrame):
                # It might be a IamDataFrame?
                (_df, _meta) = format_data(data.data.copy())
            else:
                (_df, _meta) = read_files(data, **kwargs)
        self._time_index = TimeIndex(py_dt=_df.index.values)
        _df.index = self._time_index.as_pd_index()
        _df = _df.astype(float)

        self._data, self._meta = (_df, _meta)
        self._sort_meta_cols()

    def copy(self):
        return copy.deepcopy(self)

    def _sort_meta_cols(self):
        # First columns are from REQUIRED_COLS and the remainder of the columns are alphabetically sorted
        self._meta = self._meta[
            REQUIRED_COLS + sorted(list(set(self._meta.columns) - set(REQUIRED_COLS)))
        ]

    def __len__(self):
        return len(self._meta)

    def __getitem__(self, key):
        _key_check = [key] if is_str(key) else key
        if key == "time":
            return pd.Series(self._time_index.as_pd_index(), dtype="object")
        elif key == "year":
            return pd.Series(self._time_index.years())
        if set(_key_check).issubset(self.meta.columns):
            return self.meta.__getitem__(key)
        else:
            return self._data.__getitem__(key)

    def __setitem__(self, key, value):
        _key_check = [key] if is_str(key) else key

        if key == "time":
            self._time_index = TimeIndex(py_dt=value)
            self._data.index = self._time_index.as_pd_index()
            return value
        if set(_key_check).issubset(self.meta.columns):
            return self._meta.__setitem__(key, value)

    def to_core(self, climate_model: str = "unspecified") -> Core:
        """Convert a DataFrame to a Core object

        A ScmDataFrame can only be converted to a core object if all timeseries have the same metadata. This is typically the case
        if all output comes from a single model run. If that is not the case, further filtering is needed to reduce
        to a dataframe with identical metadata.

        Raises
        ------
        ValueError
            Not all timeseries have the same metadata
        """
        meta_values = self._meta.drop(
            ["variable", "region", "unit"], axis=1
        ).drop_duplicates()
        if len(meta_values) > 1:
            raise ValueError("Not all timeseries have identical metadata")
        meta_values = meta_values.squeeze()
        if "climate_model" in meta_values:
            climate_model = meta_values.pop("climate_model")

        core = Core(climate_model, self.time_points.min(), self.time_points.max())

        for i in self._data:
            vals = self._data[i]
            metadata = self._meta.loc[i]
            variable = metadata.pop("variable")
            region = metadata.pop("region")
            unit = metadata.pop("unit")

            variable_openscm = tuple(variable.split(self.DATA_HIERARCHY_SEPARATOR))
            region_openscm = tuple(region.split(self.DATA_HIERARCHY_SEPARATOR))
            emms_view = core.parameters.get_writable_timeseries_view(
                variable_openscm,
                region_openscm,
                unit,
                self.time_points,
                ParameterType.POINT_TIMESERIES,
            )
            emms_view.set(vals.values)

        for k, v in meta_values.iteritems():
            meta_view = core.parameters.get_writable_generic_view(k, ("World",))
            meta_view.set(v)

        return core

    @property
    def time_points(self):
        return self._time_index.as_openscm()

    def timeseries(self, meta=None):
        """Return a pandas dataframe in the same format as pyam.IamDataFrame.timeseries
        Parameters
        ----------
        meta: List of strings
        The list of meta columns that will be included in the rows MultiIndex. If None (default), then all metadata will be used.

        Returns
        -------
        pd.DataFrame with datetimes as columns and each row being a timeseries
        Raises
        ------
        ValueError:
        - if the metadata are not unique between timeseries

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
    def values(self):
        return self.timeseries().values

    @property
    def meta(self):
        return self._meta.copy()

    def filter(self, keep=True, inplace=False, **kwargs):
        """Return a filtered ScmDataFrame (i.e., a subset of current data)

        Parameters
        ----------
        keep: bool, default True
            keep all scenarios satisfying the filters (if True) or the inverse
        inplace: bool, default False
            if True, do operation inplace and return None
        kwargs:
            The following columns are available for filtering:
             - metadata columns: filter by category assignment
             - 'model', 'scenario', 'region', 'variable', 'unit':
               string or list of strings, where `*` can be used as a wildcard
             - 'level': the maximum "depth" of IAM variables (number of '|')
               (exluding the strings given in the 'variable' argument)
             - 'year': takes an integer, a list of integers or a range
               note that the last year of a range is not included,
               so `range(2010, 2015)` is interpreted as `[2010, ..., 2014]`
             - arguments for filtering by `datetime.datetime`
               ('month', 'hour', 'time')
             - 'regexp=True' disables pseudo-regexp syntax in `pattern_match()`
        """
        _keep_ts, _keep_cols = self._apply_filters(kwargs)
        idx = _keep_ts[:, np.newaxis] & _keep_cols
        if not idx.shape == self._data.shape:
            raise AssertionError("Index shape does not match data shape")
        idx = idx if keep else ~idx

        ret = self.copy() if not inplace else self
        d = ret._data.where(idx)
        ret._data = d.dropna(axis=1, how="all").dropna(axis=0, how="all")
        ret._meta = ret._meta[(~d.isna()).sum(axis=0) > 0]
        ret["time"] = ret._data.index.values

        if not (len(ret._data.columns) == len(ret._meta)):
            raise AssertionError("Data and meta have become unaligned")

        if len(ret._meta) == 0:
            logger.warning("Filtered ScmDataFrame is empty!")

        if not inplace:
            return ret

    def _apply_filters(self, filters):
        """Determine rows to keep in data for given set of filters

        Parameters
        ----------
        filters: dict
            dictionary of filters ({col: values}}); uses a pseudo-regexp syntax
            by default, but accepts `regexp: True` to use regexp directly
        """
        regexp = filters.pop("regexp", False)
        keep_ts = np.array([True] * len(self._data))
        keep_col = np.array([True] * len(self.meta))

        # filter by columns and list of values
        for col, values in filters.items():
            if col == "variable":
                level = filters["level"] if "level" in filters else None
                keep_col &= pattern_match(self.meta[col], values, level, regexp).values
            elif col in self.meta.columns:
                keep_col &= pattern_match(self.meta[col], values, regexp=regexp).values
            elif col == "year":
                keep_ts &= years_match(self._time_index.years(), values)

            elif col == "month":
                keep_ts &= month_match(self._time_index.months(), values)

            elif col == "day":
                if isinstance(values, str):
                    wday = True
                elif isinstance(values, list) and isinstance(values[0], str):
                    wday = True
                else:
                    wday = False

                if wday:
                    days = self._time_index.weekdays()
                else:  # ints or list of ints
                    days = self._time_index.days()

                keep_ts &= day_match(days, values)

            elif col == "hour":
                keep_ts &= hour_match(self._time_index.hours(), values)

            elif col == "time":
                keep_ts &= datetime_match(self._time_index.as_py(), values)

            elif col == "level":
                if "variable" not in filters.keys():
                    keep_col &= pattern_match(
                        self.meta["variable"], "*", values, regexp=regexp
                    ).values
                else:
                    continue

            else:
                raise ValueError("filter by `{}` not supported".format(col))
        # TODO: rename keep_col to keep_meta
        return keep_ts, keep_col

    def head(self, *args, **kwargs):
        return self.timeseries().head(*args, **kwargs)

    def tail(self, *args, **kwargs):
        return self.timeseries().tail(*args, **kwargs)

    def rename(self, mapping, inplace=False):
        """Rename and aggregate column entries using `groupby.sum()` on values.
        When renaming models or scenarios, the uniqueness of the index must be
        maintained, and the function will raise an error otherwise.

        Parameters
        ----------
        mapping: dict
            for each column where entries should be renamed, provide current
            name and target name
            {<column name>: {<current_name_1>: <target_name_1>,
                             <current_name_2>: <target_name_2>}}
        inplace: bool, default False
            if True, do operation inplace and return None
        """
        ret = copy.deepcopy(self) if not inplace else self
        for col, _mapping in mapping.items():
            if col not in self.meta.columns:
                raise ValueError("Renaming by {} not supported!".format(col))
            ret._meta[col] = ret._meta[col].replace(_mapping)
            if ret._meta.duplicated().any():
                raise ValueError("Renaming to non-unique metadata for {}!".format(col))

        if not inplace:
            return ret

    def set_meta(self, meta, name=None, index=None):
        """Add metadata columns as pd.Series, list or value (int/float/str)

        Parameters
        ----------
        meta: pd.Series, list, int, float or str
            column to be added to metadata
        name: str, optional
            meta column name (defaults to meta pd.Series.name);
            either a meta.name or the name kwarg must be defined
        """
        # check that name is valid and doesn't conflict with data columns
        if (name or (hasattr(meta, "name") and meta.name)) in [None, False]:
            raise ValueError("Must pass a name or use a named pd.Series")
        name = name or meta.name

        # check if meta has a valid index and use it for further workflow
        if hasattr(meta, "index") and hasattr(meta.index, "names"):
            index = meta.index
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

    def _get_resampled_datetimes(self, rule):
        """
        Calculates the datetimes to resample to given a rule.

        A subset of
        Parameters
        ----------
        rule

        Returns
        -------
        list of datetimes
        """
        offset = to_offset(rule)
        orig_dts = self["time"]

        # Get the bounds
        dt_0 = offset.rollback(orig_dts.iloc[0])
        dt_1 = offset.rollforward(orig_dts.iloc[-1])

        # Iterate to find all the required timesteps
        res = [dt_0]
        while res[-1] < dt_1:
            res.append(res[-1] + offset)

        return res

    def interpolate(
        self,
        target_times: List[Union[datetime.datetime, int]],
        timeseries_type: ParameterType = ParameterType.POINT_TIMESERIES,
        interpolation_type: InterpolationType = InterpolationType.LINEAR,
        extrapolation_type: ExtrapolationType = ExtrapolationType.NONE,
    ) -> ScmDataFrameBase:
        """
        Interpolate the dataframe onto a new time frame

        Uses openscm.timeseries_converter.TimeseriesConverter internally

        Parameters
        ----------
        target_times: listlike of datetimes or integers
            If integers are provided, they should be in OpenSCM time i.e. seconds since 1970-1-1 00:00:00

        Returns
        -------
        A new ScmDataFrame containing the data interpolated onto the target_times grid
        """
        source_times_openscm = [
            convert_datetime_to_openscm_time(t) for t in self["time"]
        ]
        if isinstance(target_times[0], datetime.datetime):
            # mypy confused about typing of target_times in this block, we must
            # assume datetime
            target_times_openscm = [
                convert_datetime_to_openscm_time(t)  # type: ignore
                for t in target_times
            ]
            target_times_dt = target_times
        else:
            # mypy confused about typing of target_times in this block, we must
            # assume int
            target_times_openscm = target_times  # type: ignore
            target_times_dt = [
                convert_openscm_time_to_datetime(t)  # type: ignore
                for t in target_times
            ]

        timeseries_converter = TimeseriesConverter(
            np.asarray(source_times_openscm),
            np.asarray(target_times_openscm),
            timeseries_type,
            interpolation_type,
            extrapolation_type,
        )

        # Need to keep an object index or pandas will not be able to handle a wide time range
        timeseries_index = pd.Index(target_times_dt, dtype="object", name="time")

        res = self.copy()

        res._data = res._data.apply(
            lambda col: pd.Series(
                timeseries_converter.convert_from(col.values), index=timeseries_index
            )
        )
        res["time"] = timeseries_index

        return type(self)(res)

    def resample(self, rule="AS", **kwargs):
        """Resample the time index of the timeseries data onto a custom grid

        This helper function allows for values to be easily interpolated onto Annual or monthly timesteps using rule='AS' or 'MS'
        respectively. Internally, the interpolate function performs the regridding.

        Parameters
        ----------
        rule: string
            See the pandas `user guide <http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>` for
            a list of options. Note that Business-related offsets such as `BusinessDay` are not supported.
        kwargs:
            Other arguments to pass through to `interpolate`

        Examples
        --------

            # resample a dataframe to annual values
        >>> scm_df = ScmDataFrame(pd.Series([1, 2, 10], index=(2000, 2001, 2009)), columns={
            "model": ["a_iam"],
            "scenario": ["a_scenario"],
            "region": ["World"],
            "variable": ["Primary Energy"],
            "unit": ["EJ/y"],}
        )
        >>> scm_df.timeseries().T
        model             a_iam
        scenario     a_scenario
        region            World
        variable Primary Energy
        unit               EJ/y
        year
        2000                  1
        2010                 10

        An annual timeseries can be the created by interpolating between to the start of years using the rule 'AS'.

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


        Note that the values do not fall exactly on integer values due the period between years is not exactly the same

        References
        ----------
        See the pandas documentation for
        `resample <http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.resample.html>` for more information about
        possible arguments.

        Returns
        -------

        """
        orig_dts = self["time"]
        target_dts = generate_range(
            orig_dts.iloc[0], orig_dts.iloc[-1], to_offset(rule)
        )
        return self.interpolate(list(target_dts), **kwargs)

    def process_over(self, cols, operation, **kwargs):
        """
        Process the data over the input columns.

        Parameters
        ----------
        cols : str or list of str
            Columns to perform the operation on. The timeseries will be grouped
            by all other columns in ``self.meta``.

        operation : ['median', 'mean', 'quantile']
            The operation to perform. This uses the equivalent pandas function.
            Note that quantile means the value of the data at a given point in the
            cumulative distribution of values at each point in the timeseries, for
            each timeseries once the groupby is applied. As a result, using
            ``q=0.5`` is is the same as taking the median and not the same as
            taking the mean/average.

        **kwargs
            Keyword arguments to pass to the pandas operation.

        Returns
        -------
        pd.DataFrame
            The quantiles of the timeseries, grouped by all columns in ``self.meta``
            other than ``cols``

        Raises
        ------
        ValueError
            If the operation is not one of ['median', 'mean', 'quantile'].
        """
        cols = [cols] if isinstance(cols, str) else cols
        ts = self.timeseries()
        group_cols = list(set(ts.index.names) - set(cols))
        grouper = ts.groupby(group_cols)

        if operation == "median":
            return grouper.median(**kwargs)
        elif operation == "mean":
            return grouper.mean(**kwargs)
        elif operation == "quantile":
            return grouper.quantile(**kwargs)
        else:
            raise ValueError("operation must be on of ['median', 'mean', 'quantile']")

    def relative_to_ref_period_mean(self, append_str=None, **kwargs):
        """
        Return the timeseries relative to a given reference period mean.

        The reference period mean is subtracted from all values in the input
        timeseries.

        Parameters
        ----------
        append_str : str
            String to append to the name of all the variables in the resulting
            DataFrame to indicate that they are relevant to a given reference period.
            E.g. `'rel. to 1961-1990'`. If None, this will be autofilled with the keys
            and ranges of ``kwargs``.

        **kwargs
            Arguments to pass to ``self.filter`` to determine the data to be included
            in the reference time period. See the docs of ``self.filter`` for valid
            options.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the timeseries, adjusted to the reference period mean.
        """
        ts = self.timeseries()
        ref_period_mean = self.filter(**kwargs).timeseries().mean(axis="columns")

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
        self, other: ScmDataFrameBase, inplace: bool = False, **kwargs: Any
    ) -> Optional[ScmDataFrameBase]:
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
            raise ImportError(
                "pyam is not installed. Features involving IamDataFrame are unavailable"
            )

        class LongIamDataFrame(IamDataFrame):
            """This baseclass is a custom implementation of the IamDataFrame which handles datetime data which spans longer than pd.to_datetime
            can handle
            """

            def _format_datetime_col(self):
                if isinstance(self.data["time"].iloc[0], str):

                    def convert_str_to_datetime(inp):
                        return parser.parse(inp)

                    self.data["time"] = self.data["time"].apply(convert_str_to_datetime)

                not_datetime = [
                    not isinstance(x, datetime.datetime) for x in self.data["time"]
                ]
                if any(not_datetime):
                    bad_values = self.data[not_datetime]["time"]
                    error_msg = "All time values must be convertible to datetime. The following values are not:\n{}".format(
                        bad_values
                    )
                    raise ValueError(error_msg)

        return LongIamDataFrame(self.timeseries())

    def to_csv(self, path: str, **kwargs: Any) -> None:
        """Write timeseries data to a csv file

        Parameters
        ----------
        path: string
            file path
        """
        self.to_iamdataframe().to_csv(path, **kwargs)

    def line_plot(self, x: str = "time", y: str = "value", **kwargs: Any) -> Axes:
        """Helper to generate line plots of timeseries

        See ``pyam.IamDataFrame.line_plot`` for more information

        """
        return self.to_iamdataframe().line_plot(x, y, **kwargs)

    def scatter(self, x: str, y: str, **kwargs: Any) -> Axes:
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


def df_append(
    dfs: List[Union[ScmDataFrameBase, IamDataFrame, pd.DataFrame, np.ndarray, str]],
    inplace: bool = False,
) -> Optional[ScmDataFrameBase]:
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

    # we want to put data into timeseries format and pass into format_ts instead of format_data
    data = pd.concat(
        [d.timeseries().reorder_levels(joint_meta_set) for d in joint_dfs], sort=False
    )

    data = data.reset_index()
    data[list(joint_meta_set)] = data[joint_meta_set].replace(
        to_replace=np.nan, value=na_fill_value
    )
    data = data.set_index(list(joint_meta_set))

    data = data.groupby(data.index.names).mean()

    if inplace:
        if not isinstance(dfs[0], ScmDataFrameBase):
            raise AssertionError("Can only append inplace to an ScmDataFrameBase")
        ret = dfs[0]
    else:
        ret = scm_dfs[0].copy()

    ret._data = data.reset_index(drop=True).T
    ret._data = ret._data.sort_index()
    ret["time"] = ret._data.index.values
    ret._data = ret._data.astype(float)

    ret._meta = (
        data.index.to_frame()
        .reset_index(drop=True)
        .replace(to_replace=na_fill_value, value=np.nan)
    )
    ret._sort_meta_cols()

    if inplace:
        return None
    else:
        return ret
