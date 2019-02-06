import copy
import datetime
import re

import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from pyam.core import require_variable, categorize, filter_by_meta, validate, META_IDX, IamDataFrame

from openscm.highlevel import ScmDataFrame


def test_init_df_long_timespan(test_pd_longtime_df):
    df = ScmDataFrame(test_pd_longtime_df)

    pd.testing.assert_frame_equal(
        df.timeseries().reset_index(), test_pd_longtime_df, check_like=True
    )


def test_init_ts(test_ts, test_pd_df):
    df = ScmDataFrame(
        test_ts,
        columns={
            "index": [2005, 2010],
            "model": ["a_iam"],
            "climate_model": ["a_model"],
            "scenario": ["a_scenario", "a_scenario", "a_scenario2"],
            "region": ["World"],
            "variable": ["Primary Energy", "Primary Energy|Coal", "Primary Energy"],
            "unit": ["EJ/y"],
        },
    )

    pd.testing.assert_frame_equal(
        df.timeseries().reset_index(), test_pd_df, check_like=True
    )

    b = ScmDataFrame(test_pd_df)

    pd.testing.assert_frame_equal(df.meta, b.meta, check_like=True)
    pd.testing.assert_frame_equal(df._data, b._data)


def test_init_ts_with_index(test_pd_df):
    df = ScmDataFrame(test_pd_df)
    pd.testing.assert_frame_equal(
        df.timeseries().reset_index(), test_pd_df, check_like=True
    )


def test_init_df_with_float_cols(test_pd_df):
    _test_scm_df = test_pd_df.rename(columns={2005: 2005.0, 2010: 2010.0})
    obs = ScmDataFrame(_test_scm_df).timeseries().reset_index()
    pd.testing.assert_series_equal(obs[2005], test_pd_df[2005])


def test_init_df_from_timeseries(test_scm_df):
    df = ScmDataFrame(test_scm_df.timeseries())
    pd.testing.assert_frame_equal(df.timeseries(), test_scm_df.timeseries())


def test_init_df_with_extra_col(test_pd_df):
    tdf = test_pd_df.copy()

    extra_col = "test value"
    extra_value = "scm_model"
    tdf[extra_col] = extra_value

    df = ScmDataFrame(tdf)

    assert extra_col in df.meta
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), tdf, check_like=True)


def test_init_datetime(test_pd_df):
    tdf = test_pd_df.copy()
    tmin = datetime.datetime(2005, 6, 17)
    tmax = datetime.datetime(2010, 6, 17)
    tdf = tdf.rename({2005: tmin, 2010: tmax}, axis="columns")

    df = ScmDataFrame(tdf)

    assert df["time"].max() == tmax
    assert df["time"].min() == tmin


@pytest.mark.xfail(
    reason=(
        "pandas datetime is limited to the time period of ~1677-2262, see "
        "https://stackoverflow.com/a/37226672"
    )
)
def test_init_datetime_long_timespan(test_pd_df):
    tdf = test_pd_df.copy()
    tmin = datetime.datetime(2005, 6, 17)
    tmax = datetime.datetime(3005, 6, 17)
    tdf = tdf.rename({2005: tmin, 2010: tmax}, axis="columns")

    df = ScmDataFrame(tdf)

    assert df["time"].max() == tmax
    assert df["time"].min() == tmin


def test_init_datetime_subclass_long_timespan(test_pd_df):
    class TempSubClass(ScmDataFrame):
        def _format_datetime_col(self):
            # the subclass does not try to coerce the datetimes to pandas
            # datetimes, instead simply leaving the time column as object type,
            # so we don't run into the problem of pandas limited time period as
            # discussed in https://stackoverflow.com/a/37226672
            pass

    tdf = test_pd_df.copy()
    tmin = datetime.datetime(2005, 6, 17)
    tmax = datetime.datetime(3005, 6, 17)
    tdf = tdf.rename({2005: tmin, 2010: tmax}, axis="columns")

    df = TempSubClass(tdf)

    assert df["time"].max() == tmax
    assert df["time"].min() == tmin


def test_init_iam(test_iam_df, test_pd_df):
    a = ScmDataFrame(test_iam_df.data)
    b = ScmDataFrame(test_pd_df)

    pd.testing.assert_frame_equal(a.meta, b.meta)
    pd.testing.assert_frame_equal(a._data, b._data)


def test_as_iam(test_iam_df, test_pd_df):
    df = ScmDataFrame(test_pd_df).as_iam()

    assert isinstance(df, IamDataFrame)

    pd.testing.assert_frame_equal(test_iam_df.meta, df.meta)
    pd.testing.assert_frame_equal(test_iam_df.data, df.data)


def test_get_item(test_scm_df):
    assert test_scm_df["model"].unique() == ["a_iam"]


def test_variable_depth_0(test_scm_df):
    obs = list(test_scm_df.filter(level=0)["variable"].unique())
    exp = ["Primary Energy"]
    assert obs == exp


def test_variable_depth_0_keep_false(test_scm_df):
    obs = list(test_scm_df.filter(level=0, keep=False)["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_0_minus(test_scm_df):
    obs = list(test_scm_df.filter(level="0-")["variable"].unique())
    exp = ["Primary Energy"]
    assert obs == exp


def test_variable_depth_0_plus(test_scm_df):
    obs = list(test_scm_df.filter(level="0+")["variable"].unique())
    exp = ["Primary Energy", "Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1(test_scm_df):
    obs = list(test_scm_df.filter(level=1)["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1_minus(test_scm_df):
    obs = list(test_scm_df.filter(level="1-")["variable"].unique())
    exp = ["Primary Energy", "Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_1_plus(test_scm_df):
    obs = list(test_scm_df.filter(level="1+")["variable"].unique())
    exp = ["Primary Energy|Coal"]
    assert obs == exp


def test_variable_depth_raises(test_scm_df):
    pytest.raises(ValueError, test_scm_df.filter, level="1/")


def test_filter_error(test_scm_df):
    pytest.raises(ValueError, test_scm_df.filter, foo="foo")


def test_filter_year(test_scm_df):
    obs = test_scm_df.filter(year=2005)
    if test_scm_df.is_annual_timeseries:
        npt.assert_equal(obs["year"].unique(), 2005)
    else:
        expected = np.array(
            pd.to_datetime("2005-06-17T00:00:00.0"), dtype=np.datetime64
        )
        unique_time = obs["time"].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("test_month", [6, "June", "Jun", "jun", ["Jun", "jun"]])
def test_filter_month(test_scm_df, test_month):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("filter by `month` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_scm_df.filter(month=test_month)
    else:
        obs = test_scm_df.filter(month=test_month)
        expected = np.array(
            pd.to_datetime("2005-06-17T00:00:00.0"), dtype=np.datetime64
        )
        unique_time = obs["time"].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("test_month", [6, "Jun", "jun", ["Jun", "jun"]])
def test_filter_year_month(test_scm_df, test_month):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("filter by `month` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_scm_df.filter(year=2005, month=test_month)
    else:
        obs = test_scm_df.filter(year=2005, month=test_month)
        expected = np.array(
            pd.to_datetime("2005-06-17T00:00:00.0"), dtype=np.datetime64
        )
        unique_time = obs["time"].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("test_day", [17, "Fri", "Friday", "friday", ["Fri", "fri"]])
def test_filter_day(test_scm_df, test_day):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("filter by `day` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_scm_df.filter(day=test_day)
    else:
        obs = test_scm_df.filter(day=test_day)
        expected = np.array(
            pd.to_datetime("2005-06-17T00:00:00.0"), dtype=np.datetime64
        )
        unique_time = obs["time"].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("test_hour", [0, 12, [12, 13]])
def test_filter_hour(test_scm_df, test_hour):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("filter by `hour` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_scm_df.filter(hour=test_hour)
    else:
        obs = test_scm_df.filter(hour=test_hour)
        test_hour = [test_hour] if isinstance(test_hour, int) else test_hour
        expected_rows = test_scm_df.data["time"].apply(lambda x: x.hour).isin(test_hour)
        expected = test_scm_df.data["time"].loc[expected_rows].unique()

        unique_time = obs["time"].unique()
        npt.assert_array_equal(unique_time, expected)


def test_filter_time_exact_match(test_scm_df):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("`year` can only be filtered with ints or lists of ints")
        with pytest.raises(TypeError, match=error_msg):
            test_scm_df.filter(year=datetime.datetime(2005, 6, 17))
    else:
        obs = test_scm_df.filter(time=datetime.datetime(2005, 6, 17))
        expected = np.array(
            pd.to_datetime("2005-06-17T00:00:00.0"), dtype=np.datetime64
        )
        unique_time = obs["time"].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


def test_filter_time_range(test_scm_df):
    error_msg = r".*datetime.datetime.*"
    with pytest.raises(TypeError, match=error_msg):
        test_scm_df.filter(
            year=range(datetime.datetime(2000, 6, 17), datetime.datetime(2009, 6, 17))
        )


def test_filter_time_range_year(test_scm_df):
    obs = test_scm_df.filter(year=range(2000, 2008))

    if test_scm_df.is_annual_timeseries:
        unique_time = obs["year"].unique()
        expected = np.array([2005])
    else:
        unique_time = obs["time"].unique()
        expected = np.array(
            pd.to_datetime("2005-06-17T00:00:00.0"), dtype=np.datetime64
        )

    assert len(unique_time) == 1
    assert unique_time[0] == expected


@pytest.mark.parametrize("month_range", [range(1, 7), "Mar-Jun"])
def test_filter_time_range_month(test_scm_df, month_range):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("filter by `month` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_scm_df.filter(month=month_range)
    else:
        obs = test_scm_df.filter(month=month_range)
        expected = np.array(
            pd.to_datetime("2005-06-17T00:00:00.0"), dtype=np.datetime64
        )

        unique_time = obs["time"].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("month_range", [["Mar-Jun", "Nov-Feb"]])
def test_filter_time_range_round_the_clock_error(test_scm_df, month_range):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("filter by `month` not supported")
        with pytest.raises(ValueError, match=error_msg):
            test_scm_df.filter(month=month_range)
    else:
        error_msg = re.escape(
            "string ranges must lead to increasing integer ranges, "
            "Nov-Feb becomes [11, 2]"
        )
        with pytest.raises(ValueError, match=error_msg):
            test_scm_df.filter(month=month_range)


@pytest.mark.parametrize("day_range", [range(14, 20), "Thu-Sat"])
def test_filter_time_range_day(test_scm_df, day_range):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("filter by `day` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_scm_df.filter(day=day_range)
    else:
        obs = test_scm_df.filter(day=day_range)
        expected = np.array(
            pd.to_datetime("2005-06-17T00:00:00.0"), dtype=np.datetime64
        )
        unique_time = obs["time"].unique()
        assert len(unique_time) == 1
        assert unique_time[0] == expected


@pytest.mark.parametrize("hour_range", [range(10, 14)])
def test_filter_time_range_hour(test_scm_df, hour_range):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("filter by `hour` not supported")
        with pytest.raises(ValueError, match=error_msg):
            obs = test_scm_df.filter(hour=hour_range)
    else:
        obs = test_scm_df.filter(hour=hour_range)

        expected_rows = (
            test_scm_df.data["time"].apply(lambda x: x.hour).isin(hour_range)
        )
        expected = test_scm_df.data["time"].loc[expected_rows].unique()

        unique_time = obs["time"].unique()
        npt.assert_array_equal(unique_time, expected)


def test_filter_time_no_match(test_scm_df):
    if test_scm_df.is_annual_timeseries:
        error_msg = re.escape("`year` can only be filtered with ints or lists of ints")
        with pytest.raises(TypeError, match=error_msg):
            test_scm_df.filter(year=datetime.datetime(2004, 6, 18))
    else:
        obs = test_scm_df.filter(time=datetime.datetime(2004, 6, 18))
        assert obs.data.empty


def test_filter_time_not_datetime_error(test_scm_df):
    if test_scm_df.is_annual_timeseries:
        with pytest.raises(ValueError, match="filter by `time` not supported"):
            test_scm_df.filter(time=datetime.datetime(2004, 6, 18))
    else:
        error_msg = re.escape(
            "`time` can only be filtered with datetimes or lists of datetimes"
        )
        with pytest.raises(TypeError, match=error_msg):
            test_scm_df.filter(time=2005)


def test_filter_time_not_datetime_range_error(test_scm_df):
    if test_scm_df.is_annual_timeseries:
        with pytest.raises(ValueError, match="filter by `time` not supported"):
            test_scm_df.filter(time=range(2000, 2008))
    else:
        error_msg = re.escape(
            "`time` can only be filtered with datetimes or lists of datetimes"
        )
        with pytest.raises(TypeError, match=error_msg):
            test_scm_df.filter(time=range(2000, 2008))


def test_filter_as_kwarg(test_scm_df):
    obs = list(test_scm_df.filter(variable="Primary Energy|Coal")["scenario"].unique())
    assert obs == ["a_scenario"]


def test_filter_keep_false(test_scm_df):
    df = test_scm_df.filter(variable="Primary Energy|Coal", year=2005, keep=False)
    obs = df.filter(scenario="a_scenario").timeseries().values.ravel()
    npt.assert_array_equal(obs, [1, 6, np.nan, 3])


def test_filter_by_regexp(test_scm_df):
    obs = test_scm_df.filter(scenario="a_scenari.$", regexp=True)
    assert obs["scenario"].unique() == "a_scenario"


def test_timeseries(test_scm_df):
    dct = {
        "model": ["a_model"] * 2,
        "scenario": ["a_scenario"] * 2,
        "years": [2005, 2010],
        "value": [1, 6],
    }
    exp = pd.DataFrame(dct).pivot_table(
        index=["model", "scenario"], columns=["years"], values="value"
    )
    obs = test_scm_df.filter(
        variable="Primary Energy", scenario="a_scenario"
    ).timeseries()
    npt.assert_array_equal(obs, exp)


@pytest.mark.skip
def test_require_variable(test_scm_df):
    obs = test_scm_df.require_variable(
        variable="Primary Energy|Coal", exclude_on_fail=True
    )
    assert len(obs) == 1
    assert obs.loc[0, "scenario"] == "a_scenario2"

    assert list(test_scm_df["exclude"]) == [False, True]


@pytest.mark.skip
def test_require_variable_top_level(test_scm_df):
    obs = require_variable(
        test_scm_df, variable="Primary Energy|Coal", exclude_on_fail=True
    )
    assert len(obs) == 1
    assert obs.loc[0, "scenario"] == "a_scenario2"

    assert list(test_scm_df["exclude"]) == [False, True]


@pytest.mark.skip
def test_validate_all_pass(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy": {"up": 10}}, exclude_on_fail=True)
    assert obs is None
    assert len(test_scm_df.data) == 6  # data unchanged

    assert list(test_scm_df["exclude"]) == [False, False]  # none excluded


@pytest.mark.skip
def test_validate_nonexisting(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy|Coal": {"up": 2}}, exclude_on_fail=True)
    assert len(obs) == 1
    assert obs["scenario"].values[0] == "a_scenario"

    assert list(test_scm_df["exclude"]) == [True, False]  # scenario with failed
    # validation excluded, scenario with non-defined value passes validation


@pytest.mark.skip
def test_validate_up(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy": {"up": 6.5}}, exclude_on_fail=False)
    assert len(obs) == 1
    assert obs["year"].values[0] == 2010

    assert list(test_scm_df["exclude"]) == [False, False]  # assert none excluded


@pytest.mark.skip
def test_validate_lo(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy": {"up": 8, "lo": 2.0}})
    assert len(obs) == 1
    assert obs["year"].values[0] == 2005
    assert list(obs["scenario"].values) == ["a_scenario"]


@pytest.mark.skip
def test_validate_both(test_scm_df):
    obs = test_scm_df.validate({"Primary Energy": {"up": 6.5, "lo": 2.0}})
    assert len(obs) == 2
    assert list(obs["year"].values) == [2005, 2010]
    assert list(obs["scenario"].values) == ["a_scenario", "a_scenario2"]


@pytest.mark.skip
def test_validate_year(test_scm_df):
    obs = test_scm_df.validate(
        {"Primary Energy": {"up": 5.0, "year": 2005}}, exclude_on_fail=False
    )
    assert obs is None

    obs = test_scm_df.validate(
        {"Primary Energy": {"up": 5.0, "year": 2010}}, exclude_on_fail=False
    )
    assert len(obs) == 2


@pytest.mark.skip
def test_validate_exclude(test_scm_df):
    test_scm_df.validate({"Primary Energy": {"up": 6.0}}, exclude_on_fail=True)
    assert list(test_scm_df["exclude"]) == [False, True]


@pytest.mark.skip
def test_validate_top_level(test_scm_df):
    obs = validate(
        test_scm_df,
        criteria={"Primary Energy": {"up": 6.0}},
        exclude_on_fail=True,
        variable="Primary Energy",
    )
    assert len(obs) == 1
    assert obs["year"].values[0] == 2010
    assert list(test_scm_df["exclude"]) == [False, True]


@pytest.mark.skip
def test_category_none(test_scm_df):
    test_scm_df.categorize("category", "Testing", {"Primary Energy": {"up": 0.8}})
    assert "category" not in test_scm_df.meta.columns


@pytest.mark.skip
def test_category_pass(test_scm_df):
    dct = {
        "model": ["a_model", "a_model"],
        "scenario": ["a_scenario", "a_scenario2"],
        "category": ["foo", None],
    }
    exp = pd.DataFrame(dct).set_index(["model", "scenario"])["category"]

    test_scm_df.categorize(
        "category", "foo", {"Primary Energy": {"up": 6, "year": 2010}}
    )
    obs = test_scm_df["category"]
    pd.testing.assert_series_equal(obs, exp)


@pytest.mark.skip
def test_category_top_level(test_scm_df):
    dct = {
        "model": ["a_model", "a_model"],
        "scenario": ["a_scenario", "a_scenario2"],
        "category": ["Testing", None],
    }
    exp = pd.DataFrame(dct).set_index(["model", "scenario"])["category"]

    categorize(
        test_scm_df,
        "category",
        "Testing",
        criteria={"Primary Energy": {"up": 6, "year": 2010}},
        variable="Primary Energy",
    )
    obs = test_scm_df["category"]
    pd.testing.assert_series_equal(obs, exp)


def test_append(test_scm_df):
    test_scm_df.set_meta([5, 6, 7], name="col1")
    other = test_scm_df.filter(scenario="a_scenario2").rename(
        {"variable": {"Primary Energy": "Primary Energy clone"}}
    )

    other.set_meta(2, name="col1")
    other.set_meta("b", name="col2")

    df = test_scm_df.append(other)

    # check that the new meta.index is updated, but not the original one
    assert "col1" in test_scm_df.meta

    # assert that merging of meta works as expected
    npt.assert_array_equal(df.meta["col1"].values, [5, 6, 7, 2])
    pd.testing.assert_series_equal(
        df.meta["col2"], pd.Series([np.nan, np.nan, np.nan, "b"]), check_names=False
    )

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2], ts.iloc[3])


def test_append_duplicates(test_scm_df):
    other = copy.deepcopy(test_scm_df)
    pytest.raises(ValueError, test_scm_df.append, other=other)


@pytest.mark.skip
def test_interpolate(test_scm_df):
    test_scm_df.interpolate(2007)
    dct = {
        "model": ["a_model"] * 3,
        "scenario": ["a_scenario"] * 3,
        "years": [2005, 2007, 2010],
        "value": [1, 3, 6],
    }
    exp = pd.DataFrame(dct).pivot_table(
        index=["model", "scenario"], columns=["years"], values="value"
    )
    variable = {"variable": "Primary Energy"}
    obs = test_scm_df.filter(**variable).timeseries()
    npt.assert_array_equal(obs, exp)

    # redo the inpolation and check that no duplicates are added
    test_scm_df.interpolate(2007)
    assert not test_scm_df.filter(**variable).data.duplicated().any()


def test_set_meta_no_name(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_iam"], ["World"]],
        labels=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )
    s = pd.Series(data=[0.3], index=idx)
    pytest.raises(ValueError, test_scm_df.set_meta, s)


def test_set_meta_as_named_series(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_iam"], ["World"]],
        labels=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )

    s = pd.Series(data=[0.3], index=idx)
    s.name = "meta_values"
    test_scm_df.set_meta(s)

    exp = pd.Series(data=[0.3, 0.3, np.nan], index=test_scm_df.meta.index)
    exp.name = "meta_values"

    obs = test_scm_df["meta_values"]
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_unnamed_series(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_scenario"], ["a_iam"], ["World"]],
        labels=[[0], [0], [0]],
        names=["scenario", "model", "region"],
    )

    s = pd.Series(data=[0.3], index=idx)
    test_scm_df.set_meta(s, name="meta_values")

    exp = pd.Series(data=[0.3, 0.3, np.nan], index=test_scm_df.meta.index)
    exp.name = "meta_values"

    obs = test_scm_df["meta_values"]
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_non_unique_index_fail(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_iam"], ["a_scenario"], ["a", "b"]],
        labels=[[0, 0], [0, 0], [0, 1]],
        names=["model", "scenario", "region"],
    )
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, test_scm_df.set_meta, s)


def test_set_meta_non_existing_index_fail(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_iam", "fail_model"], ["a_scenario", "fail_scenario"]],
        labels=[[0, 1], [0, 1]],
        names=["model", "scenario"],
    )
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, test_scm_df.set_meta, s)


def test_set_meta_by_df(test_scm_df):
    df = pd.DataFrame(
        [["a_iam", "a_scenario", "World", 1]],
        columns=["model", "scenario", "region", "col"],
    )

    test_scm_df.set_meta(meta=0.3, name="meta_values", index=df)

    exp = pd.Series(data=[0.3, 0.3, np.nan], index=test_scm_df.meta.index)
    exp.name = "meta_values"

    obs = test_scm_df["meta_values"]
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_series(test_scm_df):
    # TODO: This is a bit yucky. You can pass in a series which isnt the same length as the meta table without warning
    s = pd.Series([0.3, 0.4])
    test_scm_df.set_meta(s, "meta_series")

    exp = pd.Series(data=[0.3, 0.4, np.nan], index=test_scm_df.meta.index)
    exp.name = "meta_series"

    obs = test_scm_df["meta_series"]
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_int(test_scm_df):
    test_scm_df.set_meta(3.2, "meta_int")

    exp = pd.Series(data=[3.2, 3.2, 3.2], index=test_scm_df.meta.index, name="meta_int")

    obs = test_scm_df["meta_int"]
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str(test_scm_df):
    test_scm_df.set_meta("testing", name="meta_str")

    exp = pd.Series(
        data=["testing", "testing", "testing"],
        index=test_scm_df.meta.index,
        name="meta_str",
    )

    obs = test_scm_df["meta_str"]
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str_list(test_scm_df):
    test_scm_df.set_meta(["testing", "testing2", "testing2"], name="category")
    obs = test_scm_df.filter(category="testing")
    assert obs["scenario"].unique() == "a_scenario"


def test_set_meta_as_str_by_index(test_scm_df):
    idx = pd.MultiIndex(
        levels=[["a_iam"], ["a_scenario"]],
        labels=[[0], [0]],
        names=["model", "scenario"],
    )

    test_scm_df.set_meta("foo", "meta_str", idx)

    obs = pd.Series(test_scm_df["meta_str"].values)
    pd.testing.assert_series_equal(obs, pd.Series(["foo", "foo", None]))


def test_filter_by_bool(test_scm_df):
    test_scm_df.set_meta([True, False, False], name="exclude")
    obs = test_scm_df.filter(exclude=True)
    assert obs["scenario"].unique() == "a_scenario"


def test_filter_by_int(test_scm_df):
    test_scm_df.set_meta([1, 2, 3], name="test")
    obs = test_scm_df.filter(test=[1])
    assert obs["scenario"].unique() == "a_scenario"


def test_rename_variable(test_scm_df):
    mapping = {
        "variable": {
            "Primary Energy": "Primary Energy|Total",
            "Primary Energy|Coal": "Primary Energy|Fossil",
        }
    }

    obs = test_scm_df.rename(mapping)

    exp = pd.Series(
        ["Primary Energy|Total", "Primary Energy|Fossil", "Primary Energy|Total"]
    )
    pd.testing.assert_series_equal(
        obs["variable"], exp, check_index_type=False, check_names=False
    )


def test_rename_index_fail(test_scm_df):
    mapping = {"scenario": {"a_scenario": "a_scenario2"}}
    pytest.raises(ValueError, test_scm_df.rename, mapping)


@pytest.mark.skip
def test_convert_unit():
    df = ScmDataFrame(
        pd.DataFrame(
            [
                ["model", "scen", "SST", "test_1", "A", 1, 5],
                ["model", "scen", "SDN", "test_2", "unit", 2, 6],
                ["model", "scen", "SST", "test_3", "C", 3, 7],
            ],
            columns=["model", "scenario", "region", "variable", "unit", 2005, 2010],
        )
    )

    unit_conv = {"A": ["B", 5], "C": ["D", 3]}

    obs = df.convert_unit(unit_conv).data.reset_index(drop=True)

    exp = ScmDataFrame(
        pd.DataFrame(
            [
                ["model", "scen", "SST", "test_1", "B", 5, 25],
                ["model", "scen", "SDN", "test_2", "unit", 2, 6],
                ["model", "scen", "SST", "test_3", "D", 9, 21],
            ],
            columns=["model", "scenario", "region", "variable", "unit", 2005, 2010],
        )
    ).data.reset_index(drop=True)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


@pytest.mark.skip
def test_pd_filter_by_meta(test_scm_df):
    data = df_filter_by_meta_matching_idx.set_index(["model", "region"])

    test_scm_df.set_meta([True, False], "boolean")
    test_scm_df.set_meta(0, "integer")

    obs = filter_by_meta(data, test_scm_df, join_meta=True, boolean=True, integer=None)
    obs = obs.reindex(columns=["scenario", "col", "boolean", "integer"])

    exp = data.iloc[0:2].copy()
    exp["boolean"] = True
    exp["integer"] = 0

    pd.testing.assert_frame_equal(obs, exp)


@pytest.mark.skip
def test_pd_filter_by_meta_no_index(test_scm_df):
    data = df_filter_by_meta_matching_idx

    test_scm_df.set_meta([True, False], "boolean")
    test_scm_df.set_meta(0, "int")

    obs = filter_by_meta(data, test_scm_df, join_meta=True, boolean=True, int=None)
    obs = obs.reindex(columns=META_IDX + ["region", "col", "boolean", "int"])

    exp = data.iloc[0:2].copy()
    exp["boolean"] = True
    exp["int"] = 0

    pd.testing.assert_frame_equal(obs, exp)


@pytest.mark.skip
def test_pd_filter_by_meta_nonmatching_index(test_scm_df):
    data = df_filter_by_meta_nonmatching_idx
    test_scm_df.set_meta(["a", "b"], "string")

    obs = filter_by_meta(data, test_scm_df, join_meta=True, string="b")
    obs = obs.reindex(columns=["scenario", 2010, 2020, "string"])

    exp = data.iloc[2:3].copy()
    exp["string"] = "b"

    pd.testing.assert_frame_equal(obs, exp)


@pytest.mark.skip
def test_pd_join_by_meta_nonmatching_index(test_scm_df):
    data = df_filter_by_meta_nonmatching_idx
    test_scm_df.set_meta(["a", "b"], "string")

    obs = filter_by_meta(data, test_scm_df, join_meta=True, string=None)
    obs = obs.reindex(columns=["scenario", 2010, 2020, "string"])

    exp = data.copy()
    exp["string"] = [np.nan, np.nan, "b"]

    pd.testing.assert_frame_equal(obs.sort_index(level=1), exp)
