import datetime
from dateutil import relativedelta


import pytest
import pandas as pd
import numpy as np


from openscm.highlevel import ScmDataFrame, convert_scmdataframe_to_parameterset
from openscm.scenarios import rcps
from openscm.constants import ONE_YEAR
from openscm.utils import convert_datetime_to_openscm_time


def test_init_df_long_timespan(test_pd_df):
    df = ScmDataFrame(test_pd_df)

    pd.testing.assert_frame_equal(
        df.timeseries().reset_index(), test_pd_df, check_like=True
    )


def test_init_df_datetime_error(test_pd_df):
    tdf = ScmDataFrame(test_pd_df).data
    tdf["time"] = 2010

    error_msg = r"^All time values must be convertible to datetime\. The following values are not:(.|\s)*$"
    with pytest.raises(ValueError, match=error_msg):
        ScmDataFrame(tdf)


def assert_parameterset(expected, time, paraset, name, region, unit, start, period_length):
    pview = paraset.get_timeseries_view(name, region, unit, start, period_length)
    relevant_idx = (np.abs(pview.get_times() - time)).argmin()
    np.testing.assert_allclose(pview.get(relevant_idx), expected)


def test_convert_scmdataframe_to_parameterset():
    tdata = rcps.filter(scenario="RCP26")

    res = convert_scmdataframe_to_parameterset(tdata)

    tstart_dt = tdata["time"].min()
    tstart = convert_datetime_to_openscm_time(tstart_dt)
    tperiod_length = ONE_YEAR.to("s").magnitude

    def get_comparison_time_for_year(yr):
        return convert_datetime_to_openscm_time(
            tstart_dt +
            relativedelta.relativedelta(years=yr-tstart_dt.year)
        )

    assert_parameterset(
        9.14781,
        get_comparison_time_for_year(2017),
        res,
        ("Emissions", "CO2", "MAGICC Fossil and Industrial"),
        "World",
        "GtC / yr",
        tstart,
        tperiod_length,
    )

    assert_parameterset(
        6.124 + 1.2981006,
        get_comparison_time_for_year(1993),
        res,
        ("Emissions", "CO2"),
        "World",
        "GtC / yr",
        tstart,
        tperiod_length,
    )

    assert_parameterset(
        7.2168971, get_comparison_time_for_year(1983), res, ("Emissions", "N2O"), "World", "MtN2ON / yr", tstart, tperiod_length
    )

    assert_parameterset(
        0.56591996, get_comparison_time_for_year(1766), res, ("Emissions", "OC"), "World", "MtOC / yr", tstart, tperiod_length
    )

    assert_parameterset(
        0.22445, get_comparison_time_for_year(2087), res, ("Emissions", "SF6"), "World", "ktSF6 / yr", tstart, tperiod_length
    )
