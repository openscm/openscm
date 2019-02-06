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


def assert_parameterset(expected, paraset, name, region, unit, start, period_length):
    pview = paraset.get_timeseries_view(name, region, unit, start, period_length)
    # import pdb
    # pdb.set_trace()
    assert np.testing.all_close(pview.get(), expected)


def test_convert_scmdataframe_to_parameterset():
    tdata = rcps.filter(scenario="RCP26")
    tstart = convert_datetime_to_openscm_time(tdata["time"].min())
    tperiod_length = ONE_YEAR.to("s").magnitude
    res = convert_scmdataframe_to_parameterset(tdata)

    assert_parameterset(
        3,
        res,
        ("Emissions", "CO2", "MAGICC Fossil and Industrial"),
        "World",
        "GtC / yr",
        tstart,
        tperiod_length,
    )

    assert_parameterset(
        3, res, ("Emissions", "N2O"), "World", "MtN2O / yr", tstart, tperiod_length
    )

    assert_parameterset(
        3, res, ("Emissions", "OC"), "World", "MtOC / yr", tstart, tperiod_length
    )

    assert_parameterset(
        3, res, ("Emissions", "SF6"), "World", "ktSF6 / yr", tstart, tperiod_length
    )
