import pytest
import pandas as pd


from openscm.highlevel import ScmDataFrame


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
