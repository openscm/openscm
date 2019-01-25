import pandas as pd


from openscm.highlevel import OpenSCMDataFrame


def test_init_df_long_timespan(test_pd_df):
    df = OpenSCMDataFrame(test_pd_df)

    pd.testing.assert_frame_equal(df.timeseries().reset_index(), test_pd_df)
