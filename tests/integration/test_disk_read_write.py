import pandas as pd

from openscm.scmdataframe import ScmDataFrame


def test_write_read_datafile(test_pd_df, tmp_path):
    tfile = str(tmp_path / "testfile.csv")
    tdf = ScmDataFrame(test_pd_df)

    tdf.to_csv(tfile)

    rdf = ScmDataFrame(tfile)

    pd.testing.assert_frame_equal(tdf.timeseries(), rdf.timeseries())
