from openscm.highlevel import OpenSCMDataFrame


def test_write_read_datafile(test_pd_df, tmp_path):
    tfile = str(tmp_path / "testfile.csv")
    tdf = OpenSCMDataFrame(test_pd_df)

    tdf.to_csv(tfile)

    odf = OpenSCMDataFrame(tfile)

