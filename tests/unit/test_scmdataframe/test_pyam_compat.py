import re

import pytest

from openscm.scmdataframe.pyam_compat import LongDatetimeIamDataFrame


def test_to_int_value_error(test_iam_df):
    idf = test_iam_df.data.rename({"year": "time"}, axis="columns").reset_index()
    idf.loc[:, "time"] = "2003/1/1"
    bad_val = "20311/123/1"
    idf.loc[4, "time"] = bad_val

    bad_vals = idf[idf["time"] == bad_val]["time"]
    error_msg = re.escape(
        "All time values must be convertible to datetime. The following values are "
        "not:\n{}".format(bad_vals)
    )
    with pytest.raises(ValueError, match=error_msg):
        LongDatetimeIamDataFrame(idf)
