import re

import numpy as np
import pytest

from openscm.core.time import ExtrapolationType, TimeseriesConverter
from openscm.errors import InsufficientDataError


def test_short_data(combo):
    timeseriesconverter = TimeseriesConverter(
        combo.source,
        combo.target,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    for a in [[], [0], [0, 1]]:
        with pytest.raises(InsufficientDataError):
            timeseriesconverter._convert(np.array(a), combo.source, combo.target)


def test_none_extrapolation_error(combo):
    target = np.asarray(
        [
            combo.source[0] - np.timedelta64(1, "s"),
            combo.source[0],
            combo.source[-1] + np.timedelta64(1, "s"),
        ],
        dtype=np.datetime64,
    )
    timeseriesconverter = TimeseriesConverter(
        combo.source,
        target,
        combo.timeseries_type,
        combo.interpolation_type,
        ExtrapolationType.NONE,
    )
    error_msg = re.escape(
        "Target time points are outside the source time points, use an "
        "extrapolation type other than None"
    )
    with pytest.raises(InsufficientDataError, match=error_msg):
        timeseriesconverter._convert(combo.source_values, combo.source, target)


@pytest.mark.parametrize("miss_type", ["forward", "backward"])
def test_no_overlap(combo, miss_type):
    if miss_type == "forward":
        no_overlap_target = combo.target + (
            combo.source[-1] - combo.target[0] + np.timedelta64(365, "D")
        )
    else:
        no_overlap_target = combo.target - (
            combo.target[-1] - combo.source[0] + np.timedelta64(365, "D")
        )

    with pytest.raises(InsufficientDataError):
        TimeseriesConverter(
            combo.source,
            no_overlap_target,
            combo.timeseries_type,
            combo.interpolation_type,
            ExtrapolationType.NONE,
        )

    # all ok if extrapolation is not NONE
    TimeseriesConverter(
        combo.source,
        no_overlap_target,
        combo.timeseries_type,
        combo.interpolation_type,
        ExtrapolationType.LINEAR,
    )
