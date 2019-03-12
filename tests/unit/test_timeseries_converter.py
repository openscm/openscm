from collections import namedtuple


import pytest
import numpy as np


from openscm import timeseries_converter
from openscm.errors import InsufficientDataError


def test_short_data(combo):
    timeseriesconverter = timeseries_converter.TimeseriesConverter(
        combo.source,
        combo.target,
        combo.timeseries_type,
        combo.interpolation_type,
        combo.extrapolation_type,
    )
    for a in [[], [0], [0, 1]]:
        with pytest.raises(InsufficientDataError):
            timeseriesconverter._convert(np.array(a), combo.source, combo.target)
