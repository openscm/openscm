import warnings

import numpy as np
import pytest
from conftest import assert_pint_equal

from openscm.core.units import _unit_registry


def test_pint_array_comparison():
    a = np.array([0, 2]) * _unit_registry("GtC")
    b = np.array([0, 2]) * _unit_registry("MtC")

    # no error but does raise warning about stripping units
    with warnings.catch_warnings(record=True):
        np.testing.assert_allclose(a, b)

    # actually gives an error as we want
    with pytest.raises(AssertionError):
        assert_pint_equal(a, b)
