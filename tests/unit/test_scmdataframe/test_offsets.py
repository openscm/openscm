from datetime import datetime
from types import GeneratorType

import pytest

from openscm.scmdataframe.offsets import generate_range, to_offset


@pytest.mark.parametrize(
    "offset_rule",
    ["B", "C", "BM", "BMS", "CBM", "CBMS", "BQ", "BSS", "BA", "BAS", "RE", "BH", "CBH"],
)
def test_invalid_offsets(offset_rule):
    with pytest.raises(ValueError):
        to_offset(offset_rule)


def test_annual_start():
    offset = to_offset("AS")
    dt = datetime(2001, 2, 12)

    res = offset.apply(dt)
    assert isinstance(res, datetime)
    assert res.year == 2002
    assert res.month == 1
    assert res.day == 1

    res = offset.rollback(dt)
    assert res.year == 2001
    assert res.month == 1
    assert res.day == 1

    res = offset.rollforward(dt)
    assert res.year == 2002
    assert res.month == 1
    assert res.day == 1


def test_month_start():
    offset = to_offset("MS")
    dt = datetime(2001, 2, 12)

    res = offset.apply(dt)
    assert isinstance(res, datetime)
    assert res.year == 2001
    assert res.month == 3
    assert res.day == 1

    res = offset.rollback(dt)
    assert res.year == 2001
    assert res.month == 2
    assert res.day == 1

    res = offset.rollforward(dt)
    assert res.year == 2001
    assert res.month == 3
    assert res.day == 1


def test_generate_range():
    offset = to_offset("AS")
    start = datetime(2000, 2, 12)
    end = datetime(2001, 2, 12)

    res = generate_range(start, end, offset)
    assert isinstance(res, GeneratorType)

    dts = list(res)
    assert dts[0] == datetime(2000, 1, 1)
    assert dts[1] == datetime(2001, 1, 1)
    assert dts[2] == datetime(2002, 1, 1)
