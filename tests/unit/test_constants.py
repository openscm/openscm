from openscm import constants
from datetime import timedelta


def test_constants():
    assert constants.MINUTE == timedelta(minutes=1).total_seconds()
    assert constants.HOUR == timedelta(hours=1).total_seconds()
    assert constants.DAY == timedelta(days=1).total_seconds()
    assert constants.WEEK == timedelta(weeks=1).total_seconds()
    assert constants.MONTH == timedelta(days=30).total_seconds()
    assert constants.YEAR == timedelta(days=365).total_seconds()
