import astropy.units as u
from astropy.time import Time

from arccnet.catalogs.flares.common import _generate_intervals


def test_commons():  # using 'common' somehow breaks pycharm debugging
    intervals = _generate_intervals(Time("2012-01-01T00:00"), Time("2012-01-01T00:00") + 3.6 * u.year, 3)
    assert intervals[1][-1].datetime == (intervals[0][0] + (1.2 * u.year * 3)).datetime
    assert intervals[0][1:] == intervals[1][:-1]
