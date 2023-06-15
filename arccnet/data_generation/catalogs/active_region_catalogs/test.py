import datetime

from sunpy.net import Fido
from sunpy.net import attrs as a

result = Fido.search(
    a.Time(datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 12)),
    a.Instrument.soon,
)

table = Fido.fetch(
    result,
    # path=dv.NOAA_SRS_TEXT_DIR,
    progress=True,
    overwrite=False,
)
