from pathlib import Path
from unittest import mock

import pytest
from sunpy.net.fido_factory import UnifiedResponse
from sunpy.net.hek import HEKClient, HEKTable

from astropy.time import Time

from arccnet.catalogs.flares.common import FlareCatalog
from arccnet.catalogs.flares.hek import HEKFlareCatalog


@pytest.fixture
def hek_ssw_search():
    data_path = Path(__file__).parent / "data" / "hek_ssw_latest_20120101-20120103.ecsv"
    hek_ssw_latest = HEKTable.read(data_path)
    hek_ssw_latest.client = HEKClient()
    return UnifiedResponse(hek_ssw_latest)


@pytest.fixture
def hek_swpc_search():
    data_path = Path(__file__).parent / "data" / "hek_swpc_20120101-20120103.ecsv"
    hek_ssw_latest = HEKTable.read(data_path)
    hek_ssw_latest.client = HEKClient()
    return UnifiedResponse(hek_ssw_latest)


@mock.patch("arccnet.catalogs.flares.hek.Fido.search")
def test_hek_ssw_latest_short(mock_fido_search, hek_ssw_search):
    mock_fido_search.return_value = hek_ssw_search
    ssw_latest_catalog = HEKFlareCatalog(catalog="ssw_latest")
    ssw_latest_flares = ssw_latest_catalog.search("2012-01-01T00:00", "2012-01-03")

    assert ssw_latest_flares[0]["event_starttime"].isclose(Time("2012-01-01 04:45:00.000"))
    assert ssw_latest_flares[0]["event_peaktime"].isclose(Time("2012-01-01 04:50:00.000"))
    assert ssw_latest_flares[0]["event_endtime"].isclose(Time("2012-01-01 04:57:00.000"))
    assert ssw_latest_flares[0]["fl_goescls"] == "B8.1"
    assert ssw_latest_flares[0]["hgs_x"] == -35
    assert ssw_latest_flares[0]["hgs_y"] == -25

    assert ssw_latest_flares[-1]["event_starttime"].isclose(Time("2012-01-02 14:31:00.000"))
    assert ssw_latest_flares[-1]["event_peaktime"].isclose(Time("2012-01-02 15:24:00.000"))
    assert ssw_latest_flares[-1]["event_endtime"].isclose(Time("2012-01-02 16:04:00.000"))
    assert ssw_latest_flares[-1]["fl_goescls"] == "C2.4"
    assert ssw_latest_flares[-1]["hgs_x"] == 89
    assert ssw_latest_flares[-1]["hgs_y"] == 7


@mock.patch("arccnet.catalogs.flares.hek.Fido.search")
def test_hek_ssw_latest_long(mock_fido_search, hek_ssw_search):
    mock_fido_search.side_effect = [hek_ssw_search] * 4
    ssw_latest = HEKFlareCatalog(catalog="ssw_latest")
    ssw_latest_flares = ssw_latest.search("2012-01-01T00:00", "2014-06-03")

    assert ssw_latest_flares[0]["event_starttime"].isclose(Time("2012-01-01 04:45:00.000"))
    assert ssw_latest_flares[0]["event_peaktime"].isclose(Time("2012-01-01 04:50:00.000"))
    assert ssw_latest_flares[0]["event_endtime"].isclose(Time("2012-01-01 04:57:00.000"))
    assert ssw_latest_flares[0]["fl_goescls"] == "B8.1"
    assert ssw_latest_flares[0]["hgs_x"] == -35
    assert ssw_latest_flares[0]["hgs_y"] == -25

    assert ssw_latest_flares[-1]["event_starttime"].isclose(Time("2012-01-02 14:31:00.000"))
    assert ssw_latest_flares[-1]["event_peaktime"].isclose(Time("2012-01-02 15:24:00.000"))
    assert ssw_latest_flares[-1]["event_endtime"].isclose(Time("2012-01-02 16:04:00.000"))
    assert ssw_latest_flares[-1]["fl_goescls"] == "C2.4"
    assert ssw_latest_flares[-1]["hgs_x"] == 89
    assert ssw_latest_flares[-1]["hgs_y"] == 7

    assert len(ssw_latest_flares) == 52
    assert mock_fido_search.call_count == 4


def test_hek_ssw_create_catalog(hek_ssw_search):
    ssw_latest = HEKFlareCatalog(catalog="ssw_latest")
    ssw_latest_catalog = ssw_latest.create_catalog(hek_ssw_search["hek"])

    assert isinstance(ssw_latest_catalog, FlareCatalog)
    assert len(ssw_latest_catalog) == 13


@mock.patch("arccnet.catalogs.flares.hek.Fido.search")
def test_hek_swpc_short(mock_fido_search, hek_swpc_search):
    mock_fido_search.return_value = hek_swpc_search
    swpc_catalog = HEKFlareCatalog(catalog="swpc")
    swpc_flares = swpc_catalog.search("2012-01-01T00:00", "2012-01-03")

    assert swpc_flares[0]["event_starttime"].isclose(Time("2012-01-01 04:45:00.000"))
    assert swpc_flares[0]["event_peaktime"].isclose(Time("2012-01-01 04:50:00.000"))
    assert swpc_flares[0]["event_endtime"].isclose(Time("2012-01-01 04:57:00.000"))
    assert swpc_flares[0]["fl_goescls"] == "B8.1"
    assert swpc_flares[0]["hgs_x"] == 0
    assert swpc_flares[0]["hgs_y"] == 0

    assert swpc_flares[-1]["event_starttime"].isclose(Time("2012-01-02 14:31:00.000"))
    assert swpc_flares[-1]["event_peaktime"].isclose(Time("2012-01-02 15:24:00.000"))
    assert swpc_flares[-1]["event_endtime"].isclose(Time("2012-01-02 16:04:00.000"))
    assert swpc_flares[-1]["fl_goescls"] == "C2.4"
    assert swpc_flares[-1]["hgs_x"] == 0
    assert swpc_flares[-1]["hgs_y"] == 0


@mock.patch("arccnet.catalogs.flares.hek.Fido.search")
def test_hek_swpc_long(mock_fido_search, hek_swpc_search):
    mock_fido_search.side_effect = [hek_swpc_search] * 4
    swpc_catalog = HEKFlareCatalog(catalog="swpc")
    swpc_flares = swpc_catalog.search("2012-01-01T00:00", "2014-06-03")

    assert swpc_flares[0]["event_starttime"].isclose(Time("2012-01-01 04:45:00.000"))
    assert swpc_flares[0]["event_peaktime"].isclose(Time("2012-01-01 04:50:00.000"))
    assert swpc_flares[0]["event_endtime"].isclose(Time("2012-01-01 04:57:00.000"))
    assert swpc_flares[0]["fl_goescls"] == "B8.1"
    assert swpc_flares[0]["hgs_x"] == 0
    assert swpc_flares[0]["hgs_y"] == 0

    assert swpc_flares[-1]["event_starttime"].isclose(Time("2012-01-02 14:31:00.000"))
    assert swpc_flares[-1]["event_peaktime"].isclose(Time("2012-01-02 15:24:00.000"))
    assert swpc_flares[-1]["event_endtime"].isclose(Time("2012-01-02 16:04:00.000"))
    assert swpc_flares[-1]["fl_goescls"] == "C2.4"
    assert swpc_flares[-1]["hgs_x"] == 0
    assert swpc_flares[-1]["hgs_y"] == 0

    assert len(swpc_flares) == 48


def test_hek_swpc_create_catalog(hek_swpc_search):
    ssw_latest = HEKFlareCatalog(catalog="swpc")
    ssw_latest_catalog = ssw_latest.create_catalog(hek_swpc_search["hek"])

    assert isinstance(ssw_latest_catalog, FlareCatalog)
    assert len(ssw_latest_catalog) == 12
