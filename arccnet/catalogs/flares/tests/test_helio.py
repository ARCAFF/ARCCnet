from pathlib import Path
from unittest import mock

import pytest
from sunpy.net.fido_factory import UnifiedResponse
from sunpy.net.helio import HECClient, HECResponse

from astropy.table import Table

from arccnet.catalogs.flares.helio import HECFlareCatalog


@pytest.mark.remote_data
def test_helio_gevloc_online():
    gevloc_catalog = HECFlareCatalog(catalog="gevloc")
    gevloc_flares = gevloc_catalog.search("2012-01-01T00:00", "2012-01-01T12:00:00")
    gevloc_flares.sort("time_start")

    assert gevloc_flares[0]["time_start"].isot == "2012-01-01T04:45:00.000"
    assert gevloc_flares[0]["time_peak"].isot == "2012-01-01T04:50:00.000"
    assert gevloc_flares[0]["time_end"].isot == "2012-01-01T04:57:00.000"
    assert gevloc_flares[0]["xray_class"] == "B8.1"
    assert gevloc_flares[0]["nar"] == 11389
    assert gevloc_flares[0]["long_hg"] == -35
    assert gevloc_flares[0]["lat_hg"] == -25

    assert gevloc_flares[-1]["time_start"].isot == "2012-01-01T07:24:00.000"
    assert gevloc_flares[-1]["time_peak"].isot == "2012-01-01T07:34:00.000"
    assert gevloc_flares[-1]["time_end"].isot == "2012-01-01T07:44:00.000"
    assert gevloc_flares[-1]["xray_class"] == "C3.2"
    assert gevloc_flares[-1]["nar"] == 11389
    assert gevloc_flares[-1]["long_hg"] == -34
    assert gevloc_flares[-1]["lat_hg"] == -25


@pytest.mark.remote_data
def test_hec_goes_online():
    goes_catalog = HECFlareCatalog(catalog="goes")
    goes_flares = goes_catalog.search("2012-01-01T00:00", "2012-01-03")
    goes_flares.sort("time_start")

    assert goes_flares[0]["time_start"] == "2012-01-01T04:45:00"
    assert goes_flares[0]["time_peak"] == "2012-01-01T04:50:00"
    assert goes_flares[0]["time_end"] == "2012-01-01T04:57:00"
    assert goes_flares[0]["xray_class"] == "B8.1"

    assert goes_flares[1]["time_start"] == "2012-01-01T07:24:00"
    assert goes_flares[1]["time_peak"] == "2012-01-01T07:34:00"
    assert goes_flares[1]["time_end"] == "2012-01-01T07:44:00"
    assert goes_flares[1]["xray_class"] == "C3.2"
    assert goes_flares[1]["nar"] == 11389
    assert goes_flares[1]["long_hg"] == -34
    assert goes_flares[1]["lat_hg"] == -26
    assert len(goes_flares) == 12


@pytest.fixture
def helio_gevloc_query():
    # For testing, create a test client which doesn't test for live network connection
    class HECClientTest(HECClient):
        def __init__(self):
            pass

    data_path = Path(__file__).parent / "data" / "hec_gevloc_20120101-20120103.ecsv"
    table = Table.read(data_path)
    hec_gevloc = HECResponse(table, client=HECClientTest())
    return UnifiedResponse(hec_gevloc)


@pytest.fixture
def helio_goes_query():
    # For testing, create a test client which doesn't test for live network connection
    class HECClientTest(HECClient):
        def __init__(self):
            pass

    data_path = Path(__file__).parent / "data" / "hec_goes_20120101-20120103.ecsv"
    table = Table.read(data_path)
    hec_goes = HECResponse(table, client=HECClientTest())
    return UnifiedResponse(hec_goes)


@mock.patch("arccnet.catalogs.flares.hek.Fido.search")
def test_hec_gevloc_short(mock_fido_search, helio_gevloc_query):
    mock_fido_search.side_effect = [helio_gevloc_query]
    gevloc_catalog = HECFlareCatalog(catalog="gevloc")
    gevloc_flares = gevloc_catalog.search("2012-01-01T00:00", "2012-01-01T12:00:00")
    gevloc_flares.sort("time_start")

    assert gevloc_flares[0]["time_start"] == "2012-01-01T04:45:00"
    assert gevloc_flares[0]["time_peak"] == "2012-01-01T04:50:00"
    assert gevloc_flares[0]["time_end"] == "2012-01-01T04:57:00"
    assert gevloc_flares[0]["xray_class"] == "B8.1"
    assert gevloc_flares[0]["nar"] == 11389
    assert gevloc_flares[0]["long_hg"] == -35
    assert gevloc_flares[0]["lat_hg"] == -25

    assert gevloc_flares[-1]["time_start"] == "2012-01-02T14:31:00"
    assert gevloc_flares[-1]["time_peak"] == "2012-01-02T15:24:00"
    assert gevloc_flares[-1]["time_end"] == "2012-01-02T16:04:00"
    assert gevloc_flares[-1]["xray_class"] == "C2.4"
    assert gevloc_flares[-1]["nar"] == 11384
    assert gevloc_flares[-1]["long_hg"] == 89
    assert gevloc_flares[-1]["lat_hg"] == 7


@mock.patch("arccnet.catalogs.flares.hek.Fido.search")
def test_helio_gevloc_long(mock_fido_search, helio_gevloc_query):
    mock_fido_search.side_effect = [helio_gevloc_query] * 4
    gevloc_catalog = HECFlareCatalog(catalog="gevloc")
    gevloc_flares = gevloc_catalog.search("2012-01-01T00:00", "2014-06-03")
    gevloc_flares.sort("time_start")

    assert gevloc_flares[0]["time_start"] == "2012-01-01T04:45:00"
    assert gevloc_flares[0]["time_peak"] == "2012-01-01T04:50:00"
    assert gevloc_flares[0]["time_end"] == "2012-01-01T04:57:00"
    assert gevloc_flares[0]["xray_class"] == "B8.1"
    assert gevloc_flares[0]["nar"] == 11389
    assert gevloc_flares[0]["long_hg"] == -35
    assert gevloc_flares[0]["lat_hg"] == -25

    assert gevloc_flares[-1]["time_start"] == "2012-01-02T14:31:00"
    assert gevloc_flares[-1]["time_peak"] == "2012-01-02T15:24:00"
    assert gevloc_flares[-1]["time_end"] == "2012-01-02T16:04:00"
    assert gevloc_flares[-1]["xray_class"] == "C2.4"
    assert gevloc_flares[-1]["nar"] == 11384
    assert gevloc_flares[-1]["long_hg"] == 89
    assert gevloc_flares[-1]["lat_hg"] == 7

    assert len(gevloc_flares) == 48
    assert mock_fido_search.call_count == 4


def test_create_catalog_gevloc(helio_gevloc_query):
    hec_gevloc = HECFlareCatalog(catalog="gevloc")
    catalog = hec_gevloc.create_catalog(helio_gevloc_query["hec"])
    assert len(catalog) == 12
    assert set(catalog.columns).issuperset(
        (
            "start_time",
            "peak_time",
            "end_time",
            "noaa_number",
            "hgs_latitude",
            "hgs_longitude",
            "long_carr",
            "goes_class",
        )
    )


@mock.patch("arccnet.catalogs.flares.hek.Fido.search")
def test_hec_goes_short(mock_fido_search, helio_goes_query):
    mock_fido_search.side_effect = [helio_goes_query]
    goes_catalog = HECFlareCatalog(catalog="goes")
    goes_flares = goes_catalog.search("2012-01-01T00:00", "2012-01-03")
    goes_flares.sort("time_start")

    assert goes_flares[0]["time_start"] == "2012-01-01T04:45:00"
    assert goes_flares[0]["time_peak"] == "2012-01-01T04:50:00"
    assert goes_flares[0]["time_end"] == "2012-01-01T04:57:00"
    assert goes_flares[0]["xray_class"] == "B8.1"

    assert goes_flares[1]["time_start"] == "2012-01-01T07:24:00"
    assert goes_flares[1]["time_peak"] == "2012-01-01T07:34:00"
    assert goes_flares[1]["time_end"] == "2012-01-01T07:44:00"
    assert goes_flares[1]["xray_class"] == "C3.2"
    assert goes_flares[1]["nar"] == 11389
    assert goes_flares[1]["long_hg"] == -34
    assert goes_flares[1]["lat_hg"] == -26
    assert len(goes_flares) == 12


@mock.patch("arccnet.catalogs.flares.hek.Fido.search")
def test_hec_goes_long(mock_fido_search, helio_goes_query):
    mock_fido_search.side_effect = [helio_goes_query] * 4
    goes_catalog = HECFlareCatalog(catalog="goes")
    goes_flares = goes_catalog.search("2012-01-01T00:00", "2014-06-03")
    goes_flares.sort("time_start")

    assert goes_flares[0]["time_start"] == "2012-01-01T04:45:00"
    assert goes_flares[0]["time_peak"] == "2012-01-01T04:50:00"
    assert goes_flares[0]["time_end"] == "2012-01-01T04:57:00"
    assert goes_flares[0]["xray_class"] == "B8.1"

    assert goes_flares[4]["time_start"] == "2012-01-01T07:24:00"
    assert goes_flares[4]["time_peak"] == "2012-01-01T07:34:00"
    assert goes_flares[4]["time_end"] == "2012-01-01T07:44:00"
    assert goes_flares[4]["xray_class"] == "C3.2"
    assert goes_flares[4]["nar"] == 11389
    assert goes_flares[4]["long_hg"] == -34
    assert goes_flares[4]["lat_hg"] == -26
    assert len(goes_flares) == 48


def test_create_catalog_goes(helio_goes_query):
    hec_goes = HECFlareCatalog(catalog="goes")
    catalog = hec_goes.create_catalog(helio_goes_query["hec"])
    assert len(catalog) == 12
    assert set(catalog.columns).issuperset(
        (
            "start_time",
            "peak_time",
            "end_time",
            "noaa_number",
            "hgs_latitude",
            "hgs_longitude",
            "long_carr",
            "goes_class",
        )
    )
