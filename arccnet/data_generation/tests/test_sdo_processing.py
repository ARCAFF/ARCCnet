from pathlib import Path

import pytest
import sunpy.data.sample
import sunpy.map

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

from arccnet import config
from arccnet.data_generation.timeseries.sdo_processing import crop_map, flare_check, pad_map, rand_select

data_path = config["paths"]["data_folder"]
test_path = Path(__file__).resolve().parent


def test_rand():
    combined = Table.read(f"{test_path}/data/ts_test_data.parq")
    types = ["F1", "F2", "N1", "N2"]

    # 1. test with full list of samples
    rand_comb_1 = rand_select(combined, 3, types)
    rand_comb_2 = rand_select(combined, 3, types)
    assert list(rand_comb_1) != list(rand_comb_2)
    # 2. test with partial list of samples
    rand_comb_1 = rand_select(combined, 3, ["F1", "N1"])
    rand_comb_2 = rand_select(combined, 3, ["F1", "N1"])
    assert list(rand_comb_1) != list(rand_comb_2)
    # 3. test with higher number of sizes
    rand_comb_1 = rand_select(combined, 6, types)
    rand_comb_2 = rand_select(combined, 6, types)
    assert list(rand_comb_1) != list(rand_comb_2)


@pytest.mark.remote_data
def test_padding():
    map_width = int(config["drms"]["patch_width"])
    map_height = int(config["drms"]["patch_height"])
    aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
    # 1. test with undersized AR cutout on right side of map, check map size.
    center = [-800, 0] * u.arcsec
    center = SkyCoord(*center, frame=aia_map.coordinate_frame)
    aia_smap = crop_map(aia_map, center, map_height * u.pix, (map_width - 100) * u.pix, Time(aia_map.date))
    assert int(pad_map(aia_smap, map_width).dimensions[0].value) == map_width
    # 2. test with undersized AR cutout on left side of map, check map size.
    center = [800, 0] * u.arcsec
    center = SkyCoord(*center, frame=aia_map.coordinate_frame)
    aia_smap = crop_map(aia_map, center, map_height * u.pix, (map_width - 100) * u.pix, Time(aia_map.date))
    assert int(pad_map(aia_smap, map_width).dimensions[0].value) == map_width
    # 3. test with different targ width, to check that it works with arbitrary widths.
    assert int(pad_map(aia_smap, 900).dimensions[0].value) == 900


def test_flare_check():
    combined = Table.read(f"{test_path}/data/ts_test_data.parq")
    flares = combined[combined["goes_class"] != "N"]
    flare = flares[flares["goes_class"] == "C3.7"]
    flare = flare[flare["noaa_number"] == 12644]
    assert (flare_check(flare["start_time"], flare["end_time"], flare["noaa_number"], flares)[0]) == 2
    # 2. test a non flare run known to contain flares
    ar = combined[combined["goes_class"] == "N"]
    ar = ar[ar["noaa_number"] == 12038]
    ar = ar[ar["tb_date"] == "2014-04-23"]
    erl_time = Time(ar["start_time"]) - (6 + 1) * u.hour
    assert (flare_check(erl_time, Time(ar["start_time"]) - 1 * u.hour, ar["noaa_number"], flares)[0]) == 2
    # 3. test a flare run without flares
    flares = combined[combined["goes_class"] != "N"]
    flare = combined[combined["goes_class"] == "M1.6"]
    flare = flare[flare["noaa_number"] == 12192]
    print(flare)
    assert (flare_check(flare["start_time"], flare["end_time"], flare["noaa_number"], flares)[0]) == 1
    # 4. test a non flare run without flares
    ar = combined[combined["goes_class"] == "N"]
    ar = ar[ar["noaa_number"] == 12524]
    ar = ar[ar["tb_date"] == "2016-03-20"]
    erl_time = Time(ar["start_time"]) - (6 + 1) * u.hour
    assert (flare_check(erl_time, Time(ar["start_time"]) - 1 * u.hour, ar["noaa_number"], flares)[0]) == 1
