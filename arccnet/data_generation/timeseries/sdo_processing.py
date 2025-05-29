# Define required libraries - check to see if Arccnet already has these as requirements.
import os
import sys
import glob
import itertools
from random import sample
from pathlib import Path

import drms
import numpy as np
import sunpy.map
from aiapy.calibrate import correct_degradation, register, update_pointing
from aiapy.psf import deconvolve
from sunpy.coordinates import frames, propagate_with_solar_surface
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.physics.differential_rotation import solar_rotate_coordinate

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import CompImageHDU
from astropy.table import Table, join, vstack
from astropy.time import Time

from arccnet import config
from arccnet.data_generation.mag_processing import pixel_to_bboxcoords
from arccnet.data_generation.utils.utils import save_compressed_map

os.environ["JSOC_EMAIL"] = "danielgass192@gmail.com"


__all__ = [
    "read_data",
    "hmi_l2",
    "aia_l2",
    "change_time",
    "comp_list",
    "match_files",
    "drms_pipeline",
    "add_fnames",
    "table_match",
    "crop_map",
    "map_reproject",
]


def read_data(hek_path: str, srs_path: str, size: int, duration: int):
    r"""
    Read and process data from a parquet file containing HEK catalogue information regarding flaring events.

    Parameters
    ----------
        hek_path : `str`
            The path to the parquet file containing hek flare information.
        srs_path : `str`
            The path to the parquet file containing parsed noaa srs active region information.
        size : `int`
            The size of the sample to be generated. (Generates 10% X, 30% M, 60% C)
        duration : `int`
            The duration of the data sample in hours.

    Returns
    -------
        `Astropy.Table`
            A table of tuples containing the following columns for each flare:
            - NOAA Active Region Number
            - GOES Flare class (C,M,X classes)
            - Start time (Duration + 1 hours before event in FITS format)
            - End time (1 hour before flaring event start time)
            - The date of the run, used for reprojection
            - The coordinate of the noaa active region
    """

    table = Table.read(hek_path)
    noaa_num_df = table[table["noaa_number"] > 0]
    flares = noaa_num_df[noaa_num_df["event_type"] == "FL"]
    flares = flares[flares["frm_daterun"] > "2011-01-01"]
    srs = Table.read(srs_path)
    srs = srs[srs["number"] > 0]
    srs = srs[~srs["filtered"]]
    srs["srs_date"] = srs["target_time"].value
    srs["srs_date"] = [date.split("T")[0] for date in srs["srs_date"]]
    flares["tb_date"] = flares["start_time"].value
    flares["tb_date"] = [date.split(" ")[0] for date in flares["tb_date"]]
    flares = join(flares, srs, keys_left="noaa_number", keys_right="number")

    flares = flares[abs(flares["longitude"].value) <= 70]
    flares = flares[flares["tb_date"] == flares["srs_date"]]
    x_flares = flares[[flare.startswith("X") for flare in flares["goes_class"]]]
    x_flares = x_flares[x_flares["noaa_number"] == 12192]
    x_flares = x_flares[x_flares["goes_class"] == "X1.0"]
    # x_flares = x_flares[x_flares['tb_date'] == '2014-10-27']
    # x_flares = x_flares[sample(range(len(x_flares)), k=int(0.1 * size))]
    m_flares = flares[[flare.startswith("M") for flare in flares["goes_class"]]]
    m_flares = m_flares[sample(range(len(m_flares)), k=int(0.3 * size))]
    c_flares = flares[[flare.startswith("C") for flare in flares["goes_class"]]]
    c_flares = c_flares[sample(range(len(c_flares)), k=int(0.6 * size))]

    combined = vstack([x_flares, m_flares, c_flares])
    combined["c_coord"] = [
        SkyCoord(lon * u.deg, lat * u.deg, obstime=t_time, observer="earth", frame=frames.HeliographicStonyhurst)
        for lat, lon, t_time in zip(combined["latitude"], combined["longitude"], combined["target_time"])
    ]
    combined["start_time"] = [time - (duration + 1) * u.hour for time in combined["start_time"]]
    combined["start_time"].format = "fits"
    combined["end_time"] = combined["start_time"] + duration * u.hour
    subset = combined["noaa_number", "goes_class", "start_time", "end_time", "frm_daterun", "c_coord"]

    return subset


def change_time(time: str, shift: int):
    r"""
    Change the timestamp by a given time shift.

    Parameters
    ----------
        time : `str`
            A timestamp in FITS format.
        shift : `int`
            The time shift in seconds.

    Returns
    -------
        `str`
            The updated timestamp in FITS format.
    """
    time_d = Time(time, format="fits") + shift * (u.second)
    return time_d.to_value("fits")


def comp_list(file: str, file_list: list):
    r"""
    Check if a file is present in a list of files.

    Parameters
    ----------
        file : `str`
            The file to check.
        file_list :
            `list` The list of files.

    Returns
    -------
        `list`
            A list of booleans for each element in list. True if the file is present, False otherwise.
    """
    return any(file in name for name in file_list)


def match_files(aia_maps, hmi_maps, table):
    r"""
    Matches AIA maps with corresponding HMI maps based on the closest time difference.

    Parameters
    ----------
        aia_maps : `list`
            List of AIA maps.
        hmi_maps : `list`
            List of HMI maps.
        table : `JSOCResponse`
            AIA pointing table as provided by JSOC.

    Returns
    -------
        packed_files : `list`
            A list containing named tuples of paired AIA and HMI maps.
    """
    # matched_maps = namedtuple('matched_maps', ['aia_map', 'hmi_map', 'p_table'])
    packed_files = []
    for aia_map in aia_maps:
        t_d = [abs(aia_map.date - hmi_map.date).to_value(u.s) for hmi_map in hmi_maps]
        hmi_match = hmi_maps[t_d.index(min(t_d))]
        packed_files.append([aia_map, hmi_match, table])
    return packed_files


def add_fnames(maps, paths):
    r"""
    Adds file names to fits map metadata.

    Parameters
    ----------
        maps : `list`
            List of fits maps.
        paths : `list`
            List of file paths.

    Returns
    -------
        named_map : `list`
            List of fits maps with file names added to metadata.
    """
    named_maps = []
    for map, fname in zip(maps, paths):
        map.meta["fname"] = Path(fname).name
        named_maps.append(map)
    return named_maps


def drms_pipeline(
    start_t,
    end_t,
    path: str,
    hmi_keys: list,
    aia_keys: list,
    wavelengths: str = "171, 193, 304, 211, 335, 94, 131, 1600, 4500, 1700",
    sample: int = 60,
):
    r"""
    Performs pipeline to download and process AIA and HMI data.

    Parameters
    ----------
        starts : `list`
            List of start and end times for the data retrieval.
        path : `str`
            Path to save the downloaded data.
        keys : `list`
            List of keys for the data query.
        wavelengths : `str`
            String of wavelengths in list formatting for the AIA data to be provided to drms quotestrings (default all AIA wvl).
        sample : `int`
            Sample rate for the data cadence (default 1/hr).
    Returns
    -------
        aia_maps, hmi_maps : `tuple`
            A tuple containing the AIA maps and HMI maps.
    """
    # print(start_t)
    hmi_query, hmi_export = hmi_query_export(start_t, end_t, hmi_keys, sample)
    aia_query, aia_export = aia_query_export(hmi_query, aia_keys, wavelengths)

    hmi_dls, hmi_exs = l1_file_save(hmi_export, hmi_query, path)
    aia_dls, aia_exs = l1_file_save(aia_export, aia_query, path)

    hmi_maps = sunpy.map.Map(hmi_exs)
    hmi_maps = add_fnames(hmi_maps, hmi_exs)
    aia_maps = sunpy.map.Map(aia_exs)
    aia_maps = add_fnames(aia_maps, aia_exs)
    return aia_maps, hmi_maps


def hmi_query_export(time_1, time_2, keys: list, sample: int):
    r"""
    Query and export HMI data from the JSOC database.

    Parameters
    ----------
        time_1 : `str`
            The start timestamp in FITS format.
        time_2 : `str`
            The end timestamp in FITS format.
        keys : `list`
            A list of keys to query.
        sample : `int`
            The sample rate in minutes.

    Returns
    -------
        hmi_query_full, hmi_result : `tuple`
            A tuple containing the query result (pandas df) and the export data response (drms export object).
    """
    client = drms.Client()
    duration = round((time_2 - time_1).to_value(u.hour))
    qstr_hmi = f"hmi.M_720s[{time_1.value}/{duration}h@{sample}m]" + "{magnetogram}"
    hmi_query = client.query(ds=qstr_hmi, key=keys)

    good_result = hmi_query[hmi_query.QUALITY == 0]
    good_num = good_result["*recnum*"].values
    bad_result = hmi_query[hmi_query.QUALITY != 0]

    qstrs_hmi = [f"hmi.M_720s[{time}]" + "{magnetogram}" for time in bad_result["T_REC"]]
    hmi_values = [hmi_rec_find(qstr, keys) for qstr in qstrs_hmi]
    patched_num = [*hmi_values]

    joined_num = [*good_num, *patched_num]
    joined_num = [str(num) for num in joined_num]
    hmi_num_str = str(joined_num).strip("[]")
    hmi_qstr = f"hmi.M_720s[! recnum in ({hmi_num_str}) !]" + "{magnetogram}"
    hmi_query_full = client.query(ds=hmi_qstr, key=keys)
    hmi_result = client.export(hmi_qstr, method="url", protocol="fits", email=os.environ["JSOC_EMAIL"])
    hmi_result.wait()
    return hmi_query_full, hmi_result


def aia_query_export(hmi_query, keys, wavelength):
    r"""
    Query and export AIA data from the JSOC database.

    Parameters
    ----------
        hmi_query : `drms query`
            The HMI query result.
        keys : `list`
            List of keys to query.
        wavelength : `list`
            List of AIA wavelengths.

    Returns
    -------
        aia_query_full, aia_result (tuple): A tuple containing the query result and the export data response.
    """
    client = drms.Client()
    qstrs_euv = [f"aia.lev1_euv_12s[{time}][{wavelength}]" + "{image}" for time in hmi_query["T_REC"]]
    qstrs_uv = [f"aia.lev1_uv_24s[{time}]{[1600, 1700]}" + "{image}" for time in hmi_query["T_REC"]]
    qstrs_vis = [f"aia.lev1_vis_1h[{time}]{[4500]}" + "{image}" for time in hmi_query["T_REC"]]

    euv_value = [aia_rec_find(qstr, keys, 3, 12) for qstr in qstrs_euv]
    uv_value = [aia_rec_find(qstr, keys, 2, 24) for qstr in qstrs_uv]
    vis_value = [aia_rec_find(qstr, keys, 1, 60) for qstr in qstrs_vis]
    unpacked_aia = list(itertools.chain(euv_value, uv_value, vis_value))
    unpacked_aia = [set for set in unpacked_aia if set is not None]
    unpacked_aia = list(itertools.chain.from_iterable(unpacked_aia))
    joined_num = [str(num) for num in unpacked_aia]
    aia_num_str = str(joined_num).strip("[]")
    aia_comb_qstr = f"aia.lev1[! FSN in ({aia_num_str}) !]" + "{image_lev1}"

    aia_query_full = client.query(ds=aia_comb_qstr, key=keys)

    aia_result = client.export(aia_comb_qstr, method="url", protocol="fits", email=os.environ["JSOC_EMAIL"])
    aia_result.wait()
    return aia_query_full, aia_result


def hmi_rec_find(qstr, keys):
    r"""
    Find the HMI record number for a given query string.

    Parameters
    ----------
        qstr : `str`
            A query string.
        keys : `list`
            List of keys to query.

    Returns
    -------
        `int`
            The HMI record number.
    """
    client = drms.Client()
    retries = 0
    qry = client.query(ds=qstr, key=keys)
    time = sunpy.time.parse_time(qry["T_REC"].values[0])
    while qry["QUALITY"].values[0] != 0 and retries <= 3:
        qry = client.query(ds=f"hmi.M_720s[{time}]" + "{magnetogram}", key=keys)
        time = change_time(time, 720)
        retries += 1
    return qry["*recnum*"].values[0]


def aia_rec_find(qstr, keys, retries, time_add):
    r"""
    Find the AIA record number for a given query string.

    Parameters
    ----------
        qstr : `str`
            A query string.
        keys : `list`
            List of keys to query.

    Returns
    -------
        `int` :
            The AIA FSN.
    """
    client = drms.Client()
    retry = 0
    qry = client.query(ds=qstr, key=keys)
    qstr_head = qstr.split("[")[0]
    if not qry.empty:
        time, wvl = qry["T_REC"].values[0][0:-1], qry["WAVELNTH"].values[0]
        if wvl == "4500":
            return qry["FSN"].values
        while qry["QUALITY"].values[0] != 0 and retry < retries:
            qry = client.query(ds=f"{qstr_head}[{time}][{wvl}]" + "{image}", key=keys)
            time = change_time(time, time_add)
            retry += 1
        if qry["QUALITY"].values[0] == 0:
            return qry["FSN"].values


def l1_file_save(export, query, path):
    r"""
    Save the exported data as level 1 FITS files.

    Parameters
    ----------
        export : `drms export`
            A drms data export.
        query : `drms query`
            A drms query result.
        path : `str`
            A base path to save the files.

    Returns
    -------
        export (drms export), total_files (list) : `tuple`
            A tuple containing the updated export data and the list of saved file paths.
    """
    instr = query["INSTRUME"][0][0:3]
    path_prefix = []
    export.urls.drop_duplicates(ignore_index=True, inplace=True)
    query.drop_duplicates(ignore_index=True, inplace=True)
    for time, wvl in zip(query["T_REC"], query["WAVELNTH"]):
        time = sunpy.time.parse_time(time).to_value("ymdhms")
        year, month, day = time["year"], time["month"], time["day"]
        newdir = f"{path}/01_raw/{year}/{month}/{day}/SDO/{instr}/"
        Path(newdir).mkdir(parents=True, exist_ok=True)
        path_prefix.append(f"{newdir}{int(wvl)}.")

    existing_files = [glob.glob(f"{dirs}*.fits") for dirs in np.unique(path_prefix)]
    existing_files = list(itertools.chain.from_iterable(existing_files))
    matching_files = [comp_list(file, existing_files) for file in export.urls["filename"]]
    missing_files = [not value for value in matching_files]
    export.urls["filename"] = path_prefix + export.urls["filename"]
    if len(export.urls[missing_files].index) > 0:
        total_files = list(export.urls[matching_files].index) + list(export.urls[missing_files].index)
        total_files = export.urls["filename"][total_files]
        export.download(directory="", index=export.urls[missing_files].index)
    else:
        total_files = export.urls["filename"][matching_files]
    return export, total_files.to_list()


def aia_process(aia_map, table, deconv: bool = False, degcorr: bool = False, exnorm: bool = True):
    r"""
    Process an AIA map to level 1.5.

    Parameters
    ----------
        aia_map : `sunpy.map.Map`
            The AIA map to process.
        table : `JSOCResponse`
            A pointing table as provided by get_pointing_table
        deconv : `bool`
            Whether to deconvolve the PSF.
        degcorr : `bool`
            Whether to correct for degradation.
        exnorm : `bool`
            Whether to normalize exposure.

    Returns
    -------
        aia_map : `sunpy.map.Map`
            Processed AIA map.
    """
    if deconv:
        aia_map = deconvolve(aia_map)
    aia_map = update_pointing(aia_map, pointing_table=table)
    aia_map = register(aia_map)
    if degcorr:
        aia_map = correct_degradation(aia_map)
    if exnorm:
        aiad = aia_map.data / aia_map.exposure_time
        aia_map = sunpy.map.Map(aiad.astype(int), aia_map.fits_header)
    return aia_map


def aia_reproject(aia_map, hmi_map):
    r"""
    Reproject an AIA map to the same coordinate system as an HMI map.

    Parameters
    ----------
        aia_map : `sunpy.map.Map`
            The AIA map to reproject.
        hmi_map : `sunpy.map.Map`
            The HMI map to use as the target coordinate system.

    Returns
    -------
        rpr_aia_map : `sunpy.map.Map`
            Reprojected AIA map.
    """
    rpr_aia_map = aia_map.reproject_to(hmi_map.wcs)
    rpr_aia_map.meta["wavelnth"] = aia_map.meta["wavelnth"]
    rpr_aia_map.meta["waveunit"] = aia_map.meta["waveunit"]
    rpr_aia_map.meta["quality"] = aia_map.meta["quality"]
    rpr_aia_map.meta["t_rec"] = aia_map.meta["t_rec"]
    rpr_aia_map.meta["instrume"] = aia_map.meta["instrume"]
    rpr_aia_map.meta["fname"] = aia_map.meta["fname"]
    rpr_aia_map.nickname = aia_map.nickname

    return rpr_aia_map


def hmi_mask(hmi_map):
    r"""
    Mask pixels outside of Rsun_obs in an HMI map.

    Parameters
    ----------
        hmimap : `sunpy.map.Map`
            The HMI map.

    Returns
    -------
        hmimap : `sunpy.map.Map`
            The masked HMI map.
    """
    hpc_coords = all_coordinates_from_map(hmi_map)
    mask = ~coordinate_is_on_solar_disk(hpc_coords)
    hmi_data = hmi_map.data
    hmi_data[mask] = np.nan
    hmi_map = sunpy.map.Map(hmi_data, hmi_map.meta)
    return hmi_map


def hmi_l2(hmi_map):
    r"""
    Processes the HMI map to "level 2" by applying a mask and saving it to the output directory.

    Parameters
    ----------
        hmi_map : `sunpy.map.Map`
            HMI map to be processed.
        overwrite : `bool`
            Flag which determines if l2 files are reproduced and overwritten.

    Returns
    -------
    proc_path : `str`
        Path to the processed HMI map.
    """
    path = config["paths"]["data_folder"]
    time = hmi_map.date.to_value("ymdhms")
    year, month, day = time[0], time[1], time[2]
    map_path = f"{path}/02_intermediate/{year}/{month}/{day}/SDO/{hmi_map.nickname}"
    proc_path = f"{map_path}/02_{hmi_map.meta['fname']}"

    if not os.path.exists(proc_path):
        hmi_map = hmi_mask(hmi_map)
        proc_path = l2_file_save(hmi_map, path)
    # This updates process status for tqdm more effectively.
    sys.stdout.flush()

    return proc_path


def aia_l2(packed_maps):
    r"""
    Processes the AIA map to "level 2" by leveling, rescaling, trimming, and reprojecting it to match the nearest HMI map.

    Parameters
    ----------
        packed_maps : `list`
            List containing the AIA map and its corresponding HMI map.
        overwrite : `bool`
            Flag which determines if l2 files are reproduced and overwritten.

    Returns
    -------
        proc_path : `str`
            Path to the processed AIA map.
    """
    path = config["paths"]["data_folder"]
    aia_map, hmi_match, table = packed_maps
    time = aia_map.date.to_value("ymdhms")
    year, month, day = time[0], time[1], time[2]
    map_path = f"{path}/02_intermediate/{year}/{month}/{day}/SDO/{aia_map.nickname}"
    proc_path = f"{map_path}/02_{aia_map.meta['fname']}"
    if not os.path.exists(proc_path):
        aia_map = aia_process(aia_map, table)
        aia_map = aia_reproject(aia_map, hmi_match)
        proc_path = l2_file_save(aia_map, path)
    # This updates process status for tqdm more effectively.
    sys.stdout.flush()

    return proc_path


def l2_file_save(fits_map, path: str, overwrite: bool = False):
    r"""
    Save a "level 2" FITS map.

    Parameters
    ----------
        fits_map : `sunpy.map.Map`
            The FITS map to save.
        path : `str`
            The path to save the file.
        overwrite : `bool`
            Whether to overwrite existing files.

    Returns
    -------
        fits_path : `str`
            The path of the saved file.
    """
    time = fits_map.date.to_value("ymdhms")
    year, month, day = time[0], time[1], time[2]
    map_path = f"{path}/02_intermediate/{year}/{month}/{day}/SDO/{fits_map.nickname}"
    Path(map_path).mkdir(parents=True, exist_ok=True)
    fits_path = f"{map_path}/02_{fits_map.meta['fname']}"
    if (not Path(fits_path).exists()) or overwrite:
        save_compressed_map(fits_map, fits_path, hdu_type=CompImageHDU, overwrite=True)
    return fits_path


def table_match(aia_maps, hmi_maps):
    r"""
    Matches l3 AIA submaps with corresponding HMI submaps based on the closest time difference, and returns Astropy Table

    Parameters
    ----------
        aia_maps : `list`
            List of AIA map paths.
        hmi_maps : `list`
            List of HMI map paths.

    Returns
    -------
        paired_table : `Astropy.table`
            A list containing tuples of paired AIA and HMI maps.
    """
    aia_wavelnth = []
    aia_paths = []
    aia_quality = []
    hmi_paths = []
    hmi_times = [Time(fits.open(hmi_map)[1].header["date-obs"]) for hmi_map in hmi_maps]
    hmi_quality = []

    for aia_map in aia_maps:
        t_d = [abs((Time(fits.open(aia_map)[1].header["date-obs"]) - hmi_time).value) for hmi_time in hmi_times]
        hmi_match = hmi_maps[t_d.index(min(t_d))]
        aia_paths.append(aia_map)
        hmi_paths.append(hmi_match)
        hmi_quality.append(fits.open(hmi_match)[1].header["quality"])
        aia_quality.append(fits.open(aia_map)[1].header["quality"])
        aia_wavelnth.append(fits.open(aia_map)[1].header["wavelnth"])
    paired_table = Table(
        {
            "Wavelength": aia_wavelnth,
            "AIA files": aia_paths,
            "AIA quality": aia_quality,
            "HMI files": hmi_paths,
            "HMI quality": hmi_quality,
        }
    )
    return paired_table, aia_paths, aia_quality, hmi_paths, hmi_quality


def crop_map(sdo_map, center, height, width, noaa_time):
    r"""
    Crops a provided SDO map and returns a submap centered on a flare according to provided parameters of lat, lon, height and width.

    Parameters
    ----------
        sdo_map : `sunpy.map.Map`
            Provided SDO map path (AIA/HMI).
        center : `SkyCoord`
            Coordinate of center of active region (Heliographic Stonyhurst coordinate system).
        height : `Quantity`
            The height of the submap in u.pix.
        width : `Quantity`
            The width of the submap in u.pix.
        noaa_time : `int`
            The start time registered with the noaa active region.

    Returns
    -------
        s_map : `sunpy.map.Map.submap`
            A submap centered around the provided coordinates.
    """
    sdo_map = sunpy.map.Map(sdo_map)
    t_diff = Time(sdo_map.date) - Time(noaa_time)
    t_diff = t_diff.to_value(u.s) * u.s
    new_center = solar_rotate_coordinate(center, time=t_diff)
    new_center = new_center.transform_to(sdo_map.coordinate_frame)
    pix_center = new_center.to_pixel(sdo_map.wcs)
    top_right, bottom_left = pixel_to_bboxcoords(width, height, pix_center * u.pix)
    s_map = sdo_map.submap(bottom_left=bottom_left, top_right=top_right)
    return s_map


def map_reproject(sdo_packed):
    r"""
    Reprojects a provided SDO map onto the wcs of a provided origin map. As intended, this is to reproject a "level 2" map onto the wcs of a cropped and centered level 3 HMI map.

    Parameters
    ----------
        sdo_packed : `namedtuple`
            Contains the origin map and the sdo map which is to be projected.

    Returns
    ----------
        fits_path : `str`
            The path location of the saved submap.
    """
    hmi_origin, sdo_path, ar_num = sdo_packed
    # sdo_map = sunpy.map.Map(sdo_packed.l2_map)
    sdo_map = sunpy.map.Map(sdo_path)
    with propagate_with_solar_surface():
        sdo_rpr = sdo_map.reproject_to(hmi_origin.wcs)
    time = sdo_map.date.to_value("ymdhms")
    year, month, day = time[0], time[1], time[2]
    path = config["paths"]["data_folder"]
    map_path = f"{path}/03_processed/{year}/{month}/{day}/SDO/{sdo_map.nickname}"
    Path(map_path).mkdir(parents=True, exist_ok=True)
    fits_path = f"{map_path}/{ar_num}_03_{sdo_map.meta['fname']}"
    sdo_rpr.meta["quality"] = sdo_map.meta["quality"]
    sdo_rpr.meta["wavelnth"] = sdo_map.meta["wavelnth"]
    sdo_rpr.meta["date-obs"] = sdo_map.meta["date-obs"]
    save_compressed_map(sdo_rpr, fits_path, hdu_type=CompImageHDU, overwrite=True)

    return fits_path
