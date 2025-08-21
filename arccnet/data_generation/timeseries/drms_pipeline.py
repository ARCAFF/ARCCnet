import logging
from pathlib import Path
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

from aiapy import calibrate
from tqdm import tqdm

import astropy.units as u
from astropy import log as astropy_log
from astropy.table import Table

from arccnet import config
from arccnet.data_generation.timeseries.sdo_processing import (
    aia_l2,
    crop_map,
    drms_pipeline,
    hmi_l2,
    l4_file_pack,
    map_reproject,
    match_files,
    read_data,
    table_match,
    vid_match,
)

if __name__ == "__main__":
    __all__ = []

    # Logging settings here.
    drms_log = logging.getLogger("drms")
    drms_log.setLevel("ERROR")
    reproj_log = logging.getLogger("reproject.common")
    reproj_log.setLevel("ERROR")
    # May need to find a more robust solution with filters/exceptions for this.
    astropy_log.setLevel("ERROR")
    data_path = config["paths"]["data_folder"]
    packed_maps = namedtuple("packed_maps", ["hmi_origin", "l2_map"])
    starts = read_data(
        hek_path=Path(f"{data_path}/flare_files/hek_swpc_1996-01-01T00:00:00-2023-01-01T00:00:00_dev.parq"),
        srs_path=Path(f"{data_path}/flare_files/srs_processed_catalog.parq"),
        size=1,
        duration=6,
        long_lim=65,
        types=["F1", "F2", "N1", "N2"],
    )[0]

    cores = int(config["drms"]["cores"])
    with ProcessPoolExecutor(cores) as executor:
        for record in starts:
            noaa_ar, fl_class, start, end, date, center, category = record
            pointing_table = calibrate.util.get_pointing_table(source="jsoc", time_range=[start - 6 * u.hour, end])
            start_split = start.value.split("T")[0]
            file_name = f"{category}_{start_split}_{fl_class}_{noaa_ar}"
            patch_height = int(config["drms"]["patch_height"]) * u.pix
            patch_width = int(config["drms"]["patch_width"]) * u.pix
            try:
                logging.info(
                    f"{record['noaa_number']} {record['goes_class']} {record['start_time']} {record['category']}"
                )
                aia_maps, hmi_maps = drms_pipeline(
                    start_t=start,
                    end_t=end,
                    path=config["paths"]["data_folder"],
                    hmi_keys=config["drms"]["hmi_keys"],
                    aia_keys=config["drms"]["aia_keys"],
                    wavelengths=config["drms"]["wavelengths"],
                    sample=config["drms"]["sample"],
                )
                if len(aia_maps) != 60:
                    logging.info("Bad run - missing frames, skipping.")
                    continue

                hmi_proc = list(
                    tqdm(
                        executor.map(hmi_l2, hmi_maps),
                        total=len(hmi_maps),
                    )
                )

                packed_files = match_files(aia_maps, hmi_maps, pointing_table)
                aia_proc = tqdm(
                    executor.map(aia_l2, packed_files),
                    total=len(aia_maps),
                )
                packed_maps = namedtuple("packed_maps", ["hmi_origin", "l2_map", "ar_num"])
                hmi_origin_patch = crop_map(hmi_proc[0], center, patch_height, patch_width, date)
                l2_hmi_packed = [[hmi_origin_patch, hmi_map, noaa_ar, center] for hmi_map in hmi_proc]
                l2_aia_packed = [[hmi_origin_patch, aia_map, noaa_ar, center] for aia_map in aia_proc]

                # Went back to tuples because this was failing in a weird way - something to do with pickle and concurrent futures. Left for future debugging.
                # l2_hmi_packed = [packed_maps(hmi_origin_patch, hmi_map, noaa_ar) for hmi_map in hmi_proc]
                # l2_aia_packed = [packed_maps(hmi_origin_patch, aia_map, noaa_ar) for aia_map in aia_proc]

                hmi_patch_paths = tqdm(executor.map(map_reproject, l2_hmi_packed), total=len(l2_hmi_packed))
                aia_patch_paths = tqdm(executor.map(map_reproject, l2_aia_packed), total=len(l2_aia_packed))

                # For some reason, aia_proc becomes an empty list after this function call.
                home_table, aia_patch_paths, aia_quality, aia_time, hmi_patch_paths, hmi_quality, hmi_time = (
                    table_match(list(aia_patch_paths), list(hmi_patch_paths))
                )

                batched_name = f"{config['paths']['data_folder']}/04_final"
                Path(f"{batched_name}/records").mkdir(parents=True, exist_ok=True)
                Path(f"{batched_name}/tars").mkdir(parents=True, exist_ok=True)
                hmi_away = ["HMI/" + Path(file).name for file in hmi_patch_paths]
                aia_away = ["AIA/" + Path(file).name for file in aia_patch_paths]
                aia_wvl = home_table["Wavelength"]
                away_table = Table(
                    {
                        "AIA wavelength": aia_wvl,
                        "AIA files": aia_away,
                        "AIA quality": aia_quality,
                        "HMI files": hmi_away,
                        "HMI quality": hmi_quality,
                    }
                )

                home_table.write(f"{batched_name}/records/{file_name}.csv", overwrite=True)

                vid_path = vid_match(home_table, file_name, batched_name)
                l4_file_pack(aia_patch_paths, hmi_patch_paths, batched_name, file_name, away_table, vid_path)

            except Exception as error:
                logging.error(error, exc_info=True)
