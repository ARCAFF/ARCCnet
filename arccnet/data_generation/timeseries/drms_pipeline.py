import logging
from time import perf_counter
from pathlib import Path
from itertools import repeat
from collections import namedtuple
from multiprocessing import Semaphore
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

    ss = perf_counter()

    drms_limit = Semaphore(4)
    # Logging settings here.
    drms_log = logging.getLogger("drms")
    drms_log.setLevel("ERROR")
    reproj_log = logging.getLogger("reproject.common")
    reproj_log.setLevel("ERROR")
    astropy_log.setLevel("ERROR")
    data_path = config["paths"]["data_folder"]
    wavelengths = config["drms"]["wavelengths"]
    packed_maps = namedtuple("packed_maps", ["hmi_origin", "l2_map"])
    starts, before_fl_tables, after_fl_tables = read_data(
        hek_path=Path(f"{data_path}/flare_files/hek_swpc_1996-01-01T00:00:00-2023-01-01T00:00:00_dev.parq"),
        srs_path=Path(f"{data_path}/flare_files/srs_processed_catalog.parq"),
        # Set size to -1 for all AR's in a year
        size=10,
        duration=6,
        long_lim=65,
        # Use these instead of years if generating old flare target data.
        # types=["F1", "F2", "N1", "N2"],
        years=[2014],
    )

    cores = int(config["drms"]["cores"])

    with ProcessPoolExecutor(None) as executor:
        for rec_num in range(len(starts)):
            record = starts[rec_num]
            noaa_ar, mag_class, mcintosh, end, start, date, center = record
            before_fls = before_fl_tables[rec_num]
            after_fls = after_fl_tables[rec_num]
            b_x, b_m, b_c = before_fls[1]["X"], before_fls[1]["M"], before_fls[1]["C"]
            a_x, a_m, a_c = after_fls[1]["X"], after_fls[1]["M"], after_fls[1]["C"]
            pointing_table = calibrate.util.get_pointing_table(source="jsoc", time_range=[start - 6 * u.hour, end])
            start_split = end.value.split("T")[0]
            file_name = (
                f"{start_split}_{noaa_ar}_{mag_class}_{mcintosh}_Xb{b_x}_Mb{b_m}_Cb{b_c}_Xa{a_x}_Ma{a_m}_Ca{a_c}"
            )
            patch_height = int(config["drms"]["patch_height"]) * u.pix
            patch_width = int(config["drms"]["patch_width"]) * u.pix
            try:
                logging.info(file_name)
                aia_maps, hmi_maps = drms_pipeline(
                    start_t=start,
                    end_t=end,
                    path=config["paths"]["data_folder"],
                    hmi_keys=config["drms"]["hmi_keys"],
                    aia_keys=config["drms"]["aia_keys"],
                    wavelengths=config["drms"]["wavelengths"],
                    sample=config["drms"]["sample"],
                    drms_limit=drms_limit,
                )
                # WILL NEED TO ADJUST IF USING MORE/LESS THAN 6 TIME STEPS
                if len(aia_maps) != (60):
                    logging.info("Bad run - missing frames, skipping.")
                    continue

                hmi_proc = list(
                    tqdm(
                        executor.map(hmi_l2, hmi_maps),
                        total=len(hmi_maps),
                    )
                )

                packed_files = match_files(aia_maps, hmi_maps, pointing_table)
                aia_proc = tqdm(executor.map(aia_l2, packed_files), total=len(aia_maps), desc="AIA prep")
                packed_maps = namedtuple("packed_maps", ["hmi_origin", "l2_map", "ar_num"])
                hmi_origin_patch = crop_map(hmi_proc[0], center, patch_height, patch_width, date)
                # l2_hmi_packed = ((hmi_origin_patch, hmi_map, noaa_ar, center) for hmi_map in hmi_proc)
                # l2_aia_packed = ((hmi_origin_patch, aia_map, noaa_ar, center) for aia_map in aia_proc)

                # Went back to tuples because this was failing in a weird way - something to do with pickle and concurrent futures. Left for future debugging.
                # l2_hmi_packed = [packed_maps(hmi_origin_patch, hmi_map, noaa_ar) for hmi_map in hmi_proc]
                # l2_aia_packed = [packed_maps(hmi_origin_patch, aia_map, noaa_ar) for aia_map in aia_proc]

                hmi_patch_paths = tqdm(
                    executor.map(map_reproject, repeat(hmi_origin_patch), hmi_proc, repeat(noaa_ar)),
                    total=len(hmi_proc),
                    desc="HMI reprojection",
                )
                aia_patch_paths = tqdm(
                    executor.map(map_reproject, repeat(hmi_origin_patch), aia_proc, repeat(noaa_ar)),
                    total=len(aia_proc),
                    desc="AIA reprojection",
                )

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
                l4_file_pack(
                    aia_patch_paths,
                    hmi_patch_paths,
                    batched_name,
                    file_name,
                    away_table,
                    before_fls[0],
                    after_fls[0],
                    vid_path,
                )

            except Exception as error:
                logging.error(error, exc_info=True)

    ee = perf_counter()
    print(f"Total time took {(ee - ss) / 60} minutes.")
