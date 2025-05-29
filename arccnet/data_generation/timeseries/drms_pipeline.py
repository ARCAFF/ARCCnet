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
    map_reproject,
    match_files,
    read_data,
    table_match,
)

__all__ = []

# Logging settings here.
drms_log = logging.getLogger("drms")
drms_log.setLevel("WARNING")
reproj_log = logging.getLogger("reproject.common")
reproj_log.setLevel("WARNING")
# May need to find a more robust solution with filters/exceptions for this.
astropy_log.setLevel("ERROR")

packed_maps = namedtuple("packed_maps", ["hmi_origin", "l2_map"])

if __name__ == "__main__":
    starts = read_data(
        hek_path="/Users/danielgass/Desktop/ARCCnet/ARCCnet/hek_swpc_1996-01-01T00:00:00-2023-01-01T00:00:00_dev.parq",
        srs_path="/Users/danielgass/Desktop/ARCCnet/ARCCnet/arccnet/data_generation/timeseries/srs_processed_catalog.parq",
        size=10,
        duration=24,
    )
    cores = int(config["drms"]["cores"])
    with ProcessPoolExecutor(cores) as executor:
        for record in [starts[0]]:
            noaa_ar, fl_class, start, end, date, center = record
            pointing_table = calibrate.util.get_pointing_table(source="jsoc", time_range=[start - 6 * u.hour, end])

            start_split = start.value.split("T")[0]
            file_name = f"{fl_class}_{noaa_ar}_{start_split}"
            patch_height = int(config["drms"]["patch_height"]) * u.pix
            patch_width = int(config["drms"]["patch_width"]) * u.pix
            try:
                print(record["noaa_number"], record["goes_class"], record["start_time"])
                aia_maps, hmi_maps = drms_pipeline(
                    start_t=start,
                    end_t=end,
                    path=config["paths"]["data_folder"],
                    hmi_keys=config["drms"]["hmi_keys"],
                    aia_keys=config["drms"]["aia_keys"],
                    wavelengths=config["drms"]["wavelengths"],
                    sample=config["drms"]["sample"],
                )

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
                l2_hmi_packed = [[hmi_origin_patch, hmi_map, noaa_ar] for hmi_map in hmi_proc]
                l2_aia_packed = [[hmi_origin_patch, aia_map, noaa_ar] for aia_map in aia_proc]

                # Went back to tuples because this was failing in a weird way - something to do with pickle and concurrent futures. Left for future debugging.
                # l2_hmi_packed = [packed_maps(hmi_origin_patch, hmi_map, noaa_ar) for hmi_map in hmi_proc]
                # l2_aia_packed = [packed_maps(hmi_origin_patch, aia_map, noaa_ar) for aia_map in aia_proc]

                hmi_patch_paths = tqdm(executor.map(map_reproject, l2_hmi_packed), total=len(l2_hmi_packed))
                aia_patch_paths = tqdm(executor.map(map_reproject, l2_aia_packed), total=len(l2_aia_packed))
                print(hmi_patch_paths)

                # For some reason, aia_proc becomes an empty list after this function call.
                home_table, aia_patch_paths, aia_quality, hmi_patch_paths, hmi_quality = table_match(
                    list(aia_patch_paths), list(hmi_patch_paths)
                )

                # This can probably be streamlined/functionalized to make the pipeline look better.
                batched_name = f"{config['paths']['data_folder']}/04_final"
                Path(f"{batched_name}/records").mkdir(parents=True, exist_ok=True)
                Path(f"{batched_name}/tars").mkdir(parents=True, exist_ok=True)
                hmi_away = ["HMI/" + Path(file).name for file in hmi_patch_paths]
                aia_away = ["AIA/" + Path(file).name for file in aia_patch_paths]
                away_table = Table(
                    {
                        "AIA files": aia_away,
                        "AIA quality": aia_quality,
                        "HMI files": hmi_away,
                        "HMI quality": hmi_quality,
                    }
                )

                home_table.write(f"{batched_name}/records/{file_name}.csv", overwrite=True)

            ## Commented out until we're ready to package.
            # away_table.write(f"{batched_name}/records/out_{file_name}.csv", overwrite=True)
            # with tarfile.open(f"{batched_name}/tars/{file_name}.tar", "w") as tar:
            #     for file in aia_maps:
            #         name = Path(file).name
            #         tar.add(file, arcname=f"AIA/{name}")
            #     for file in np.unique(hmi_maps):
            #         name = Path(file).name
            #         tar.add(file, arcname=f"HMI/{name}")
            #     tar.add(f"{batched_name}/records/out_{file_name}.csv", arcname=f"{file_name}.csv")

            except Exception as error:
                logging.error(error, exc_info=True)
# 70 X class flares.
# Random sample (100 ish for M and below flares)
# Make a .gif of patch for each run
