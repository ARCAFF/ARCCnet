import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from aiapy import calibrate
from tqdm import tqdm

import astropy.units as u
from astropy import log
from astropy.table import Table

from arccnet import config
from arccnet.data_generation.timeseries.sdo_processing import (
    aia_l2,
    crop_map,
    drms_pipeline,
    hmi_l2,
    match_files,
    read_data,
    table_match,
)

# Logging settings here.
# May need to find a more robust solution with filters/exceptions for this.
log.setLevel("ERROR")

if __name__ == "__main__":
    starts = read_data(
        "/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/hek_swpc_1996-01-01T00:00:00-2023-01-01T00:00:00_dev.parq", 10, 6
    )
    cores = int(config["drms"]["cores"])
    with ProcessPoolExecutor(cores) as executor:
        for record in [starts[1]]:
            noaa_ar, fl_class, start, end, date, lat, lon = record
            pointing_table = calibrate.util.get_pointing_table(source="jsoc", time_range=[start - 3 * u.hour, end])
            date = start.value.split("T")[0]
            file_name = f"{fl_class}_{noaa_ar}_{date}"
            patch_height = int(config["drms"]["patch_height"])
            patch_width = int(config["drms"]["patch_width"])
            try:
                print(record)
                aia_maps, hmi_maps = drms_pipeline(
                    start_t=start,
                    end_t=end,
                    path=config["paths"]["data_folder"],
                    hmi_keys=config["drms"]["hmi_keys"],
                    aia_keys=config["drms"]["aia_keys"],
                    wavelengths=config["drms"]["wavelengths"],
                    sample=config["drms"]["sample"],
                )

                hmi_proc = tqdm(
                    executor.map(hmi_l2, hmi_maps),
                    total=len(hmi_maps),
                )
                packed_files = match_files(aia_maps, hmi_maps, pointing_table)

                aia_proc = tqdm(
                    executor.map(aia_l2, packed_files),
                    total=len(aia_maps),
                )

                l2_hmi_packed = [[hmi_map, lat, lon, patch_height, patch_width] for hmi_map in hmi_proc]
                l2_aia_packed = [[aia_map, lat, lon, patch_height, patch_width] for aia_map in aia_proc]

                hmi_patch_paths = tqdm(executor.map(crop_map, l2_hmi_packed), total=len(l2_hmi_packed))
                aia_patch_paths = tqdm(executor.map(crop_map, l2_aia_packed), total=len(l2_aia_packed))

                # For some reason, aia_proc becomes an empty list after this function call.
                home_table, aia_patch_paths, aia_quality, hmi_patch_paths, hmi_quality = table_match(
                    list(aia_patch_paths), list(hmi_patch_paths)
                )

                # This can probably streamlined/functionalized to make the pipeline look better.
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
            #         name = PurePath(file).name
            #         tar.add(file, arcname=f"AIA/{name}")
            #     for file in np.unique(hmi_maps):
            #         name = PurePath(file).name
            #         tar.add(file, arcname=f"HMI/{name}")
            #     tar.add(f"{batched_name}/records/out_{file_name}.csv", arcname=f"{file_name}.csv")

            except Exception as error:
                logging.error(error, exc_info=True)
# 70 X class flares.
# Read the flare list.
# Not just HEK, look for save files for flares.
# Random sample (100 ish for M and below flares)
