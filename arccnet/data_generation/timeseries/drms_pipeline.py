import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from astropy.table import Table

from arccnet import config
from arccnet.data_generation.timeseries.sdo_processing import (
    aia_l2,
    drms_pipeline,
    hmi_l2,
    l2_table_match,
    match_files,
    read_data,
)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    starts = read_data(
        "/Users/danielgass/Desktop/ARCCnetDan/ARCCnet/hek_swpc_1996-01-01T00:00:00-2023-01-01T00:00:00_dev.parq", 10, 6
    )
    cores = int(config["drms"]["cores"])
    with ProcessPoolExecutor(cores) as executor:
        for record in [starts[2]]:
            noaa_ar, fl_class, start, end = record[0], record[1], record[2], record[3]
            date = start.value.split("T")[0]
            file_name = f"{fl_class}_{noaa_ar}_{date}"
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
                    desc=f"{fl_class}_{start}_HMI Processing",
                )
                packed_files = match_files(aia_maps, hmi_maps)
                aia_proc = tqdm(
                    executor.map(aia_l2, packed_files),
                    total=len(aia_maps),
                    desc=f"{fl_class}_{start}_AIA Processing",
                )
                # For some reason, aia_proc becomes an empty list after this function call.
                home_table, aia_maps, aia_quality, hmi_maps, hmi_quality = l2_table_match(
                    list(aia_proc), list(hmi_proc)
                )

                # This can probably streamlined/functionalized to make the pipeline look better.
                batched_name = f"{config['paths']['data_folder']}/04_final"
                Path(f"{batched_name}/records").mkdir(parents=True, exist_ok=True)
                Path(f"{batched_name}/tars").mkdir(parents=True, exist_ok=True)
                aia_away = ["AIA/" + Path(file).name for file in aia_maps]
                hmi_away = ["HMI/" + Path(file).name for file in hmi_maps]
                away_table = Table(
                    {
                        "AIA files": aia_away,
                        "AIA quality": aia_quality,
                        "HMI files": hmi_away,
                        "HMI quality": hmi_quality,
                    }
                )

                home_table.write(f"{batched_name}/records/{file_name}.csv", overwrite=True)

            ## Commented out until we're ready to deliver.
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
