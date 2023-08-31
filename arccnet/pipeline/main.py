from pathlib import Path
from datetime import datetime

import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.data_manager import DataManager
from arccnet.data_generation.mag_processing import MagnetogramProcessor, RegionExtractor
from arccnet.data_generation.utils.data_logger import logger

if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")

    data_download = False
    mag_process = False
    region_extraction = True

    if data_download:
        data_manager = DataManager(
            start_date=dv.DATA_START_TIME,
            end_date=dv.DATA_END_TIME,
            merge_tolerance=pd.Timedelta("30m"),
            download_fits=True,
            overwrite_fits=False,
            save_to_csv=True,
        )

    if mag_process:
        mag_processor = MagnetogramProcessor(
            csv_in_file=Path(dv.MAG_INTERMEDIATE_HMIMDI_DATA_CSV),
            csv_out_file=Path(dv.MAG_INTERMEDIATE_HMIMDI_PROCESSED_DATA_CSV),
            columns=["download_path_hmi", "download_path_mdi"],
            processed_data_dir=Path(dv.MAG_INTERMEDIATE_DATA_DIR),
            process_data=True,
            use_multiprocessing=True,
        )

    if region_extraction:
        region_extractor = RegionExtractor(
            dataframe=Path(dv.MAG_INTERMEDIATE_HMIMDI_PROCESSED_DATA_CSV),
            out_fnames=["mdi", "hmi"],
            datetimes=["datetime_mdi", "datetime_hmi"],
            data_cols=["processed_download_path_mdi", "processed_download_path_hmi"],
            new_cols=["cutout_mdi", "cutout_hmi"],
            cutout_sizes=[
                (int(dv.X_EXTENT / 4), int(dv.Y_EXTENT / 4)),
                (int(dv.X_EXTENT), int(dv.Y_EXTENT)),
            ],
            common_datetime_col="datetime_srs",
            num_random_attempts=10,
        )
