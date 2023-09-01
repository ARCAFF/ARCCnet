from pathlib import Path

import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.data_manager import DataManager
from arccnet.data_generation.mag_processing import MagnetogramProcessor
from arccnet.data_generation.utils.data_logger import logger

if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")

    data_download = True
    mag_process = True

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
