import tempfile
from pathlib import Path

import pandas as pd

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.data_manager import DataManager
from arccnet.data_generation.mag_processing import MagnetogramProcessor, RegionExtractor
from arccnet.data_generation.region_detection import RegionDetection
from arccnet.data_generation.utils.data_logger import logger

if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")

    data_download = False
    mag_process = False
    region_extraction = True
    region_detection = True

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

    # Build 03_processed directory
    paths_03 = [
        Path(dv.MAG_PROCESSED_FITS_DIR),
        Path(dv.MAG_PROCESSED_QSSUMMARYPLOTS_DIR),
        Path(dv.MAG_PROCESSED_QSFITS_DIR),
    ]
    for path in paths_03:
        if not path.exists():
            path.mkdir(parents=True)

    if region_extraction:
        region_extractor = RegionExtractor(
            dataframe=Path(dv.MAG_INTERMEDIATE_HMIMDI_PROCESSED_DATA_CSV),
            out_fnames=["mdi", "hmi"],
            datetimes=["datetime_mdi", "datetime_hmi"],
            data_cols=["processed_download_path_mdi", "processed_download_path_hmi"],
            # new_cols=["cutout_mdi", "cutout_hmi"],
            cutout_sizes=[
                (int(dv.X_EXTENT / 4), int(dv.Y_EXTENT / 4)),
                (int(dv.X_EXTENT), int(dv.Y_EXTENT)),
            ],
            common_datetime_col="datetime_srs",
            num_random_attempts=10,
        )

        # Save the AR Classification dataset
        region_extractor.activeregion_classification_df.to_csv(
            Path(dv.MAG_PROCESSED_DIR) / Path("ARExtraction.csv"), index=False
        )
        # Drop SRS-related rows (minus "datetime_srs")
        region_extractor.quietsun_df.drop(
            columns=[
                "ID",
                "Number",
                "Carrington Longitude",
                "Area",
                "Z",
                "Longitudinal Extent",
                "Number of Sunspots",
                "Mag Type",
                "Latitude",
                "Longitude",
                "filepath_srs",
                "filename_srs",
                "loaded_successfully_srs",
                "catalog_created_on_srs",
            ]
        ).to_csv(Path(dv.MAG_PROCESSED_DIR) / Path("QSExtraction.csv"), index=False)

    if region_detection:
        hs_match = pd.read_csv(dv.MAG_INTERMEDIATE_HMISHARPS_DATA_CSV)

        # need to alter the original code, but change the download_path to processed data
        hs_match["download_path"] = hs_match["download_path"].str.replace("01_raw", "02_intermediate")
        with tempfile.TemporaryDirectory() as tmpdirname:
            # save the temporary change to the data
            filepath = Path(tmpdirname) / Path("temp.csv")
            hs_match.to_csv(filepath)
            region_detection = RegionDetection(filepath)

            region_detection.regiondetection_df.to_csv(
                Path(dv.MAG_PROCESSED_DIR) / Path("ARDetection.csv"), index=False
            )
