import sys
import logging
import tempfile
from pathlib import Path

import pandas as pd

from astropy.table import QTable

import arccnet.data_generation.utils.default_variables as dv
from arccnet.catalogs.active_regions.swpc import ClassificationCatalog, Query, Result, SWPCCatalog, filter_srs
from arccnet.data_generation.data_manager import DataManager
from arccnet.data_generation.mag_processing import MagnetogramProcessor, RegionExtractor
from arccnet.data_generation.region_detection import RegionDetection
from arccnet.data_generation.utils.data_logger import get_logger
from arccnet.data_generation.utils.data_logger import logger as old_logger

logger = get_logger(__name__, logging.DEBUG)


def process_srs(config):
    logger.info(f"Processing SRS with config: {config}")
    swpc = SWPCCatalog()

    data_root = config["paths"]["data_root"]

    srs_query_file = Path(data_root) / "01_raw" / "noaa_srs" / "srs_query.parq"
    srs_results_file = Path(data_root) / "02_intermediate" / "noaa_srs" / "srs_results.parq"
    srs_raw_catalog_file = Path(data_root) / "02_intermediate" / "noaa_srs" / "srs_raw_catalog.parq"
    srs_processed_catalog_file = Path(data_root) / "03_final" / "noaa_srs" / "srs_processed_catalog.parq"
    srs_clean_catalog_file = Path(data_root) / "03_final" / "noaa_srs" / "srs_clean_catalog.parq"

    srs_query_file.parent.mkdir(exist_ok=True, parents=True)
    srs_results_file.parent.mkdir(exist_ok=True, parents=True)
    srs_processed_catalog_file.parent.mkdir(exist_ok=True, parents=True)

    srs_query = Query.create_empty(config["dates"]["start_date"], config["dates"]["end_date"])
    if srs_query_file.exists():
        srs_query = Query.read(srs_query_file)

    srs_query = swpc.search(srs_query)
    srs_query.write(srs_query_file, format="parquet", overwrite=True)

    # move to reporting / vis
    # num_expected_srs = len(query)
    # num_missing_srs = sum(query["url"].mask)
    # num_found_srs = num_expected_srs - num_missing_srs
    # logger.info(f"Found: {num_found_srs} out of {num_expected_srs} SRS files")

    srs_results = srs_query.copy()
    if srs_results_file.exists():
        srs_results = Result.read(srs_results_file, format="parquet")

    srs_results = swpc.download(srs_results, path=data_root / "01_raw" / "noaa_srs" / "txt")
    srs_results.write(srs_results_file, format="parquet", overwrite=True)

    srs_raw_catalog = srs_results.copy()
    if srs_raw_catalog_file.exists():
        srs_raw_catalog = ClassificationCatalog.read(srs_raw_catalog_file)

    srs_raw_catalog = swpc.create_catalog(srs_raw_catalog)
    srs_raw_catalog.write(srs_raw_catalog_file, format="parquet", overwrite=True)

    srs_processed_catalog = srs_raw_catalog.copy()
    if srs_processed_catalog_file.exists():
        srs_processed_catalog = ClassificationCatalog.read(srs_processed_catalog_file)

    srs_processed_catalog = filter_srs(srs_processed_catalog)
    srs_processed_catalog.write(srs_processed_catalog_file, format="parquet", overwrite=True)

    srs_clean_catalog = QTable(srs_processed_catalog)[srs_processed_catalog["filtered"] == False]  # noqa
    srs_clean_catalog.write(srs_clean_catalog_file, format="parquet", overwrite=True)

    return srs_query, srs_results, srs_raw_catalog, srs_processed_catalog, srs_clean_catalog


def get_config():
    cwd = Path()
    config = {"paths": {"data_root": cwd / "data"}, "dates": {"start_date": "1996-01-01", "end_date": "2023-01-01"}}
    return config


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel("DEBUG")

    logger.debug("Starting main")
    config = get_config()
    query, results, raw_catalog, processed_catalog, clean_cata = process_srs(config)
    logger.debug("Finished main")
    return 0


if __name__ == "__main__":
    main()
    old_logger.info(f"Executing {__file__} as main program")

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
    sys.exit()
