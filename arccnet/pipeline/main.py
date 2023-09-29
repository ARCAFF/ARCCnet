import sys
import logging
import tempfile
from pathlib import Path

import pandas as pd

from astropy.table import QTable

from arccnet import config
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

    data_dir_raw = Path(config["paths"]["data_dir_raw"])
    data_dir_intermediate = Path(config["paths"]["data_dir_intermediate"])
    data_dir_processed = Path(config["paths"]["data_dir_processed"])
    data_dir_final = Path(config["paths"]["data_dir_final"])

    srs_raw_files_dir = data_dir_raw / "noaa_srs" / "txt"

    srs_query_file = data_dir_raw / "noaa_srs" / "srs_query.parq"
    srs_results_file = data_dir_intermediate / "noaa_srs" / "srs_results.parq"
    srs_raw_catalog_file = data_dir_intermediate / "noaa_srs" / "srs_raw_catalog.parq"
    srs_processed_catalog_file = data_dir_processed / "noaa_srs" / "srs_processed_catalog.parq"
    srs_clean_catalog_file = data_dir_final / "srs_clean_catalog.parq"

    srs_query_file.parent.mkdir(exist_ok=True, parents=True)
    srs_results_file.parent.mkdir(exist_ok=True, parents=True)
    srs_processed_catalog_file.parent.mkdir(exist_ok=True, parents=True)
    srs_clean_catalog_file.parent.mkdir(exist_ok=True, parents=True)

    srs_query = Query.create_empty(config["general"]["start_date"], config["general"]["end_date"])
    if srs_query_file.exists():
        srs_query = Query.read(srs_query_file)

    srs_query = swpc.search(srs_query)
    srs_query.write(srs_query_file, format="parquet", overwrite=True)

    srs_results = srs_query.copy()
    if srs_results_file.exists():
        srs_results = Result.read(srs_results_file, format="parquet")

    srs_results = swpc.download(srs_results, path=srs_raw_files_dir)
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


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel("DEBUG")

    logger.debug("Starting main")
    query, results, raw_catalog, processed_catalog, clean_cata = process_srs(config)
    logger.debug("Finished main")

    old_logger.info(f"Executing {__file__} as main program")

    data_download = False
    mag_process = False
    region_extraction = True
    region_detection = True

    if data_download:
        DataManager(
            start_date=config["general"]["start_date"],
            end_date=config["general"]["end_date"],
            merge_tolerance=pd.Timedelta(config["general"]["merge_tolerance"]),
            download_fits=True,
            overwrite_fits=False,
            save_to_csv=True,
        )

    if mag_process:
        MagnetogramProcessor(
            csv_in_file=Path(config["paths"]["mag_intermediate_hmimdi_data_csv"]),
            csv_out_file=Path(config["paths"]["mag_intermediate_hmimdi_processed_data_csv"]),
            columns=["download_path_hmi", "download_path_mdi"],
            processed_data_dir=Path(config["paths"]["mag_intermediate_data_dir"]),
            process_data=True,
            use_multiprocessing=True,
        )

    # Build 03_processed directory
    paths_03 = [
        Path(config["paths"]["mag_processed_fits_dir"]),
        Path(config["paths"]["mag_processed_qssummaryplots_dir"]),
        Path(config["paths"]["mag_processed_qsfits_dir"]),
    ]
    for path in paths_03:
        if not path.exists():
            path.mkdir(parents=True)

    if region_extraction:
        x_extent = config.getint("magnetograms.cutouts", "x_extent")
        y_extent = config.getint("magnetograms.cutouts", "y_extent")
        region_extractor = RegionExtractor(
            dataframe=Path(config["paths"]["mag_intermediate_hmimdi_processed_data_csv"]),
            out_fnames=["mdi", "hmi"],
            datetimes=["datetime_mdi", "datetime_hmi"],
            data_cols=["processed_download_path_mdi", "processed_download_path_hmi"],
            # new_cols=["cutout_mdi", "cutout_hmi"],
            cutout_sizes=[
                (int(x_extent / 4), int(x_extent / 4)),
                (int(y_extent), int(y_extent)),
            ],
            common_datetime_col="datetime_srs",
            num_random_attempts=10,
        )

        # Save the AR Classification dataset
        region_extractor.activeregion_classification_df.to_csv(
            Path(config["paths"]["mag_processed_dir"]) / "ARExtraction.csv", index=False
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
        ).to_csv(Path(config["paths"]["mag_processed_dir"]) / "QSExtraction.csv", index=False)

    if region_detection:
        hs_match = pd.read_csv(config["paths"]["mag_intermediate_hmisharps_data_csv"])

        # need to alter the original code, but change the download_path to processed data
        hs_match["download_path"] = hs_match["download_path"].str.replace("01_raw", "02_intermediate")
        with tempfile.TemporaryDirectory() as tmpdirname:
            # save the temporary change to the data
            filepath = Path(tmpdirname) / Path("temp.csv")
            hs_match.to_csv(filepath)
            region_detection = RegionDetection(filepath)

            region_detection.regiondetection_df.to_csv(
                Path(config["paths"]["mag_processed_dir"]) / "ARDetection.csv", index=False
            )


if __name__ == "__main__":
    main()
    sys.exit()
