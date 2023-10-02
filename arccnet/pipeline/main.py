import logging
from pathlib import Path
from datetime import timedelta

import pandas as pd

from astropy.table import QTable

import arccnet import config
from arccnet.catalogs.active_regions.swpc import ClassificationCatalog, Query, Result, SWPCCatalog, filter_srs
from arccnet.data_generation.data_manager import DataManager
from arccnet.data_generation.magnetograms.instruments import (
    HMILOSMagnetogram,
    HMISHARPs,
    MDILOSMagnetogram,
    MDISMARPs,
)
from arccnet.data_generation.utils.data_logger import get_logger

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


def process_mag(config):
    # provide list
    mag_objs = [
        HMILOSMagnetogram(),
        MDILOSMagnetogram(),
        HMISHARPs(),
        MDISMARPs(),
    ]  # unsure if this should just be put into the config file.
    batch_freq = 4  # move to config
    freq = timedelta(days=1)  # move to config?

    data_root = config["paths"]["data_root"]

    # query files
    hmi_query_file = Path(data_root) / "01_raw" / "mag" / "hmi_query.parq"
    mdi_query_file = Path(data_root) / "01_raw" / "mag" / "mdi_query.parq"
    sharps_query_file = Path(data_root) / "01_raw" / "mag" / "sharps_query.parq"
    smarps_query_file = Path(data_root) / "01_raw" / "mag" / "smarps_query.parq"
    query_files = [hmi_query_file, mdi_query_file, sharps_query_file, smarps_query_file]

    # results files
    hmi_results_file = Path(data_root) / "02_intermediate" / "mag" / "hmi_results.parq"
    mdi_results_file = Path(data_root) / "02_intermediate" / "mag" / "mdi_results.parq"
    sharps_results_file = Path(data_root) / "02_intermediate" / "mag" / "sharps_results.parq"
    smarps_results_file = Path(data_root) / "02_intermediate" / "mag" / "smarps_results.parq"
    results_files = [hmi_results_file, mdi_results_file, sharps_results_file, smarps_results_file]

    hmi_results_file_raw = Path(data_root) / "01_raw" / "mag" / "hmi_results.parq"
    mdi_results_file_raw = Path(data_root) / "01_raw" / "mag" / "mdi_results.parq"
    sharps_results_file_raw = Path(data_root) / "01_raw" / "mag" / "sharps_results.parq"
    smarps_results_file_raw = Path(data_root) / "01_raw" / "mag" / "smarps_results.parq"
    results_files_raw = [hmi_results_file_raw, mdi_results_file_raw, sharps_results_file_raw, smarps_results_file_raw]

    # merged files
    Path(data_root) / "03_final" / "mag" / "srs_hmi_mdi_merged.parq"
    Path(data_root) / "03_final" / "mag" / "hmi_sharps_merged.parq"
    Path(data_root) / "03_final" / "mag" / "mdi_smarps_merged.parq"

    # !TODO implement reading of files if they exist.

    # !TODO consider providing a custom class for each BaseMagnetogram
    dm = DataManager(
        start_date=config["dates"]["start_date"],
        end_date=config["dates"]["end_date"],
        frequency=freq,
        magnetograms=mag_objs,
    )

    query_objects = dm.query_objects

    for qo, qf in zip(query_objects, query_files):
        qo.write(qf, format="parquet", overwrite=True)

    # problem here is that the urls aren't around forever.
    metadata_raw, results_objects = dm.search(batch_frequency=batch_freq)

    for df, rf in zip(metadata_raw, results_files_raw):
        df.to_parquet(path=rf)

    for ro, rf in zip(results_objects, results_files):
        ro.write(rf, format="parquet", overwrite=True)
        # probably want to save [Result.write(..)]

    # Merge SRS, HMI, MDI; srs_hmi_mdi_file
    # Merge HMI SHARPs - hmi_sharps_file
    # Merge MDI SMARPs - hmi_smarps_file
    download_objects = dm.download(
        results_objects, path=Path(data_root) / "02_intermediate" / "mag" / "fits", overwrite=False, retry_missing=False
    )

    # minimal Result
    download_result = [download_obj[["target_time", "url", "path"]] for download_obj in download_objects]

    # this is dodgy...
    srs = None
    hmi = download_result[0]
    mdi = download_result[1]
    sharps = download_result[2]
    smarps = download_result[3]

    # 1. merge SRS-HMI-MDI
    # srs = None
    # hmi = None
    # mdi = None
    # utilise that time_srs and target_time_hmi are the same, e.g. generated from start,end,frequency
    srsmdihmi = pd.merge(srs.add_suffix("_srs"), hmi.add_suffix("_hmi"), left_on="time_srs", right_on="target_time_hmi")
    srsmdihmi = pd.merge(srsmdihmi, mdi.add_suffix("_mdi"), left_on="time_srs", right_on="target_time_mdi")
    dropped_rows = srsmdihmi.copy()
    # maybe change to path_srs/mdi/hmi etc.
    srsmdihmi_dropped = srsmdihmi.dropna(subset=["url_srs"]).reset_index(drop=True)
    srsmdihmi_dropped = srsmdihmi_dropped.dropna(subset=["url_hmi", "url_mdi"], how="all").reset_index(drop=True)
    srsmdihmi_minimal = srsmdihmi_dropped[["path_srs", "url_hmi", "url_mdi"]]
    logger.debug(
        print(
            f"len(srsmdihmi): {len(srsmdihmi)}, len(srsmdihmi_dropped): {len(srsmdihmi_dropped)}; and there are {len(dropped_rows[~dropped_rows.index.isin(srsmdihmi_dropped.index)])} dropped rows"
        )
    )
    logger.debug(srsmdihmi_minimal.head())

    # 2. merge HMI-SHARPs
    # sharps = None
    # hmi = None

    hmi_sharps = pd.merge(
        hmi.dropna(subset=["filename"]).reset_index(drop=True),
        sharps.dropna(subset=["filename"]).reset_index(drop=True).add_suffix("_arc"),
        left_on="datetime",
        right_on="datetime_arc",
    )

    logger.debug(hmi_sharps.head())

    # 3. merge MDI-SMARPs
    # smarps = None
    # mdi = None

    mdi_smarps = pd.merge(
        mdi.dropna(subset=["filename"]).reset_index(drop=True),
        smarps.dropna(subset=["filename"]).reset_index(drop=True).add_suffix("_arc"),
        left_on="datetime",
        right_on="datetime_arc",
    )

    logger.debug(mdi_smarps.head())

    return query_objects, results_objects, download_objects, download_result


def get_config():
    cwd = Path()
    config = {
        "paths": {"data_root": cwd / "data"},
        "dates": {"start_date": "1996-01-01", "end_date": "2022-12-31"},
    }  # until the end of 2022
    return config


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel("DEBUG")

    logger.debug("Starting main")
    query, results, raw_catalog, processed_catalog, clean_cata = process_srs(config)
    mag_query, mag_results, mag_download = process_mag(config)

    process_mag(config)
    logger.debug("Finished main")

if __name__ == "__main__":
    main()
    # old_logger.info(f"Executing {__file__} as main program")

    # data_download = False
    # mag_process = False
    # region_extraction = True
    # region_detection = True

    # if data_download:
    #     data_manager = DataManager(
    #         start_date=dv.DATA_START_TIME,
    #         end_date=dv.DATA_END_TIME,
    #         merge_tolerance=pd.Timedelta("30m"),
    #         download_fits=True,
    #         overwrite_fits=False,
    #         save_to_csv=True,
    #     )

    # if mag_process:
    #     mag_processor = MagnetogramProcessor(
    #         csv_in_file=Path(dv.MAG_INTERMEDIATE_HMIMDI_DATA_CSV),
    #         csv_out_file=Path(dv.MAG_INTERMEDIATE_HMIMDI_PROCESSED_DATA_CSV),
    #         columns=["download_path_hmi", "download_path_mdi"],
    #         processed_data_dir=Path(dv.MAG_INTERMEDIATE_DATA_DIR),
    #         process_data=True,
    #         use_multiprocessing=True,
    #     )

    # # Build 03_processed directory
    # paths_03 = [
    #     Path(dv.MAG_PROCESSED_FITS_DIR),
    #     Path(dv.MAG_PROCESSED_QSSUMMARYPLOTS_DIR),
    #     Path(dv.MAG_PROCESSED_QSFITS_DIR),
    # ]
    # for path in paths_03:
    #     if not path.exists():
    #         path.mkdir(parents=True)

    # if region_extraction:
    #     region_extractor = RegionExtractor(
    #         dataframe=Path(dv.MAG_INTERMEDIATE_HMIMDI_PROCESSED_DATA_CSV),
    #         out_fnames=["mdi", "hmi"],
    #         datetimes=["datetime_mdi", "datetime_hmi"],
    #         data_cols=["processed_download_path_mdi", "processed_download_path_hmi"],
    #         # new_cols=["cutout_mdi", "cutout_hmi"],
    #         cutout_sizes=[
    #             (int(dv.X_EXTENT / 4), int(dv.Y_EXTENT / 4)),
    #             (int(dv.X_EXTENT), int(dv.Y_EXTENT)),
    #         ],
    #         common_datetime_col="datetime_srs",
    #         num_random_attempts=10,
    #     )

    #     # Save the AR Classification dataset
    #     region_extractor.activeregion_classification_df.to_csv(
    #         Path(dv.MAG_PROCESSED_DIR) / Path("ARExtraction.csv"), index=False
    #     )
    #     # Drop SRS-related rows (minus "datetime_srs")
    #     region_extractor.quietsun_df.drop(
    #         columns=[
    #             "ID",
    #             "Number",
    #             "Carrington Longitude",
    #             "Area",
    #             "Z",
    #             "Longitudinal Extent",
    #             "Number of Sunspots",
    #             "Mag Type",
    #             "Latitude",
    #             "Longitude",
    #             "filepath_srs",
    #             "filename_srs",
    #             "loaded_successfully_srs",
    #             "catalog_created_on_srs",
    #         ]
    #     ).to_csv(Path(dv.MAG_PROCESSED_DIR) / Path("QSExtraction.csv"), index=False)

    # if region_detection:
    #     hs_match = pd.read_csv(dv.MAG_INTERMEDIATE_HMISHARPS_DATA_CSV)

    #     # need to alter the original code, but change the download_path to processed data
    #     hs_match["download_path"] = hs_match["download_path"].str.replace("01_raw", "02_intermediate")
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         # save the temporary change to the data
    #         filepath = Path(tmpdirname) / Path("temp.csv")
    #         hs_match.to_csv(filepath)
    #         region_detection = RegionDetection(filepath)

    #         region_detection.regiondetection_df.to_csv(
    #             Path(dv.MAG_PROCESSED_DIR) / Path("ARDetection.csv"), index=False
    #         )
    # sys.exit()
