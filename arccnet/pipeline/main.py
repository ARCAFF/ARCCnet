import sys
import logging
from pathlib import Path
from datetime import timedelta

from astropy.table import QTable, join

from arccnet import config
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
    if srs_query_file.exists():  # this is fine only if the query agrees
        srs_query = Query.read(srs_query_file)

    srs_query = swpc.search(srs_query)
    srs_query.write(srs_query_file, format="parquet", overwrite=True)

    srs_results = srs_query.copy()
    if srs_results_file.exists():
        srs_results = Result.read(srs_results_file, format="parquet")

    srs_results = swpc.download(srs_results, path=srs_raw_files_dir, progress=True)
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

    return (
        srs_query,
        srs_results,
        srs_raw_catalog,
        srs_processed_catalog,
        srs_clean_catalog,
    )


def process_hmi(config):
    mag_objs = [
        HMILOSMagnetogram(),
        HMISHARPs(),
    ]

    data_root = config["paths"]["data_root"]
    download_path = Path(data_root) / "02_intermediate" / "mag" / "fits"

    # query files
    hmi_query_file = Path(data_root) / "01_raw" / "mag" / "hmi_query.parq"
    sharps_query_file = Path(data_root) / "01_raw" / "mag" / "sharps_query.parq"
    # results files
    hmi_results_file_raw = Path(data_root) / "01_raw" / "mag" / "hmi_results.parq"
    sharps_results_file_raw = Path(data_root) / "01_raw" / "mag" / "sharps_results.parq"
    hmi_results_file = Path(data_root) / "02_intermediate" / "mag" / "hmi_results.parq"
    sharps_results_file = Path(data_root) / "02_intermediate" / "mag" / "sharps_results.parq"
    # save the downloads files in 02_intermediate as they do not link to processed data
    hmi_downloads_file = Path(data_root) / "02_intermediate" / "mag" / "hmi_downloads.parq"
    sharps_downloads_file = Path(data_root) / "02_intermediate" / "mag" / "sharps_downloads.parq"

    hmi_query_file.parent.mkdir(exist_ok=True, parents=True)
    hmi_results_file_raw.parent.mkdir(exist_ok=True, parents=True)
    hmi_results_file.parent.mkdir(exist_ok=True, parents=True)
    hmi_downloads_file.parent.mkdir(exist_ok=True, parents=True)

    download_objects = _process_mag(
        config=config,
        download_path=download_path,
        mag_objs=mag_objs,
        query_files=[hmi_query_file, sharps_query_file],
        results_files_raw=[hmi_results_file_raw, sharps_results_file_raw],
        results_files=[hmi_results_file, sharps_results_file],
        downloads_files=[hmi_downloads_file, sharps_downloads_file],
        freq=timedelta(days=1),
        batch_frequency=3,
        merge_tolerance=timedelta(minutes=30),
        overwrite_downloads=False,
    )

    return download_objects


def process_mdi(config):
    mag_objs = [
        MDILOSMagnetogram(),
        MDISMARPs(),
    ]

    data_root = config["paths"]["data_root"]
    download_path = Path(data_root) / "02_intermediate" / "mag" / "fits"

    # query files
    mdi_query_file = Path(data_root) / "01_raw" / "mag" / "mdi_query.parq"
    smarps_query_file = Path(data_root) / "01_raw" / "mag" / "smarps_query.parq"
    # results files
    mdi_results_file_raw = Path(data_root) / "01_raw" / "mag" / "mdi_results.parq"
    smarps_results_file_raw = Path(data_root) / "01_raw" / "mag" / "smarps_results.parq"
    mdi_results_file = Path(data_root) / "02_intermediate" / "mag" / "mdi_results.parq"
    smarps_results_file = Path(data_root) / "02_intermediate" / "mag" / "smarps_results.parq"
    # save the downloads files in 02_intermediate as they do not link to processed data
    mdi_downloads_file = Path(data_root) / "02_intermediate" / "mag" / "mdi_downloads.parq"
    smarps_downloads_file = Path(data_root) / "02_intermediate" / "mag" / "smarps_downloads.parq"

    mdi_query_file.parent.mkdir(exist_ok=True, parents=True)
    mdi_results_file_raw.parent.mkdir(exist_ok=True, parents=True)
    mdi_results_file.parent.mkdir(exist_ok=True, parents=True)
    mdi_downloads_file.parent.mkdir(exist_ok=True, parents=True)

    download_objects = _process_mag(
        config=config,
        download_path=download_path,
        mag_objs=mag_objs,
        query_files=[mdi_query_file, smarps_query_file],
        results_files_raw=[mdi_results_file_raw, smarps_results_file_raw],
        results_files=[mdi_results_file, smarps_results_file],
        downloads_files=[mdi_downloads_file, smarps_downloads_file],
        freq=timedelta(days=1),
        batch_frequency=3,
        merge_tolerance=timedelta(minutes=30),
        overwrite_downloads=False,
    )

    return download_objects


def _process_mag(
    config,
    download_path,
    mag_objs,
    query_files,
    results_files_raw,
    results_files,
    downloads_files,
    freq=timedelta(days=1),
    batch_frequency=3,
    merge_tolerance=timedelta(minutes=30),
    overwrite_downloads=False,
):
    # !TODO implement reading of files if they exist.
    # !TODO consider providing a custom class for each BaseMagnetogram
    dm = DataManager(
        start_date=config["general"]["start_date"],
        end_date=config["general"]["end_date"],
        frequency=freq,
        magnetograms=mag_objs,
    )

    query_objects = dm.query_objects

    for qo, qf in zip(query_objects, query_files):
        qo.write(qf, format="parquet", overwrite=True)

    # problem here is that the urls aren't around forever.
    metadata_raw, results_objects = dm.search(
        batch_frequency=batch_frequency,
        merge_tolerance=merge_tolerance,
    )

    for df, rf in zip(metadata_raw, results_files_raw):
        print(rf)
        df.to_parquet(path=rf)  # this is a `DataFrame`

    for ro, rf in zip(results_objects, results_files):
        print(rf)
        ro.write(rf, format="parquet", overwrite=True)

    download_objects = dm.download(
        results_objects,
        path=download_path,
        overwrite=overwrite_downloads,
    )

    for do, dfiles in zip(download_objects, downloads_files):
        print(dfiles)
        do.write(dfiles, format="parquet", overwrite=True)

    # !TODO implement processing of magnetogram

    return download_objects


def merge_mag_tables(config, srs, hmi, mdi, sharps, smarps):
    """
    return merged files:
        - srs-hmi-mdi
        - hmi-sharps
        - mdi-sharps
    """
    data_root = config["paths"]["data_root"]
    srs_hmi_mdi_merged_file = Path(data_root) / "04_final" / "mag" / "srs_hmi_mdi_merged.parq"
    hmi_sharps_merged_file = Path(data_root) / "04_final" / "mag" / "hmi_sharps_merged.parq"
    mdi_smarps_merged_file = Path(data_root) / "04_final" / "mag" / "mdi_smarps_merged.parq"
    srs_hmi_mdi_merged_file.parent.mkdir(exist_ok=True, parents=True)

    #     # this is dodgy...
    #     srs = srs_catalog.copy()
    #     hmi = download_objects[0]  # .to_pandas()
    #     mdi = download_objects[1]  # .to_pandas()
    #     sharps = download_objects[2]  # .to_pandas()
    #     smarps = download_objects[3]  # .to_pandas()

    #     # 1. merge SRS-HMI-MDI
    #     # srs = None
    #     # hmi = None
    #     # mdi = None
    #     # utilise that time_srs and target_time_hmi are the same, e.g. generated from start,end,frequency
    #     # srsmdihmi = pd.merge(srs.add_suffix("_srs"), hmi.add_suffix("_hmi"), left_on="time_srs", right_on="target_time_hmi")
    #     # srsmdihmi = pd.merge(srsmdihmi, mdi.add_suffix("_mdi"), left_on="time_srs", right_on="target_time_mdi")
    #     # dropped_rows = srsmdihmi.copy()
    #     # # maybe change to path_srs/mdi/hmi etc.
    #     # srsmdihmi_dropped = srsmdihmi.dropna(subset=["url_srs"]).reset_index(drop=True) # drop on url, not on path...
    #     # srsmdihmi_dropped = srsmdihmi_dropped.dropna(subset=["url_hmi", "url_mdi"], how="all").reset_index(drop=True)
    #     # srsmdihmi_minimal = srsmdihmi_dropped[["path_srs", "path_hmi", "path_mdi"]]
    #     # logger.debug(
    #     #     print(
    #     #         f"len(srsmdihmi): {len(srsmdihmi)}, len(srsmdihmi_dropped): {len(srsmdihmi_dropped)}; and there are {len(dropped_rows[~dropped_rows.index.isin(srsmdihmi_dropped.index)])} dropped rows"
    #     #     )
    #     # )
    #     # logger.debug(srsmdihmi_minimal.head())

    #     # QTable.from_pandas(srsmdihmi_dropped).write(srs_hmi_mdi_merged_file, format="parquet", overwrite=True)

    #     # There are issues with saving the QTable after doing the pandas merge (e.g. 'QUALITY_mdi','DATAVALS_mdi' go from int64 -> object)
    #     # srsmdihmi[['QUALITY_mdi','DATAVALS_mdi']]

    # There must be a better way to rename columns
    srs_renamed = srs.copy()  # Create a copy of the original table
    for colname in srs_renamed.colnames:
        srs_renamed.rename_column(colname, colname + "_srs")

    hmi_renamed = hmi.copy()  # Create a copy of the original table
    for colname in hmi_renamed.colnames:
        hmi_renamed.rename_column(colname, colname + "_hmi")

    mdi_renamed = mdi.copy()  # Create a copy of the original table
    for colname in mdi_renamed.colnames:
        mdi_renamed.rename_column(colname, colname + "_mdi")

    srsmdihmi_table = join(
        QTable(srs_renamed),
        QTable(hmi_renamed),
        keys_left="time_srs",
        keys_right="target_time_hmi",
        table_names=["srs", "hmi"],
    )
    srsmdihmi_table = join(
        srsmdihmi_table,
        QTable(mdi_renamed),
        keys_left="time_srs",
        keys_right="target_time_mdi",
        table_names=["srs_hmi", "mdi"],
    )

    print(srsmdihmi_table["url_srs"])

    # Drop rows with NaN values in the 'url_srs' column
    srsmdihmi_dropped = srsmdihmi_table[~srsmdihmi_table["url_srs"].mask]

    # Drop rows where 'url_hmi' and 'url_mdi' are all NaN
    srsmdihmi_dropped = srsmdihmi_dropped[~(srsmdihmi_dropped["url_hmi"].mask & srsmdihmi_dropped["url_mdi"].mask)]
    srsmdihmi_dropped.write(srs_hmi_mdi_merged_file, format="parquet", overwrite=True)

    # 2. merge HMI-SHARPs
    hmi_filtered = QTable(hmi.copy())
    hmi_filtered = hmi_filtered[~hmi_filtered["filename"].mask].copy()

    sharps_filtered = QTable(sharps.copy())
    sharps_filtered = sharps_filtered[~sharps_filtered["filename"].mask].copy()
    for colname in sharps_filtered.colnames:
        sharps_filtered.rename_column(colname, colname + "_arc")

    hmi_sharps_table = join(
        hmi_filtered,
        sharps_filtered,
        keys_left="datetime",
        keys_right="datetime_arc",
        table_names=["hmi", "sharps"],
    )
    hmi_sharps_table.write(hmi_sharps_merged_file, format="parquet", overwrite=True)

    # 3. merge MDI-SMARPs
    mdi_filtered = QTable(mdi.copy())
    mdi_filtered = mdi_filtered[~mdi_filtered["filename"].mask].copy()

    smarps_filtered = QTable(smarps.copy())
    smarps_filtered = smarps_filtered[~smarps_filtered["filename"].mask].copy()
    for colname in smarps_filtered.colnames:
        smarps_filtered.rename_column(colname, colname + "_arc")

    mdi_smarps_table = join(
        mdi_filtered,
        smarps_filtered,
        keys_left="datetime",
        keys_right="datetime_arc",
        table_names=["mdi", "smarps"],
    )
    mdi_smarps_table.write(mdi_smarps_merged_file, format="parquet", overwrite=True)

    return srsmdihmi_dropped, hmi_sharps_table, mdi_smarps_table


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel("DEBUG")

    logger.debug("Starting main")
    _, _, _, _, clean_catalog = process_srs(config)
    hmi_download_obj, sharps_download_obj = process_hmi(config)
    mdi_download_obj, smarps_download_obj = process_mdi(config)
    _ = merge_mag_tables(
        config,
        srs=clean_catalog,
        hmi=hmi_download_obj,
        mdi=mdi_download_obj,
        sharps=sharps_download_obj,
        smarps=smarps_download_obj,
    )
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
    sys.exit()
