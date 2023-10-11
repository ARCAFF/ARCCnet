import sys
import logging
from pathlib import Path
from datetime import timedelta

import pandas as pd

import astropy.units as u
from astropy.table import QTable, join

from arccnet import config
from arccnet.catalogs.active_regions.swpc import ClassificationCatalog, Query, Result, SWPCCatalog, filter_srs
from arccnet.data_generation.data_manager import DataManager
from arccnet.data_generation.data_manager import Query as MagQuery
from arccnet.data_generation.mag_processing import MagnetogramProcessor, RegionExtractor
from arccnet.data_generation.magnetograms.instruments import (
    HMILOSMagnetogram,
    HMISHARPs,
    MDILOSMagnetogram,
    MDISMARPs,
)
from arccnet.data_generation.region_detection import RegionDetection
from arccnet.data_generation.utils.data_logger import get_logger

logger = get_logger(__name__, logging.DEBUG)


def process_srs(config):
    logger.info(f"Processing SRS with config: {config}")  # should print_config()
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
    """
    Process HMI (Helioseismic and Magnetic Imager) data.

    Parameters
    ----------
    config : `dict`
        A dictionary containing configuration parameters.

    Returns
    -------
    `list`
        A list of download objects for processed HMI data.
    """
    logger.info("Processing HMI/SHARPs")

    mag_objs = [
        HMILOSMagnetogram(),
        HMISHARPs(),
    ]

    data_root = config["paths"]["data_root"]
    download_path = Path(data_root) / "02_intermediate" / "mag" / "fits"
    processed_path = Path(data_root) / "04_final" / "mag" / "fits"

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
    hmi_processed_file = Path(data_root) / "04_final" / "mag" / "hmi_processed.parq"

    download_path.mkdir(exist_ok=True, parents=True)
    processed_path.mkdir(exist_ok=True, parents=True)
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

    processed_data = MagnetogramProcessor(download_objects[0], save_path=processed_path, column_name="path")
    processed_table = processed_data.process(use_multiprocessing=True, overwrite=False)
    logger.debug(f"Writing {hmi_processed_file}")
    processed_table.write(hmi_processed_file, format="parquet", overwrite=True)

    return [processed_table, download_objects[1]]


def process_mdi(config):
    """
    Process MDI (Michelson Doppler Imager) data.

    Parameters
    ----------
    config : `dict`
        A dictionary containing configuration parameters.

    Returns
    -------
    `list`
        A list of download objects for processed MDI data.
    """
    logger.info("Processing MDI/SMARPs")

    mag_objs = [
        MDILOSMagnetogram(),
        MDISMARPs(),
    ]

    data_root = config["paths"]["data_root"]
    download_path = Path(data_root) / "02_intermediate" / "mag" / "fits"
    processed_path = Path(data_root) / "04_final" / "mag" / "fits"

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
    mdi_processed_file = Path(data_root) / "04_final" / "mag" / "mdi_processed.parq"

    download_path.mkdir(exist_ok=True, parents=True)
    processed_path.mkdir(exist_ok=True, parents=True)
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

    processed_data = MagnetogramProcessor(download_objects[0], save_path=processed_path, column_name="path")
    processed_table = processed_data.process(use_multiprocessing=True, overwrite=False)
    logger.debug(f"Writing {mdi_processed_file}")
    processed_table.write(mdi_processed_file, format="parquet", overwrite=True)

    return [processed_table, download_objects[1]]


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
    """
    Process magnetogram data using specified magnetogram objects.

    Parameters
    ----------
    config : `dict`
        A dictionary containing configuration parameters.
    download_path : `str`
        Path where downloaded data will be saved.
    mag_objs : `list`
        List of magnetogram objects to be processed.
    query_files : `list`
        List of query files.
    results_files_raw : `list`
        List of raw results files.
    results_files : `list`
        List of processed results files.
    downloads_files : `list`
        List of download files.
    freq : `timedelta`, optional
        Frequency of data processing (default: timedelta(days=1)).
    batch_frequency : `int`, optional
        Batch frequency for data processing (default: 3).
    merge_tolerance : `timedelta`, optional
        Merge tolerance for data processing (default: timedelta(minutes=30)).
    overwrite_downloads : `bool`, optional
        Whether to overwrite existing download data (default: False).

    Returns
    -------
    `list`
        A list of download objects for processed magnetogram data.
    """
    # !TODO consider providing a custom class for each BaseMagnetogram
    dm = DataManager(
        start_date=config["general"]["start_date"],
        end_date=config["general"]["end_date"],
        frequency=freq,
        magnetograms=mag_objs,
    )

    query_objects = dm.query_objects

    # read raw_results and results_objects if all exist.
    all_files_exist = all(file.exists() for file in results_files_raw + results_files)
    if all_files_exist:
        logger.debug("Loading results files")
        metadata_raw = []
        results_objects = []
        for raw_file, file in zip(results_files_raw, results_files):
            logger.debug(f"reading {str(file)}")
            metadata_raw.append(pd.read_parquet(raw_file))
            results_objects.append(MagQuery(QTable.read(file, format="parquet")))
    else:
        # only save the query object if we're not loading the results_files
        for qo, qf in zip(query_objects, query_files):
            logger.debug(f"Writing {qf}")
            qo.write(qf, format="parquet", overwrite=True)

        logger.debug("performing search")
        # problem here is that the urls aren't around forever.
        metadata_raw, results_objects = dm.search(
            batch_frequency=batch_frequency,
            merge_tolerance=merge_tolerance,
        )

        for df, rfr in zip(metadata_raw, results_files_raw):
            logger.debug(f"Writing {rfr}")
            df.to_parquet(path=rfr)  # this is a `DataFrame`

        for ro, rf in zip(results_objects, results_files):
            logger.debug(f"Writing {rf}")
            ro.write(rf, format="parquet", overwrite=True)

    download_objects = dm.download(
        results_objects,
        path=download_path,
        overwrite=overwrite_downloads,
    )

    for do, dfiles in zip(download_objects, downloads_files):
        logger.debug(f"Writing {dfiles}")
        do.write(dfiles, format="parquet", overwrite=True)

    return download_objects


def merge_mag_tables(config, srs, hmi, mdi, sharps, smarps):
    """
    Merge magnetogram data tables from different sources.

    Parameters
    ----------
    config : dict
        A dictionary containing configuration parameters.
    srs : QTable
        SRS (Solar Region Summary) data table.
    hmi : QTable
        Processed HMI (Helioseismic and Magnetic Imager) data table.
    mdi : QTable
        Processed MDI (Michelson Doppler Imager) data table.
    sharps : QTable
        SHARPs (Space-weather HMI Active Region Patches) data table.
    smarps : QTable
        SMARPs (Space-weather MDI Active Region Patches) data table.

    Returns
    -------
    tuple
        Three merged data tables: (srs_hmi_mdi, hmi_sharps, mdi_smarps).
    """
    # !TODO move to separate functions
    logger.info("Merging magnetogram tables")

    data_root = config["paths"]["data_root"]
    srs_hmi_mdi_merged_file = Path(data_root) / "04_final" / "mag" / "srs_hmi_mdi_merged.parq"
    hmi_sharps_merged_file = Path(data_root) / "04_final" / "mag" / "hmi_sharps_merged.parq"
    mdi_smarps_merged_file = Path(data_root) / "04_final" / "mag" / "mdi_smarps_merged.parq"
    srs_hmi_mdi_merged_file.parent.mkdir(exist_ok=True, parents=True)

    catalog_mdi = join(
        QTable(srs),
        QTable(mdi),
        keys_left="time",
        keys_right="target_time",
        table_names=["catalog", "image"],
    )
    # attempting to remove the object
    catalog_mdi.replace_column("path_catalog", [str(pc) for pc in catalog_mdi["path_catalog"]])

    catalog_hmi = join(
        QTable(srs),
        QTable(hmi),
        keys_left="time",
        keys_right="target_time",
        table_names=["catalog", "image"],
    )
    # attempting to remove the object
    catalog_hmi.replace_column("path_catalog", [str(pc) for pc in catalog_hmi["path_catalog"]])

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

    # Drop rows with NaN values in the 'url_srs' column
    srsmdihmi_dropped = srsmdihmi_table[~srsmdihmi_table["url_srs"].mask]

    # Drop rows where 'url_hmi' and 'url_mdi' are all NaN
    srsmdihmi_dropped = srsmdihmi_dropped[~(srsmdihmi_dropped["url_hmi"].mask & srsmdihmi_dropped["url_mdi"].mask)]
    logger.debug(f"Writing {srs_hmi_mdi_merged_file}")
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
    logger.debug(f"Writing {hmi_sharps_merged_file}")
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
    logger.debug(f"Writing {mdi_smarps_merged_file}")
    mdi_smarps_table.write(mdi_smarps_merged_file, format="parquet", overwrite=True)

    return catalog_hmi, catalog_mdi, srsmdihmi_dropped, hmi_sharps_table, mdi_smarps_table


def region_extraction(config, srs_hmi, srs_mdi):
    data_root = config["paths"]["data_root"]
    data_plot_path = Path(data_root) / "04_final" / "mag" / "region_extraction" / "fits"
    summary_plot_path = Path(data_root) / "04_final" / "mag" / "region_extraction" / "summary"

    data_plot_path.mkdir(exist_ok=True, parents=True)
    summary_plot_path.mkdir(exist_ok=True, parents=True)

    hmi = RegionExtractor(srs_hmi).extract_regions(
        cutout_size=(
            int(config["magnetograms.cutouts"]["x_extent"]) * u.pix,
            int(config["magnetograms.cutouts"]["y_extent"]) * u.pix,
        ),
        data_path=data_plot_path,
        summary_plot_path=summary_plot_path,
        qs_random_attempts=10,
    )
    mdi = RegionExtractor(srs_mdi).extract_regions(
        cutout_size=(
            int(config["magnetograms.cutouts"]["x_extent"]) / 2 * u.pix,
            int(config["magnetograms.cutouts"]["y_extent"]) / 2 * u.pix,
        ),
        data_path=data_plot_path,
        summary_plot_path=summary_plot_path,
        qs_random_attempts=10,
    )

    # save?
    return hmi, mdi


def region_detection(config, hmi_sharps, mdi_smarps):
    data_root = config["paths"]["data_root"]
    region_detection_path = Path(data_root) / "04_final" / "mag" / "region_detection"
    region_detection_path.mkdir(exist_ok=True, parents=True)

    hmidetection = RegionDetection(table=hmi_sharps)
    hmi_sharps_detection_table, hmi_sharps_detection_bboxes = hmidetection.get_bboxes()

    mdidetection = RegionDetection(table=mdi_smarps)
    mdi_smarps_detection_table, mdi_smarps_detection_bboxes = mdidetection.get_bboxes()

    return mdi_smarps_detection_table, hmi_sharps_detection_table


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel("DEBUG")

    logger.debug("Starting main")
    _, _, _, _, clean_catalog = process_srs(config)
    hmi_download_obj, sharps_download_obj = process_hmi(config)
    mdi_download_obj, smarps_download_obj = process_mdi(config)
    srs_hmi, srs_mdi, srs_mdi_hmi, hmi_sharps, mdi_smarps = merge_mag_tables(
        config,
        srs=clean_catalog,
        hmi=hmi_download_obj,
        mdi=mdi_download_obj,
        sharps=sharps_download_obj,
        smarps=smarps_download_obj,
    )

    srs_hmi_cutouts, srs_mdi_cutouts = region_extraction(config, srs_hmi, srs_mdi)
    srs_hmi_cutouts_ar, srs_hmi_cutouts_qs = srs_hmi_cutouts
    srs_mdi_cutouts_ar, srs_mdi_cutouts_qs = srs_mdi_cutouts
    # need to convert the Path column to string to save.

    mdi_smarps_detection_table, hmi_sharps_detection_table = region_detection(config, hmi_sharps, mdi_smarps)

    logger.debug("Finished main")


if __name__ == "__main__":
    main()
    sys.exit()
