import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

import astropy.units as u
from astropy.table import Column, MaskedColumn, QTable, Table, join, vstack

import arccnet
from arccnet import config
from arccnet.catalogs.active_regions.swpc import ClassificationCatalog, Query, Result, SWPCCatalog, filter_srs
from arccnet.catalogs.flares.common import FlareCatalog
from arccnet.catalogs.flares.hek import HEKFlareCatalog
from arccnet.catalogs.flares.helio import HECFlareCatalog
from arccnet.catalogs.utils import retrieve_noaa_mapping
from arccnet.data_generation.data_manager import DataManager
from arccnet.data_generation.data_manager import Query as MagQuery
from arccnet.data_generation.mag_processing import MagnetogramProcessor, RegionExtractor
from arccnet.data_generation.magnetograms.instruments import (
    HMILOSMagnetogram,
    HMISHARPs,
    MDILOSMagnetogram,
    MDISMARPs,
)
from arccnet.data_generation.region_detection import RegionDetection, RegionDetectionTable
from arccnet.utils.logging import get_logger

logger = get_logger(__name__, logging.DEBUG)


def process_srs(config):
    logger.info(f"Processing SRS with config: {config}")  # should print_config()
    swpc = SWPCCatalog()

    data_dir_raw = Path(config["paths"]["data_dir_raw"])
    data_dir_intermediate = Path(config["paths"]["data_dir_intermediate"])
    data_dir_processed = Path(config["paths"]["data_dir_processed"])
    data_dir_final = Path(config["paths"]["data_dir_final"])

    srs_raw_files_dir = data_dir_raw / "metadata" / "noaa_srs" / "txt"

    srs_query_file = data_dir_raw / "metadata" / "noaa_srs" / "srs_query.parq"
    srs_results_file = data_dir_intermediate / "metadata" / "noaa_srs" / "srs_results.parq"
    srs_raw_catalog_file = data_dir_intermediate / "metadata" / "noaa_srs" / "srs_raw_catalog.parq"
    srs_processed_catalog_file = data_dir_processed / "metadata" / "noaa_srs" / "srs_processed_catalog.parq"
    srs_clean_catalog_file = data_dir_final / "srs_clean_catalog.parq"

    srs_query_file.parent.mkdir(exist_ok=True, parents=True)
    srs_results_file.parent.mkdir(exist_ok=True, parents=True)
    srs_processed_catalog_file.parent.mkdir(exist_ok=True, parents=True)
    srs_clean_catalog_file.parent.mkdir(exist_ok=True, parents=True)

    srs_query = Query.create_empty(
        config["general"]["start_date"].isoformat(), config["general"]["end_date"].isoformat()
    )

    # Check if the query file exists
    if srs_query_file.exists():
        # Read the query from the file
        file_query = Query.read(srs_query_file)

        # Raise an error if the queries do not match
        identical = (
            QTable(srs_query)[["start_time", "end_time"]]
            .to_pandas()
            .equals(QTable(file_query)[["start_time", "end_time"]].to_pandas())
        )

        if not identical:
            msg = "The requested SRS query does not match the saved query"
            logger.error(msg)
            raise ValueError(msg)
        else:
            srs_query = file_query

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

    srs_processed_catalog = filter_srs(
        catalog=srs_processed_catalog,
        lat_limit=float(config["srs"]["lat_lim_degrees"]) * u.degree,
        lon_limit=float(config["srs"]["lon_lim_degrees"]) * u.degree,
    )
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


def process_flares(config):
    logger.info("Processing Flare with config")

    catalogs = [
        HEKFlareCatalog(catalog="swpc"),
        HEKFlareCatalog(catalog="ssw_latest"),
        HECFlareCatalog(catalog="gevloc"),
        HECFlareCatalog(catalog="goes"),
    ]

    data_dir_raw = Path(config["paths"]["data_dir_raw"])
    data_dir_intermediate = Path(config["paths"]["data_dir_intermediate"])
    data_dir_processed = Path(config["paths"]["data_dir_processed"])
    Path(config["paths"]["data_dir_final"])

    flare_catalogs = {c.catalog: None for c in catalogs}

    for catalog in catalogs:
        version = arccnet.__version__ if "dev" not in arccnet.__version__ else "dev"  # unless it's a release use dev
        start = config["general"]["start_date"].isoformat()
        end = config["general"]["end_date"].isoformat()
        start = start if isinstance(start, datetime) else datetime.fromisoformat(start)
        end = end if isinstance(end, datetime) else datetime.fromisoformat(end)
        file_name = (
            f"{catalog.catalog}_{config['general']['start_date'].isoformat()}"
            f"-{config['general']['end_date'].isoformat()}_{version}.parq"
        )

        flare_query_file = data_dir_raw / "metadata" / "flares" / file_name
        flare_raw_catalog_file = data_dir_intermediate / "metadata" / "flares" / file_name
        flare_processed_catalog_file = data_dir_processed / "metadata" / "flares" / file_name

        flare_query_file.parent.mkdir(exist_ok=True, parents=True)
        flare_raw_catalog_file.parent.mkdir(exist_ok=True, parents=True)
        flare_processed_catalog_file.parent.mkdir(exist_ok=True, parents=True)

        if flare_query_file.exists():  # this is fine only if the query agrees
            try:
                flare_query = Table.read(flare_query_file)
            except ValueError as e:
                if "No include_names specified" in str(e):  # Astropy bug (#16236) can't read empty table from parquet
                    flare_query = Table()

        else:
            flare_query = catalog.search(
                start_time=config["general"]["start_date"], end_time=config["general"]["end_date"]
            )
            flare_query.write(flare_query_file, format="parquet")

        if flare_raw_catalog_file.exists():
            try:
                flare_raw_catalog = FlareCatalog.read(flare_raw_catalog_file)
            except ValueError as e:
                if "No include_names specified" in str(e):  # Astropy bug (#16236) can't read empty table from parquet
                    flare_raw_catalog = Table()
        else:
            flare_raw_catalog = catalog.create_catalog(flare_query)
            flare_raw_catalog.write(flare_raw_catalog_file, format="parquet", overwrite=True)

            flare_catalogs[catalog.catalog] = {"query": flare_query, "catalog": flare_raw_catalog}

    return flare_catalogs


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
    download_path = Path(data_root) / "02_intermediate" / "data" / "mag" / "fits" / "hmi"
    processed_path = Path(data_root) / "03_processed" / "data" / "mag" / "fits" / "hmi"

    # query files
    hmi_query_file = Path(data_root) / "01_raw" / "data" / "mag" / "hmi_query.parq"
    sharps_query_file = Path(data_root) / "01_raw" / "data" / "mag" / "sharps_query.parq"
    # results files
    hmi_results_file_raw = Path(data_root) / "01_raw" / "data" / "mag" / "hmi_results_empty.parq"
    sharps_results_file_raw = Path(data_root) / "01_raw" / "data" / "mag" / "sharps_results_empty.parq"
    hmi_results_file = Path(data_root) / "02_intermediate" / "data" / "mag" / "hmi_results.parq"
    sharps_results_file = Path(data_root) / "02_intermediate" / "data" / "mag" / "sharps_results.parq"
    # save the downloads files in 02_intermediate as they do not link to processed data
    hmi_downloads_file = Path(data_root) / "02_intermediate" / "data" / "mag" / "hmi_downloads.parq"
    sharps_downloads_file = Path(data_root) / "02_intermediate" / "data" / "mag" / "sharps_downloads.parq"
    hmi_processed_file = Path(data_root) / "03_processed" / "data" / "mag" / "hmi_processed.parq"

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
        results_files_empty=[hmi_results_file_raw, sharps_results_file_raw],
        results_files=[hmi_results_file, sharps_results_file],
        downloads_files=[hmi_downloads_file, sharps_downloads_file],
        freq=timedelta(days=1),
        batch_frequency=3,
        merge_tolerance=timedelta(minutes=30),
        overwrite_downloads=False,
    )

    processed_data = MagnetogramProcessor(download_objects[0], save_path=processed_path, column_name="path")
    processed_table = processed_data.process(use_multiprocessing=False, overwrite=False)
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
    download_path = Path(data_root) / "02_intermediate" / "data" / "mag" / "fits" / "mdi"
    processed_path = Path(data_root) / "03_processed" / "data" / "mag" / "fits" / "mdi"

    # query files
    mdi_query_file = Path(data_root) / "01_raw" / "data" / "mag" / "mdi_query.parq"
    smarps_query_file = Path(data_root) / "01_raw" / "data" / "mag" / "smarps_query.parq"
    # results files
    mdi_results_file_raw = Path(data_root) / "01_raw" / "data" / "mag" / "mdi_results_empty.parq"
    smarps_results_file_raw = Path(data_root) / "01_raw" / "data" / "mag" / "smarps_results_empty.parq"
    mdi_results_file = Path(data_root) / "02_intermediate" / "data" / "mag" / "mdi_results.parq"
    smarps_results_file = Path(data_root) / "02_intermediate" / "data" / "mag" / "smarps_results.parq"
    # save the downloads files in 02_intermediate as they do not link to processed data
    mdi_downloads_file = Path(data_root) / "02_intermediate" / "data" / "mag" / "mdi_downloads.parq"
    smarps_downloads_file = Path(data_root) / "02_intermediate" / "data" / "mag" / "smarps_downloads.parq"
    mdi_processed_file = Path(data_root) / "03_processed" / "data" / "mag" / "mdi_processed.parq"

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
        results_files_empty=[mdi_results_file_raw, smarps_results_file_raw],
        results_files=[mdi_results_file, smarps_results_file],
        downloads_files=[mdi_downloads_file, smarps_downloads_file],
        freq=timedelta(days=1),
        batch_frequency=4,
        merge_tolerance=timedelta(minutes=30),
        overwrite_downloads=False,
    )

    processed_data = MagnetogramProcessor(download_objects[0], save_path=processed_path, column_name="path")
    processed_table = processed_data.process(use_multiprocessing=False, overwrite=False)
    logger.debug(f"Writing {mdi_processed_file}")
    processed_table.write(mdi_processed_file, format="parquet", overwrite=True)

    return [processed_table, download_objects[1]]


def _process_mag(
    config,
    download_path,
    mag_objs,
    query_files,
    results_files_empty,
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
    results_files_empty : `list`
        List of empty results files.
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

    # read empty_results and results_objects if all exist.
    all_files_exist = all(file.exists() for file in results_files)
    if all_files_exist:
        logger.debug("Loading results files")
        results_objects = []
        for file in results_files:
            logger.debug(f"... reading {str(file)}")
            results_objects.append(MagQuery(QTable.read(file, format="parquet")))
    else:
        # only save the query object if we're not loading the results_files
        for qo, qf in zip(query_objects, query_files):
            logger.debug(f"Writing {qf}")
            qo.write(qf, format="parquet", overwrite=True)

        logger.debug("performing search")
        # problem here is that the urls aren't around forever.
        metadata_empty_search, results_objects = dm.search(
            batch_frequency=batch_frequency,
            merge_tolerance=merge_tolerance,
        )

        # write the empty search (a `DataFrame`) to parquet
        for df, rfr in zip(metadata_empty_search, results_files_empty):
            logger.debug(f"Writing {rfr}")
            df.to_parquet(path=rfr)

        # write the results (with urls) to a parquet file
        for ro, rf in zip(results_objects, results_files):
            logger.debug(f"Writing {rf}")
            ro.write(rf, format="parquet", overwrite=True)

    download_objects = dm.download(
        results_objects,
        path=download_path,
        overwrite=overwrite_downloads,
    )

    # write the table (with downloaded file paths) to a parquet file
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
    srs_hmi_merged_file = Path(data_root) / "03_processed" / "data" / "mag" / "srs_hmi_merged.parq"
    srs_mdi_merged_file = Path(data_root) / "03_processed" / "data" / "mag" / "srs_mdi_merged.parq"
    srs_hmi_mdi_merged_file = Path(data_root) / "03_processed" / "data" / "mag" / "srs_hmi_mdi_merged.parq"
    hmi_sharps_merged_file = Path(data_root) / "03_processed" / "data" / "mag" / "hmi_sharps_merged.parq"
    mdi_smarps_merged_file = Path(data_root) / "03_processed" / "data" / "mag" / "mdi_smarps_merged.parq"
    srs_hmi_mdi_merged_file.parent.mkdir(exist_ok=True, parents=True)

    # ---------
    # MDI - SRS
    # ---------

    # filter MDI on QUALITY
    filtered_mdi = mdi.copy()
    filtered_mdi.rename_column("processed_path", "processed_path_image")

    # -- Ensure `filtered` and `filter_reason`columns exists
    if "filtered" not in filtered_mdi.colnames:
        filtered_mdi.add_column(Column(np.full(len(filtered_mdi), False, dtype=bool), name="filtered"))
    if "filter_reason" not in filtered_mdi.colnames:
        filtered_mdi.add_column(Column(np.full(len(filtered_mdi), "", dtype=str), name="filter_reason"))

    # -- Convert the "filter_reason" column to a numpy array
    filter_reason_column_mdi = np.array(filtered_mdi["filter_reason"], dtype=object)
    # Now, update the "filter_reason" column only for masked rows
    for idx, row in enumerate(filtered_mdi):
        if filtered_mdi["processed_path_image"].mask[idx]:
            row["filtered"] = True
            filter_reason_column_mdi[row.index] += "no_magnetogram,"
        if not filtered_mdi["QUALITY"].mask[idx]:
            # QUALITY limit from https://github.com/mbobra/SMARPs/blob/main/example_gallery/Compare_SMARP_and_SHARP_bitmaps.ipynb
            if np.int32(int(filtered_mdi["QUALITY"][idx], 16)) >= 262144:
                row["filtered"] = True
                filter_reason_column_mdi[row.index] += "QUALITY,"
            if (int(filtered_mdi["QUALITY"][idx], 16) & 0b01111100) != 0:
                # checking MDI bits 2,3,4,5,6 (http://soi.stanford.edu/production/QUALITY/DATASWtable.html)
                # https://docs.astropy.org/en/stable/nddata/bitmask.html
                row["filtered"] = True
                filter_reason_column_mdi[row.index] += "QUALITY(Missing%),"

    filtered_mdi["filter_reason"] = Column([str(reason) for reason in filter_reason_column_mdi], dtype=str)

    # -- Join SRS and HMI
    # and combine their filtered/filter_reason columns
    catalog_mdi = join(
        QTable(srs),
        QTable(filtered_mdi),
        keys="target_time",
        table_names=["catalog", "image"],
    )
    catalog_mdi["filtered"] = _combine_filtered_columns(catalog_mdi["filtered_catalog"], catalog_mdi["filtered_image"])
    catalog_mdi["filter_reason"] = _combine_filter_reason_columns(
        catalog_mdi["filter_reason_catalog"], catalog_mdi["filter_reason_image"]
    )
    catalog_mdi.remove_columns(["filtered_catalog", "filtered_image", "filter_reason_catalog", "filter_reason_image"])
    catalog_mdi.replace_column("path_catalog", [str(pc) for pc in catalog_mdi["path_catalog"]])
    logger.debug(f"Writing {srs_mdi_merged_file}")
    catalog_mdi.write(srs_mdi_merged_file, format="parquet", overwrite=True)

    # ---------
    # HMI - SRS
    # ---------

    # filter HMI on QUALITY
    filtered_hmi = hmi.copy()
    filtered_hmi.rename_column("processed_path", "processed_path_image")

    # -- Ensure `filtered` and `filter_reason`columns exists
    if "filtered" not in filtered_hmi.colnames:
        filtered_hmi.add_column(Column(np.full(len(filtered_hmi), False, dtype=bool), name="filtered"))
    if "filter_reason" not in filtered_hmi.colnames:
        filtered_hmi.add_column(Column(np.full(len(filtered_hmi), "", dtype=str), name="filter_reason"))

    # -- Convert the "filter_reason" column to a numpy array of dtype=object
    filter_reason_column_hmi = np.array(filtered_hmi["filter_reason"], dtype=object)
    # Now, update the "filter_reason" column only for masked rows
    for idx, row in enumerate(filtered_hmi):
        if filtered_hmi["processed_path_image"].mask[idx]:
            row["filtered"] = True
            filter_reason_column_hmi[row.index] += "no_magnetogram,"
        if not filtered_hmi["QUALITY"].mask[idx]:
            # QUALITY limit from https://github.com/mbobra/SMARPs/blob/main/example_gallery/Compare_SMARP_and_SHARP_bitmaps.ipynb
            if np.int32(int(filtered_hmi["QUALITY"][idx], 16)) >= 65536:
                row["filtered"] = True
                filter_reason_column_hmi[row.index] += "QUALITY,"

    filtered_hmi["filter_reason"] = Column([str(reason) for reason in filter_reason_column_hmi], dtype=str)

    # -- Join SRS and HMI
    # and combine their filtered/filter_reason columns
    catalog_hmi = join(
        QTable(srs),
        QTable(filtered_hmi),
        keys="target_time",
        table_names=["catalog", "image"],
    )
    catalog_hmi["filtered"] = _combine_filtered_columns(catalog_hmi["filtered_catalog"], catalog_hmi["filtered_image"])
    catalog_hmi["filter_reason"] = _combine_filter_reason_columns(
        catalog_hmi["filter_reason_catalog"], catalog_hmi["filter_reason_image"]
    )
    catalog_hmi.remove_columns(["filtered_catalog", "filtered_image", "filter_reason_catalog", "filter_reason_image"])
    catalog_hmi.replace_column("path_catalog", [str(pc) for pc in catalog_hmi["path_catalog"]])
    logger.debug(f"Writing {srs_hmi_merged_file}")
    catalog_hmi.write(srs_hmi_merged_file, format="parquet", overwrite=True)

    # ------------
    # HMI - SHARPS
    # ------------

    harp_noaa_map = retrieve_noaa_mapping(
        url="http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt",
        identifier_col_name="record_HARPNUM_arc",
    )

    # 1. merge HMI-SHARPs
    hmi_filtered = QTable(filtered_hmi.copy())
    sharps_filtered = QTable(sharps.copy())
    for colname in sharps_filtered.colnames:
        sharps_filtered.rename_column(colname, colname + "_arc")

    hmi_sharps_table = join(
        hmi_filtered,
        sharps_filtered,
        keys_left="target_time",
        keys_right="target_time_arc",
        table_names=["hmi", "arc"],
    )

    # 2. filter if there is a datetime mismatch between full-disk and active region cutout
    sh_table = hmi_sharps_table.to_pandas().copy()
    dt_mask = sh_table.datetime == sh_table.datetime_arc
    sh_table.loc[~dt_mask, "filtered"] = True
    sh_table.loc[~dt_mask, "filter_reason"] += "datetime,"
    hmi_sharps_table2 = QTable.from_pandas(sh_table)

    # 3. Split the table into rows with and without an associated HARPNUM
    # joining with the harp-to-noaa mapping
    rows_with_harpnum = hmi_sharps_table2[~hmi_sharps_table2["record_HARPNUM_arc"].mask]
    rows_without_harpnum = hmi_sharps_table2[hmi_sharps_table2["record_HARPNUM_arc"].mask]
    joined_table = join(rows_with_harpnum, harp_noaa_map, keys="record_HARPNUM_arc", join_type="left")
    srshmi_table_2 = vstack([joined_table, rows_without_harpnum])
    srshmi_table_2.sort("target_time")

    # 4. Filter rows where the SHARP is associated with > 1 NOAA region
    srshmi_table = srshmi_table_2.to_pandas().copy()
    srshmi_table["NOAANUM"] = srshmi_table["NOAANUM"].fillna(0)
    one_to_one = srshmi_table["NOAANUM"] <= 1  # not filling gets <NA>
    srshmi_table.loc[~one_to_one, "filtered"] = True  # mark any where one_to_one is False
    srshmi_table.loc[~one_to_one, "filter_reason"] += "no_1-1_noaa_harp,"
    srshmi_harpnoaa2 = QTable.from_pandas(srshmi_table)

    # remove this? what the hell?
    srshmi_harpnoaa2["NOAANUM"] = Column([int(val) for val in srshmi_harpnoaa2["NOAANUM"]], dtype=int)

    # 5. Merge HARP and NOAA (from SRS)
    # doesn't include info about the filtered SRS, but I think it's fine as long as Stanford say the mapping exists?
    catalog_hmi_min = catalog_hmi[
        [
            "target_time",
            "id",
            "number",
            "mcintosh_class",
            "magnetic_class",
            "latitude",
            "longitude",
            "longitudinal_extent",
        ]
    ]
    catalog_hmi_min.rename_column("number", "NOAA")
    rows_with_noaanum = srshmi_harpnoaa2[~srshmi_harpnoaa2["NOAA"].mask]
    rows_without_noaanum = srshmi_harpnoaa2[srshmi_harpnoaa2["NOAA"].mask]
    srshmi_harpnoaa3 = join(
        rows_with_noaanum,
        catalog_hmi_min,
        keys=["target_time", "NOAA"],
        table_names=["A", "B"],
    )
    joined_table = vstack([srshmi_harpnoaa3, rows_without_noaanum])
    joined_table.sort("target_time")

    # 6. Filter any(date) where there is a region on disk with > 1 NOAA number per box
    srshmi_harpnoaa4 = filter_grouped_table(joined_table.group_by("processed_path_image"))
    logger.debug(f"Writing {hmi_sharps_merged_file}")
    srshmi_harpnoaa4.write(hmi_sharps_merged_file, format="parquet", overwrite=True)

    # ------------
    # MDI - SMARPS
    # ------------

    mdi_filtered = QTable(filtered_mdi.copy())
    tarp_noaa_map = retrieve_noaa_mapping(
        url="http://jsoc.stanford.edu/doc/data/mdi/all_tarps_with_noaa_ars.txt",
        identifier_col_name="record_TARPNUM_arc",
    )
    smarps_filtered = QTable(smarps.copy())
    for colname in smarps_filtered.colnames:
        smarps_filtered.rename_column(colname, colname + "_arc")

    # 1.
    mdi_smarps_table = join(
        mdi_filtered,
        smarps_filtered,
        keys_left="target_time",
        keys_right="target_time_arc",
        table_names=["mdi", "smarps"],
    )

    # 2.
    sh_table_mdi = mdi_smarps_table.to_pandas().copy()
    dt_mask_mdi = sh_table_mdi.datetime == sh_table_mdi.datetime_arc
    sh_table_mdi.loc[~dt_mask_mdi, "filtered"] = True
    sh_table_mdi.loc[~dt_mask_mdi, "filter_reason"] += "datetime,"
    mdi_smarps_table2 = QTable.from_pandas(sh_table_mdi)

    # 3.
    rows_with_tarpnum = mdi_smarps_table2[~mdi_smarps_table2["record_TARPNUM_arc"].mask]
    rows_without_tarpnum = mdi_smarps_table2[mdi_smarps_table2["record_TARPNUM_arc"].mask]
    joined_table = join(rows_with_tarpnum, tarp_noaa_map, keys="record_TARPNUM_arc", join_type="left")
    srsmdi_table_2 = vstack([joined_table, rows_without_tarpnum])
    srsmdi_table_2.sort("target_time")

    # 4.
    srsmdi_table = srsmdi_table_2.to_pandas().copy()
    srsmdi_table["NOAANUM"] = srsmdi_table["NOAANUM"].fillna(0)
    one_to_on2e = srsmdi_table["NOAANUM"] <= 1  # not filling gets <NA>
    srsmdi_table.loc[~one_to_on2e, "filtered"] = True  # mark any where one_to_one is False
    srsmdi_table.loc[~one_to_on2e, "filter_reason"] += "no_1-1_noaa_harp,"
    srsmdi_tarpnoaa2 = QTable.from_pandas(srsmdi_table)

    # remove this? what the hell?
    srsmdi_tarpnoaa2["NOAANUM"] = Column([int(val) for val in srsmdi_tarpnoaa2["NOAANUM"]], dtype=int)

    # 5.
    catalog_mdi_min = catalog_mdi[
        [
            "target_time",
            "id",
            "number",
            "mcintosh_class",
            "magnetic_class",
            "latitude",
            "longitude",
            "longitudinal_extent",
        ]
    ]
    catalog_mdi_min.rename_column("number", "NOAA")
    rows_with_noaanum_mdi = srsmdi_tarpnoaa2[~srsmdi_tarpnoaa2["NOAA"].mask]
    rows_without_noaanum_mdi = srsmdi_tarpnoaa2[srsmdi_tarpnoaa2["NOAA"].mask]
    srsmdi_tarpnoaa3 = join(
        rows_with_noaanum_mdi,
        catalog_mdi_min,
        keys=["target_time", "NOAA"],
        table_names=["A", "B"],
    )
    joined_table_mdi = vstack([srsmdi_tarpnoaa3, rows_without_noaanum_mdi])
    joined_table_mdi.sort("target_time")

    # 6.
    srsmdi_tarpnoaa4 = filter_grouped_table(joined_table_mdi.group_by("processed_path_image"))
    logger.debug(f"Writing {mdi_smarps_merged_file}")
    srsmdi_tarpnoaa4.write(mdi_smarps_merged_file, format="parquet", overwrite=True)

    return (
        catalog_hmi,
        catalog_mdi,
        srshmi_harpnoaa4,
        srsmdi_tarpnoaa4,
    )


def region_cutouts(config, srs_hmi, srs_mdi):
    logger.info("Generating `Region Cutout` dataset")
    data_root = config["paths"]["data_root"]

    intermediate_files = Path(data_root) / "02_intermediate" / "data" / "region_cutouts"
    data_plot_path_root = Path(data_root) / "04_final" / "data" / "region_cutouts"
    data_plot_path = data_plot_path_root / "fits"
    summary_plot_path = data_plot_path_root / "quicklook"
    classification_file = data_plot_path_root / "region_classification.parq"

    data_plot_path.mkdir(exist_ok=True, parents=True)
    summary_plot_path.mkdir(exist_ok=True, parents=True)
    intermediate_files.mkdir(exist_ok=True, parents=True)

    mdi_cutout = (
        int(config["magnetograms.cutouts"]["x_extent"]) / 4 * u.pix,
        int(config["magnetograms.cutouts"]["y_extent"]) / 4 * u.pix,
    )

    hmi_cutout = (
        int(config["magnetograms.cutouts"]["x_extent"]) * u.pix,
        int(config["magnetograms.cutouts"]["y_extent"]) * u.pix,
    )

    hmi_file = intermediate_files / "hmi_region_cutouts.parq"
    if hmi_file.exists():
        hmi_table = QTable.read(hmi_file)
    else:
        hmi = RegionExtractor(srs_hmi)
        hmi = hmi.extract_regions(
            cutout_size=hmi_cutout,
            data_path=data_plot_path,
            summary_plot_path=summary_plot_path,
            qs_random_attempts=10,
        )
        hmi_table = hmi[2]
        logger.debug(f"writing {hmi_file}")
        hmi_table.write(hmi_file, format="parquet", overwrite=True)

    mdi_file = intermediate_files / "mdi_region_cutouts.parq"
    if mdi_file.exists():
        mdi_table = QTable.read(mdi_file)
    else:
        mdi = RegionExtractor(srs_mdi)
        mdi = mdi.extract_regions(
            cutout_size=mdi_cutout,
            data_path=data_plot_path,
            summary_plot_path=summary_plot_path,
            qs_random_attempts=10,
        )
        mdi_table = mdi[2]
        logger.debug(f"writing {mdi_file}")
        mdi_table.write(mdi_file, format="parquet", overwrite=True)

    if classification_file.exists():
        logger.debug(f"reading {classification_file}")
        ar_classification_hmi_mdi = QTable.read(classification_file)
    else:
        column_subset = [
            "target_time",
            "region_type",
            "number",
            "carrington_longitude",
            "area",
            "mcintosh_class",
            "longitudinal_extent",
            "number_of_sunspots",
            "magnetic_class",
            "latitude",
            "longitude",
            "processed_path_image",
            "top_right_cutout",
            "bottom_left_cutout",
            "path_image_cutout",
            "dim_image_cutout",
            "sum_ondisk_nans",
            "quicklook_path",
            "filtered",
            "filter_reason",
            "QUALITY",
        ]

        ar_classification_hmi_mdi = join(
            QTable(hmi_table[column_subset]),
            QTable(mdi_table[column_subset]),
            join_type="outer",  # keep all columns
            keys=["target_time", "number"],
            table_names=["hmi", "mdi"],
        )

        # trying to combine columns with masks. Can't use as keys in the merge as there are missing values
        # !TODO there must be a better way to do this, but for now, this can suffice.
        ar_classification_hmi_mdi["region_type"] = _combine_columns(
            ar_classification_hmi_mdi["region_type_hmi"], ar_classification_hmi_mdi["region_type_mdi"]
        )

        ar_classification_hmi_mdi["magnetic_class"] = _combine_columns(
            ar_classification_hmi_mdi["magnetic_class_hmi"], ar_classification_hmi_mdi["magnetic_class_mdi"]
        )
        ar_classification_hmi_mdi["carrington_longitude"] = _combine_columns(
            ar_classification_hmi_mdi["carrington_longitude_hmi"], ar_classification_hmi_mdi["carrington_longitude_mdi"]
        )
        ar_classification_hmi_mdi["area"] = _combine_columns(
            ar_classification_hmi_mdi["area_hmi"], ar_classification_hmi_mdi["area_mdi"]
        )
        ar_classification_hmi_mdi["mcintosh_class"] = _combine_columns(
            ar_classification_hmi_mdi["mcintosh_class_hmi"], ar_classification_hmi_mdi["mcintosh_class_mdi"]
        )
        ar_classification_hmi_mdi["longitudinal_extent"] = _combine_columns(
            ar_classification_hmi_mdi["longitudinal_extent_hmi"], ar_classification_hmi_mdi["longitudinal_extent_mdi"]
        )
        ar_classification_hmi_mdi["number_of_sunspots"] = _combine_columns(
            ar_classification_hmi_mdi["number_of_sunspots_hmi"], ar_classification_hmi_mdi["number_of_sunspots_mdi"]
        )

        # Define the fill values for each column
        # !TODO fix this later; shouldn't have to replace the fill values
        columns_with_fill_values = {
            "region_type": "XX",
            "processed_path_image_mdi": "",
            "processed_path_image_hmi": "",
            "quicklook_path_hmi": "",
            "quicklook_path_mdi": "",
            "path_image_cutout_hmi": "",
            "path_image_cutout_mdi": "",
            "filter_reason_hmi": "",
            "filter_reason_mdi": "",
            "magnetic_class": "",
            "mcintosh_class": "",
            "carrington_longitude": np.nan,
            "area": np.nan,
            "longitudinal_extent": np.nan,
            "number_of_sunspots": -1,
            "QUALITY_hmi": "",
            "QUALITY_mdi": "",
        }

        # Update the columns
        ar_classification_hmi_mdi = _update_column_fillvals(ar_classification_hmi_mdi, columns_with_fill_values)

        # List of columns to remove
        columns_to_remove = [
            "region_type_hmi",
            "region_type_mdi",
            "magnetic_class_hmi",
            "magnetic_class_mdi",
            "carrington_longitude_hmi",
            "carrington_longitude_mdi",
            "area_hmi",
            "area_mdi",
            "mcintosh_class_hmi",
            "mcintosh_class_mdi",
            "longitudinal_extent_hmi",
            "longitudinal_extent_mdi",
            "number_of_sunspots_hmi",
            "number_of_sunspots_mdi",
        ]

        # Loop through the list and remove the specified columns
        for col_name in columns_to_remove:
            if col_name in ar_classification_hmi_mdi.colnames:
                ar_classification_hmi_mdi.remove_column(col_name)

        logger.debug(f"writing {classification_file}")
        # problem is this now has filtered values (before we drop all filtered and merge on SRS)
        ar_classification_hmi_mdi.write(classification_file, format="parquet", overwrite=True)

    # filter: hmi/mdi cutout size...
    # one merged catalogue file with both MDI/HMI each task classification and detection
    return ar_classification_hmi_mdi


def _update_column_fillvals(ar_classification_hmi_mdi, columns_with_fill_values):
    for column, fill_value in columns_with_fill_values.items():
        ar_classification_hmi_mdi[column] = MaskedColumn(
            data=ar_classification_hmi_mdi[column].filled(fill_value),
            mask=ar_classification_hmi_mdi[column].mask,
            fill_value=fill_value,
        )

    return ar_classification_hmi_mdi


def _combine_columns(column1, column2):
    """
    given two columns from QTable, attempt to combine as long as values are not different
    """
    combined_column = MaskedColumn(np.empty(len(column1), dtype=column1.dtype))

    for i in range(len(column1)):
        if column1.mask[i] and column2.mask[i]:
            # Both values are masked, set the combined column to masked
            combined_column[i] = np.ma.masked
        elif column1.mask[i]:
            # Use the value from column2 when column1 is masked
            combined_column[i] = column2[i]
        elif column2.mask[i]:
            # Use the value from column1 when column2 is masked
            combined_column[i] = column1[i]
        elif column1[i] == column2[i]:
            # Values are identical, set the combined column to the same value
            combined_column[i] = column1[i]
        else:
            # Values are different and not both masked, raise an error
            raise ValueError(f"Elements at index {i} are different or have different masks: {column1[i]}, {column2[i]}")

    return combined_column


def region_detection(config, hmi_sharps, mdi_smarps):
    logger.info("Generating `Region Detection` dataset")
    data_root = config["paths"]["data_root"]
    region_detection_path_intermediate = Path(data_root) / "02_intermediate" / "data" / "region_detection"
    region_detection_path = Path(data_root) / "04_final" / "data" / "region_detection"

    region_detection_fits_path = region_detection_path / "quicklook"

    region_detection_path_intermediate.mkdir(exist_ok=True, parents=True)
    region_detection_fits_path.mkdir(exist_ok=True, parents=True)

    hmi_ar_det_file = region_detection_path_intermediate / "hmi_region_detection.parq"
    mdi_ar_det_file = region_detection_path_intermediate / "mdi_region_detection.parq"
    reg_det_file = region_detection_path / "region_detection.parq"
    region_detection_path.mkdir(exist_ok=True, parents=True)

    if hmi_ar_det_file.exists():
        logger.debug(f"reading {hmi_ar_det_file}")
        hmi_sharps_detection_table = QTable.read(hmi_ar_det_file)
    else:
        hmidetection = RegionDetection(
            table=hmi_sharps, col_group_path="processed_path_image", col_cutout_path="path_arc"
        )
        hmi_sharps_detection_table, hmi_sharps_detection_bboxes = hmidetection.get_bboxes()
        hmi_sharps_detection_table.write(hmi_ar_det_file, format="parquet", overwrite=True)
        logger.debug(f"writing {hmi_ar_det_file}")

    if mdi_ar_det_file.exists():
        logger.debug(f"reading {mdi_ar_det_file}")
        mdi_smarps_detection_table = QTable.read(mdi_ar_det_file)
    else:
        mdidetection = RegionDetection(
            table=mdi_smarps, col_group_path="processed_path_image", col_cutout_path="path_arc"
        )
        mdi_smarps_detection_table, mdi_smarps_detection_bboxes = mdidetection.get_bboxes()
        # cols_to_remove = [
        #     "top_right_cutout",
        #     "bottom_left_cutout",
        # ]
        # mdi_smarps_detection_table.remove_columns(cols_to_remove)
        mdi_smarps_detection_table.write(mdi_ar_det_file, format="parquet", overwrite=True)
        logger.debug(f"writing {mdi_ar_det_file}")

    if reg_det_file.exists():
        logger.debug(f"reading {reg_det_file}")
        ar_detection = QTable.read(reg_det_file)
    else:
        column_subset = [
            "target_time",
            "datetime",
            "instrument",
            "path",
            "processed_path_image",
            "target_time_arc",
            "datetime_arc",
            "record_T_REC_arc",
            "path_arc",
            "top_right_cutout",
            "bottom_left_cutout",
            "filtered",
            "filter_reason",
            "NOAA",
            "longitudinal_extent",
            "longitude",
            "latitude",
            "id",
            "mcintosh_class",
            "magnetic_class",
        ]

        # subset of columns
        hmi_sharps_detection_table["instrument"] = "HMI"
        hmi_column_subset = column_subset + ["record_HARPNUM_arc"]
        hmi_sharps_detection_table_subset = hmi_sharps_detection_table[hmi_column_subset]

        mdi_smarps_detection_table["instrument"] = "MDI"
        mdi_column_subset = column_subset + ["record_TARPNUM_arc"]
        mdi_smarps_detection_table_subset = mdi_smarps_detection_table[mdi_column_subset]

        # "instrument" column allows each instrument to be studied individually.
        ar_detection = vstack(
            [
                QTable(mdi_smarps_detection_table_subset),
                QTable(hmi_sharps_detection_table_subset),
            ]
        )
        ar_detection.sort("target_time")

        # ar_detection.write(reg_det_file, format="parquet", overwrite=True)
        # logger.debug(f"writing {reg_det_file}")

    return ar_detection


def process_ars(config, catalog):
    logger.info("Processing AR images with config")

    hmi_download_obj, sharps_download_obj = process_hmi(config)
    mdi_download_obj, smarps_download_obj = process_mdi(config)

    # generate:
    #   SRS-HMI: merge SRS with HMI
    #   SRS-MDI: merge SRS with MDI
    #   HMI-SHARPS: merge HMI and SHARPs (null datetime dropped before merge)
    #   MDI-SMARPS: merge HMI and SHARPs (null datetime dropped before merge)
    srs_hmi, srs_mdi, hmi_sharps, mdi_smarps = merge_mag_tables(
        config,
        srs=catalog,
        hmi=hmi_download_obj,
        mdi=mdi_download_obj,
        sharps=sharps_download_obj,
        smarps=smarps_download_obj,
    )

    # extract AR/QS regions from HMI and MDI
    _ = region_cutouts(config, srs_hmi, srs_mdi)

    # bounding box locations of cutouts (in pixel space) on the full-disk images
    # !TODO perform region_detection after the merging of noaa_harp

    ardeten = region_detection(config, hmi_sharps, mdi_smarps)
    # merged_table = merge_noaa_harp(arclass, ardeten)

    merged_table_quicklook = RegionDetection.summary_plots(
        RegionDetectionTable(ardeten),
        Path(config["paths"]["data_root"]) / "04_final" / "data" / "region_detection" / "quicklook",
    )
    # merged_table_quicklook.replace_column("quicklook_path", [str(p) for p in merged_table_quicklook["quicklook_path"]])
    merged_table_quicklook["quicklook_path"] = MaskedColumn(
        data=[str(p) for p in merged_table_quicklook["quicklook_path"]],
        mask=merged_table_quicklook["quicklook_path"].mask,
        fill_value="",
    )

    filename = (
        Path(config["paths"]["data_root"])
        / "04_final"
        / "data"
        / "region_detection"
        / "region_detection_noaa-xarp.parq"
    )

    logger.debug(f"Writing {filename}")
    merged_table_quicklook.write(
        filename,
        format="parquet",
        overwrite=True,
    )


def _combine_filtered_columns(filtered1, filtered2):
    """
    Combine two filtered columns using logical OR.
    """
    return filtered1 | filtered2


def _combine_filter_reason_columns(reason1, reason2):
    """
    Combine two filter_reason columns by concatenating their strings.
    """
    combined_reason = Column(np.empty(len(reason1), dtype=object))

    reason1 = np.array(reason1, dtype=object)
    reason2 = np.array(reason2, dtype=object)

    for i in range(len(reason1)):
        combined_reason[i] = reason1[i] + reason2[i]

    return Column([str(reason) for reason in combined_reason], dtype=str)


def filter_grouped_table(grouped_table: QTable) -> QTable:
    """
    Filter the grouped table by setting 'filtered' to True if any NOAANUM > 1.

    Parameters:
    ----------
    grouped_table (QTable): The grouped table to filter.

    Returns:
    -------
    QTable: The filtered grouped table.
    """
    filter_reason_column = np.array(list(grouped_table["filter_reason"]), dtype=object)
    # if np.unique(filter_reason_column) != "":
    #     raise ValueError("`filter_reason_column` is already populated")

    for date in grouped_table.groups:
        if any(date["NOAANUM"] > 1):
            date["filtered"] = True
            indices = np.where(grouped_table["processed_path_image"] == date["processed_path_image"][0])[0]
            for idx in indices:
                filter_reason_column[idx] += "any(date[NOAANUM] > 1),"

    grouped_table["filter_reason"] = [str(fr) for fr in filter_reason_column]

    return grouped_table


def process_ar_catalogs(config):
    logger.info("Processing AR catalogs with config")
    _, _, _, processed_catalog, _ = process_srs(config)
    return processed_catalog


def main():
    logger.debug("Starting main")
    process_flares(config)
    catalog = process_ar_catalogs(config)
    process_ars(config, catalog)


if __name__ == "__main__":
    main()
    sys.exit()
