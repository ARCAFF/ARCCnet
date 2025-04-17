from __future__ import annotations  # noqa

import logging
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import parfive
from sunpy.io.special import srs
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net.attr import AttrOr
from sunpy.net.dataretriever import QueryResponse, SRSClient
from sunpy.net.fido_factory import UnifiedResponse
from sunpy.physics.differential_rotation import diff_rot

import astropy.units as u
from astropy.table import MaskedColumn, QTable, join, vstack
from astropy.time import Time

from arccnet.catalogs.active_regions import HALE_CLASSES, MCINTOSH_CLASSES
from arccnet.utils.logging import get_logger

logger = get_logger(__name__, logging.DEBUG)


__all__ = ["Query", "Result", "ClassificationCatalog", "SWPCCatalog", "filter_srs"]


class Query(QTable):
    r"""
    Query object defines both the query and results.

    The query is defined by a row with 'start_time', 'end_time' and 'url'. 'Url' is `MaskedColum` and where the
    mask is `True` can be interpreted as missing data.

    Notes
    -----
    Under the hood uses QTable and Masked columns to define if an expected result is present or missing

    """

    required_column_types = {"start_time": Time, "end_time": Time, "url": str}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not set(self.colnames).issuperset(set(self.required_column_types.keys())):
            raise ValueError(
                f"{self.__class__.__name__} must contain {list(self.required_column_types.keys())} columns."
            )
        self["url"] = MaskedColumn(self["url"])

    @property
    def is_empty(self) -> bool:
        r"""Empty query"""
        return np.all(self["url"].mask == np.full(len(self), True))

    @property
    def missing(self) -> Query:
        r"""Rows which are missing."""
        return self[self["url"].mask == True]  # noqa

    @classmethod
    def create_empty(cls, start, end) -> Query:
        r"""
        Create an 'empty' Query.

        Parameters
        ----------
        start : `str`, `datetime` or `Time`
            Start time, any format supported by `astropy.time.Time`
        end : `str`, `datetime` or `Time`
            End time, any format supported by `astropy.time.Time`

        Returns
        -------
        Query
            An empty Query
        """
        start = Time(start)
        end = Time(end)
        start_pydate = start.to_datetime()
        end_pydate = end.to_datetime()
        dt = end_pydate - start_pydate
        start_times = Time([start_pydate + timedelta(days=days) for days in range(dt.days + 1)])

        if not start_times[-1].isclose(end):
            raise ValueError(f"Expected end time {start_times[-1]} does not match supplied end time: {end}")
        end_times = start_times + 1 * u.day - 1 * u.ms
        urls = MaskedColumn(data=[""] * len(start_times), mask=np.full(len(start_times), True))
        empty_query = cls(data=[start_times, end_times, urls], names=["start_time", "end_time", "url"])
        return empty_query


class Result(QTable):
    r"""
    Result object defines both the result and download status.

    The value of the 'path' is used to encode if the corresponding file was downloaded or not.

    Notes
    -----
    Under the hood uses QTable and Masked columns to define if a file was downloaded or not

    """

    required_column_types = {"start_time": Time, "end_time": Time, "url": str, "path": str}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not set(self.colnames).issuperset(set(self.required_column_types.keys())):
            raise ValueError(
                f"{self.__class__.__name__} must contain {list(self.required_column_types.keys())} columns"
            )


class ClassificationCatalog(QTable):
    r"""
    Active region classification catalog.
    """

    required_column_types = {
        "target_time": Time,
        "longitude": u.deg,
        "latitude": u.deg,
        "magnetic_class": str,
        "mcintosh_class": str,
        "url": str,
        "path": str,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not set(self.colnames).issuperset(set(self.required_column_types.keys())):
            raise ValueError(
                f"{self.__class__.__name__} must contain {list(self.required_column_types.keys())} columns"
            )
        self["url"] = MaskedColumn(self["url"])
        self["path"] = MaskedColumn(self["path"])

    @classmethod
    def read(cls, *args, **kwargs) -> ClassificationCatalog:
        r"""
        Read the catalog from a file.
        """
        table = QTable.read(*args, **kwargs)
        paths = [Path(p) for p in table["path"]]
        table.replace_column("path", paths)
        return cls(table)

    def write(self, *args, **kwargs) -> None:
        r"""
        Write the catalog to a file.
        """
        paths = [str(p) for p in self["path"]]
        self["path"] = paths
        return super(QTable, self).write(*args, **kwargs)


class SWPCCatalog:
    r"""
    SWPC catalog
    """

    def search(self, query: Query, overwrite: bool = False, retry_missing: bool = False) -> Query:
        r"""
        Search the SWPC catalog based on the query.

        Parameters
        ----------
        query : `Query`
            Search query.
        overwrite : `boolean`
            Set true to overwrite previous query results.
        retry_missing : `boolean`
            Try to obtain missing results based on a previous query.

        Returns
        -------
        Query
            Search results
        """
        logger.debug("Entering search")
        if overwrite is True and retry_missing is True:
            raise ValueError("Keywords overwrite and retry_missing can not both be True")

        times = None
        result = query.copy()

        if not query.is_empty and overwrite is False and retry_missing is False:
            logger.debug("No search required returning input")
            return result

        if overwrite is True or query.is_empty:
            logger.debug(f"Full search (overwrite = {overwrite})")
            times = a.Time(query["start_time"].min(), query["end_time"].max())

        if retry_missing is True:
            logger.debug(f"Retrying missing (retry_missing = {retry_missing})")
            query = QTable(query)
            missing = query["url"].mask == True  # noqa
            times = [a.Time(row["start_time"], row["end_time"]) for row in query[missing]]
            new_query = Fido.search(AttrOr(times), a.Instrument.soon)
            stacked = vstack(new_query["srs"])
            for row in stacked:
                match = query["start_time"].jd == row["start_time"].jd
                query["url"][match] = row["url"]  # astropy oddity need to set with columns then index
            return Query(query)

        if times is not None:
            query = Fido.search(times, a.Instrument.soon)
            stacked = query["srs"]
            stacked.rename_columns(stacked.colnames, [n.lower().replace(" ", "_") for n in stacked.colnames])
            stacked["temp_t"] = stacked["start_time"].jd
            result["temp_t"] = result["start_time"].jd
            # join doesn't seem to like custom table objects, also some issues joining on time objects due to rounding
            # so have to convert to an actual string/number for comparison
            result = join(QTable(result), stacked, join_type="left", keys="temp_t", table_names=["query", "result"])
            result.rename_column("url_result", "url")
            result.remove_columns(["temp_t", "url_query"])
            result.remove_columns([name for name in result.colnames if name.endswith("_result")])
            result.rename_columns(result.colnames, [n.removesuffix("_query") for n in result.colnames])
            result = Query(result.columns)

        logger.debug("Exiting search")
        return result

    def download(
        self, query: Query | Result, path=None, overwrite=False, retry_missing=False, progress=False
    ) -> Result:
        r"""
        Download query results.

        Parameters
        ----------
        query : `Query` or `Result`
            Query to download data for.
        path : `str`
            Path to download data to.
        overwrite : `bool`
            Overwrite existing data
        retry_missing : `bool`
            Try and re-download missing data
        progress : `bool`
            Display progress bar

        Returns
        -------
        Result
            Downloaded data
        """
        logger.debug("Entering download")
        new_query = None
        results = query.copy()

        if isinstance(query, Result) and overwrite is False and retry_missing is False:
            logger.debug("Nothing to download returning input")
            return results

        if overwrite is True or "path" not in query.colnames:
            logger.debug(f"Full download with overwrite: (overwrite = {overwrite})")
            new_query = QTable(query)

        if retry_missing is True:
            logger.debug(f"Downloading with retry_missing: (retry_missing = {retry_missing})")
            missing = query["path"] == ""
            new_query = query[missing]

        if new_query is not None:
            logger.debug("Downloading ...")
            downloads = self._download(new_query[~new_query["url"].mask], overwrite, path, progress=progress)
            results = self._match(results, downloads)

        logger.debug("Exiting download")
        return results

    def create_catalog(self, results: Result) -> ClassificationCatalog:
        r"""
        Create a catalog from downloaded files.

        Parameters
        ----------
        results : `Result`
            Downloaded file results.

        Returns
        -------
        ClassificationCatalog
            Classification catalog
        """
        logger.debug("Entering create catalog")
        srs_data = []
        time_now = Time.now()

        if isinstance(results, ClassificationCatalog):
            logger.debug("Catalog already creating returning input")
            return results

        downloaded_filepaths = [Path(filepath) for filepath in results["path"].filled("")]  # if filepath != ""

        with ThreadPoolExecutor() as thread_pool:
            parsed_srs = thread_pool.map(self._parse_srs, downloaded_filepaths, timeout=10)

        for (filepath, srs_table), info in zip(parsed_srs, results):
            file_info = QTable(
                [
                    {
                        "target_time": info["start_time"],
                        "path": filepath,
                        "filename": filepath.name,
                        "url": info["url"].astype(str),
                        "loaded_successfully": False,
                        "catalog_created_on": time_now,
                    }
                ]
            )

            if srs_table is not False:
                file_info["loaded_successfully"] = True

                if len(srs_table) == 0:
                    srs_data.append(file_info)
                else:
                    srs_table.add_columns(file_info.columns)
                    srs_data.append(srs_table)

            elif srs_table is False:
                srs_data.append(file_info)

        catalog = vstack(srs_data, metadata_conflicts="silent")
        catalog = ClassificationCatalog(catalog)
        logger.debug("Finished creating catalog")
        return catalog

    def _download(self, query: Query, overwrite: bool, path: str, progress=False) -> UnifiedResponse:
        r"""
        Download query results.

        Parameters
        ----------
        query : `Query`
            Query to download.
        overwrite : `bool`
            Overwrite existing files.
        path : `str`
            Path to save files to see `Fido` for details

        Returns
        -------
        UnifiedResponse
            The download results response
        """
        orig_get_ftp = parfive.downloader.get_ftp_size

        async def dummy_get_ftp(*args, **kwargs):
            return 0

        parfive.downloader.get_ftp_size = dummy_get_ftp

        query = QueryResponse(query)
        query.client = SRSClient()

        results = Fido.fetch(
            query,
            path=path,
            progress=progress,
            overwrite=overwrite,
        )
        # Replace original method - not sure if this is needed
        parfive.downloader.get_ftp_size = orig_get_ftp

        return results

    def _match(self, results: Result, downloads: UnifiedResponse) -> Result:
        r"""
        Match new download to existing results entries.

        Filename are parsed to extract time which is then match to times in results.

        Parameters
        ----------
        results : `Result`
            Existing results
        downloads : `UnifiedResponse`
            List of downloaded files

        Returns
        -------
        Result
            The updated results
        """
        results = QTable(results)
        logger.info("Downloads to query or new data")
        downloads = np.array(downloads)
        results.sort("start_time")
        time_edges_jd = np.unique(
            [results[~results["end_time"].mask]["end_time"].jd, results[~results["start_time"].mask]["start_time"].jd]
        )
        file_times = Time.strptime([Path(filename).name for filename in downloads], "%Y%m%dSRS.txt")
        orig_indices = file_times.argsort()
        filetimes = file_times.sort()
        matched_indices = (np.digitize(filetimes.jd, time_edges_jd) - 1) // 2

        # Table weirdness
        if "path" in results.colnames:
            results.remove_column("path")
        results["path"] = None
        results["path"][matched_indices] = downloads[orig_indices]
        results["path"][results["path"] == None] = ""  # noqa: E711
        tmp_path = MaskedColumn(results["path"].data.tolist())
        tmp_path.mask = results["url"].mask
        results.replace_column("path", tmp_path)
        results = Result(results.columns)
        return results

    @staticmethod
    def _parse_srs(filepath: str) -> tuple[str, bool | QTable]:
        expected_colnames = [
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
        ]
        try:
            srs_table = srs.read_srs(filepath)
            if srs_table.colnames != expected_colnames:
                logger.error(
                    f"Error parsing file parsed columns: {srs_table.colnames} "
                    f"do not match expected columns: {expected_colnames}."
                )
                srs_table = False
            else:
                srs_table.rename_columns(
                    expected_colnames, [name.lower().replace(" ", "_") for name in expected_colnames]
                )
                srs_table.rename_columns(["z", "mag_type"], ["mcintosh_class", "magnetic_class"])

            # Seems can't serialise SkyCoords with different frames or something
            # pos = [SkyCoord(row['longitude'], row['latitude'],  frame=HeliographicStonyhurst,
            #                 obstime=Time(row.meta['issued']) - 30 * u.min) for row in srs_table]
            # srs_table['position'] = pos

        except Exception:
            logger.warning(f"Error reading file {str(filepath)}", exc_info=True)
            srs_table = False

        return filepath, srs_table


def filter_srs(
    catalog,
    lat_limit: u.Quantity[u.degree] = 60 * u.degree,
    lon_limit: u.Quantity[u.degree] = 85 * u.degree,
    lat_rate_limit: u.Quantity[u.degree / u.day] = 5 * u.deg,
    lon_rate_limit: u.Quantity[u.deg / u.day] = 5 * u.deg,
) -> ClassificationCatalog:
    r"""
    Filter SRS for unphysical position or positions rate of change.

    Parameters
    ----------
    catalog : `pandas.Dataframe`
        SRS catalog
    lat_limit: `astropy.units.Quantity`
        Latitude limit filters `abs(Lat > lat_limit)`
    lon_limit: `astropy.units.Quantity`
        Longitude limit filters `abs(Lon > lon_limit)`
    lat_rate_limit : `astropy.units.Quantity`
        Latitude rate limit filter based on rate of change of lat
    lon_rate_limit : `astropy.units.Quantity`
        Longitude rate limit filter based on rate of change of lat
    Returns
    -------
    `pandas.DataFrame`
        The filtered srs catalog
    """
    catalog_df = catalog.to_pandas()
    active_regions_df = catalog_df.copy()
    active_regions_df.magnetic_class = active_regions_df.magnetic_class.str.title()
    active_regions_df.mcintosh_class = active_regions_df.mcintosh_class.str.title()
    active_regions_df["filtered"] = False
    active_regions_df["filter_reason"] = ""

    load_success = active_regions_df.loaded_successfully == True  # noqa
    active_regions_df.loc[~load_success, "filtered"] = True
    active_regions_df.loc[~load_success, "filter_reason"] += "load_unsuccessful,"

    ar_plage_mask = np.isin(catalog_df.id, ["I", "IA"])
    active_regions_df.loc[~ar_plage_mask, "filtered"] = True
    active_regions_df.loc[~ar_plage_mask, "filter_reason"] += "not_ar,"

    ar_mask = catalog_df.id == "I"
    valid_classes = {"magnetic_class": HALE_CLASSES, "mcintosh_class": MCINTOSH_CLASSES}
    for col, vals in valid_classes.items():
        result = active_regions_df[col].isin(vals)
        mask = (active_regions_df.id == "I") & ~result
        active_regions_df.loc[mask, "filtered"] = True
        active_regions_df.loc[mask, "filter_reason"] += f"invalid_{col},"

    bad_latitudes = active_regions_df.latitude.abs() > lat_limit
    active_regions_df.loc[bad_latitudes, "filtered"] = True
    active_regions_df.loc[bad_latitudes, "filter_reason"] += "bad_lat,"

    bad_longitudes = active_regions_df.longitude.abs() > lon_limit
    active_regions_df.loc[bad_longitudes, "filtered"] = True
    active_regions_df.loc[bad_longitudes, "filter_reason"] += "bad_lon,"

    for number, group in active_regions_df[ar_mask].groupby("number"):
        lat = group.latitude.values * u.deg
        mean_lat = np.mean(lat)

        # Without further checks hard to know which data point is incorrect so drop
        bad_lat_rates = np.abs(lat - mean_lat) > lat_rate_limit
        if np.any(bad_lat_rates):
            filtered = group.index.values[bad_lat_rates]
            if len(group) < 3:  # too small so drop all
                filtered = group.index.values
            active_regions_df.loc[filtered, "filtered"] = True
            active_regions_df.loc[filtered, "filter_reason"] += "bad_lat_rate,"

        # account for differential rotation
        dt = (group.target_time - group.target_time.min()).dt.days << u.day
        expected_diff_rot = diff_rot(dt, mean_lat, frame_time="synodic")
        corrected_lon = group.longitude.values * u.deg - expected_diff_rot
        mean_lon = np.mean(corrected_lon)

        bad_lon_rates = np.abs(corrected_lon - mean_lon) > lon_rate_limit
        if np.any(bad_lon_rates):
            filtered = group.index.values[bad_lat_rates]
            if len(group) < 3:  # too small so drop all
                filtered = group.index.values
            active_regions_df.loc[filtered, "filtered"] = True
            active_regions_df.loc[filtered, "filter_reason"] += "bad_lon_rate,"

    out_catalog = QTable.from_pandas(active_regions_df)
    out_catalog["longitude"].unit = u.deg
    out_catalog["latitude"].unit = u.deg
    out_catalog["number"] = np.array(out_catalog["number"], dtype=int)
    out_catalog["number_of_sunspots"] = np.array(out_catalog["number_of_sunspots"], dtype=int)

    return ClassificationCatalog(out_catalog)
