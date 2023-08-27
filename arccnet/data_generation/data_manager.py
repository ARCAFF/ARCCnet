from pathlib import Path
from datetime import datetime

import pandas as pd
from parfive import Results
from sunpy.util.parfive_helpers import Downloader

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.catalogs.active_region_catalogs.swpc import SWPCCatalog
from arccnet.data_generation.magnetograms.instruments import (
    HMILOSMagnetogram,
    HMISHARPs,
    MDILOSMagnetogram,
    MDISMARPs,
)
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["DataManager"]


class DataManager:
    """
    Main data management class.

    This class instantiates and handles data acquisition for the individual instruments and cutouts
    """

    def __init__(
        self,
        start_date: datetime = dv.DATA_START_TIME,
        end_date: datetime = dv.DATA_END_TIME,
        merge_tolerance: pd.Timedelta = pd.Timedelta("30m"),
        download_fits: bool = True,
    ):
        """
        Initialize the DataManager.

        Parameters
        ----------
        start_date : datetime, optional
            Start date of data acquisition period. Default is `arccnet.data_generation.utils.default_variables.DATA_START_TIME`.

        end_date : datetime, optional
            End date of data acquisition period. Default is `arccnet.data_generation.utils.default_variables.DATA_END_TIME`.

        merge_tolerance : pd.Timedelta, optional
            Time tolerance for merging operations. Default is pd.Timedelta("30m").

        download_fits : bool, optional
            Whether to download FITS files. Default is True.
        """
        self.start_date = start_date
        self.end_date = end_date

        logger.info(f"Instantiated `DataManager` for {self.start_date} -> {self.end_date}")

        # instantiate classes
        self.swpc = SWPCCatalog()
        # !TODO change this into an iterable
        # Full-disk Magnetograms
        self.hmi = HMILOSMagnetogram()
        self.mdi = MDILOSMagnetogram()
        # Cutouts
        self.sharps = HMISHARPs()
        self.smarps = MDISMARPs()

        # 1. fetch metadata
        logger.info(">> Fetching metadata")
        (
            self.srs,
            self.mdi_keys,
            self.hmi_keys,
            self.sharp_keys,
            self.smarp_keys,
        ) = self.fetch_metadata()
        srs_raw, _ = self.srs
        logger.info(f"\n{srs_raw}")

        # 2. clean metadata
        logger.info(">> Cleaning NOAA SRS Metadata")
        # self.clean_metadata()
        self.srs_clean = self.swpc.clean_catalog()
        logger.info(f"\n{self.srs_clean}")

        # 3. merge metadata sources
        # logger.info(f">> Merging Metadata with tolerance {merge_tolerance}")
        self.merged_df, self.merged_df_dropped_rows = self.merge_hmimdi_metadata(tolerance=merge_tolerance)
        self.hmi_sharps = self.merge_activeregionpatchs(self.hmi_keys, self.sharp_keys)
        self.mdi_smarps = self.merge_activeregionpatchs(self.mdi_keys, self.smarp_keys)

        self.save_df(
            dataframe_list=[self.merged_df, self.hmi_sharps, self.mdi_smarps],
            save_location_list=[
                dv.MAG_INTERMEDIATE_HMIMDI_DATA_CSV,
                dv.MAG_INTERMEDIATE_HMISHARPS_DATA_CSV,
                dv.MAG_INTERMEDIATE_MDISMARPS_DATA_CSV,
            ],
            to_csv=True,
            to_html=False,
        )

        # 4a. check if image data exists
        # !TODO implement this checking if each file that is expected exists.
        # Compile list of URLs to download
        merged_url_columns = [col for col in self.merged_df.columns if col.startswith("url_")]
        hmi_sharps_url_columns = [col for col in self.hmi_sharps.columns if col.startswith("url_")]
        mdi_smarps_url_columns = [col for col in self.mdi_smarps.columns if col.startswith("url_")]

        self.urls_to_download = pd.Series(
            pd.concat(
                [
                    self.merged_df[merged_url_columns],
                    self.hmi_sharps[hmi_sharps_url_columns],
                    self.mdi_smarps[mdi_smarps_url_columns],
                ]
            )
            .stack()
            .dropna()
            .unique()
        )
        if download_fits:
            results = self.fetch_magnetograms(self.urls_to_download)
            logger.info(f"\n{results}")
            # !TODO handle the output... want a csv with the filepaths
            logger.info("Download completed successfully")
        else:
            logger.info(
                "To fetch the magnetograms, use the `.fetch_magnetograms()` method with a list(str) of urls, e.g. the `.urls_to_download` attribute`"
            )

    def save_df(
        self,
        dataframe_list: list(pd.DataFrame),
        save_location_list: list(pd.DataFrame),
        to_csv: bool = True,
        to_html: bool = False,
    ) -> None:
        """
        Save DataFrames to CSV and/or HTML files.

        Parameters
        ----------
        dataframes : list
            List of DataFrames to be saved.

        save_locations : list
            List of file locations to save DataFrames.

        to_csv : bool, optional
            Whether to save as CSV. Default is True.

        to_html : bool, optional
            Whether to save as HTML. Default is True.
        """
        if not to_csv and not to_html:
            logger.info("No action taken. Both `to_csv` and `to_html` are False.")

        for df, loc in zip(dataframe_list, save_location_list):
            if loc.is_file():
                directory_path = loc.parent
                if not directory_path.exists():
                    directory_path.mkdir(parents=True)
            else:
                raise ValueError(f"{loc} is not a directory")

            if to_csv:
                df.to_csv(loc)
            if to_html:
                df.to_html(loc)

    def fetch_metadata(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch and return data from various sources.

        Returns
        -------
        tuple
            Tuple containing data from different sources.
        """
        # download the txt files and create an SRS catalog
        self.swpc.fetch_data(self.start_date, self.end_date)
        srs = self.swpc.create_catalog()

        # HMI & MDI
        # !TODO itereate over children of `BaseMagnetogram`
        hmi_keys = self.hmi.fetch_metadata(self.start_date, self.end_date, batch_frequency=12)
        mdi_keys = self.mdi.fetch_metadata(self.start_date, self.end_date, batch_frequency=12)
        sharp_keys = self.sharps.fetch_metadata(self.start_date, self.end_date, batch_frequency=4)
        smarp_keys = self.smarps.fetch_metadata(self.start_date, self.end_date, batch_frequency=4)

        # logging
        for name, dataframe in {
            "HMI Keys": hmi_keys,
            "SHARP Keys": sharp_keys,
            "MDI Keys": mdi_keys,
            "SMARP Keys": smarp_keys,
        }.items():
            logger.info(
                f"{name}: \n{dataframe[['T_REC','T_OBS','DATE-OBS','DATE__OBS','datetime','magnetogram_fits', 'url']]}"
            )

        return srs, mdi_keys, hmi_keys, sharp_keys, smarp_keys

    def merge_activeregionpatchs(
        self,
        full_disk_data,
        cutout_data,
    ) -> pd.DataFrame:
        """
        Merge active region patch data.

        Parameters
        ----------
        full_disk_data : pd.DataFrame
            Data from full disk observations.

        cutout_data : pd.DataFrame
            Data from active region cutouts.

        Returns
        -------
        pd.DataFrame
            Merged DataFrame of active region patch data.
        """
        # merge srs_clean and hmi
        mag_cols = ["magnetogram_fits", "datetime", "url"]

        # !TODO do a check for certain keys (no duplicates...)
        # extract only the relevant HMI keys, and rename
        # (should probably do this earlier on)
        full_disk_data = full_disk_data[mag_cols]
        full_disk_data = full_disk_data.add_suffix("_hmi")
        full_disk_data_dropna = full_disk_data.dropna().reset_index(drop=True)

        cutout_data = cutout_data[mag_cols]
        cutout_data = cutout_data.add_suffix("_sharps")
        cutout_data_dropna = cutout_data.dropna().reset_index(drop=True)

        # need to figure out if a left merge is what we want...
        merged_df = pd.merge(
            full_disk_data_dropna, cutout_data_dropna, left_on="datetime_hmi", right_on="datetime_sharps"
        )  # no tolerance as should be exact

        return merged_df

    def merge_hmimdi_metadata(
        self,
        tolerance: pd.Timedelta = pd.Timedelta("30m"),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge HMI and MDI metadata.

        Parameters
        ----------
        tolerance : pd.Timedelta, optional
            Time tolerance for merging operations. Default is pd.Timedelta("30m").

        Returns
        -------
        pd.DataFrame
            Merged DataFrame of HMI and MDI metadata.

        pd.DataFrame
            DataFrame of dropped rows during merging.
        """
        # merge srs_clean and hmi
        mag_cols = ["magnetogram_fits", "datetime", "url"]

        # !TODO do a check for certain keys (no duplicates...)
        # extract only the relevant HMI keys, and rename
        # (should probably do this earlier on)
        hmi_keys = self.hmi_keys[mag_cols]
        hmi_keys = hmi_keys.add_suffix("_hmi")
        hmi_keys_dropna = hmi_keys.dropna().reset_index(drop=True)

        # both `pd.DataFrame` must be sorted based on the key !
        merged_df = pd.merge_asof(
            left=self.srs_clean.rename(
                columns={
                    "datetime": "datetime_srs",
                    "filepath": "filepath_srs",
                    "filename": "filename_srs",
                    "loaded_successfully": "loaded_successfully_srs",
                    "catalog_created_on": "catalog_created_on_srs",
                }
            ),
            right=hmi_keys_dropna,
            left_on="datetime_srs",
            right_on="datetime_hmi",
            suffixes=["_srs", "_hmi"],
            tolerance=tolerance,  # HMI is at 720s (12 min) cadence
            direction="nearest",
        )

        mdi_keys = self.mdi_keys[mag_cols]
        mdi_keys = mdi_keys.add_suffix("_mdi")
        mdi_keys_dropna = mdi_keys.dropna().reset_index(drop=True)

        merged_df = pd.merge_asof(
            left=merged_df,
            right=mdi_keys_dropna,
            left_on="datetime_srs",
            right_on="datetime_mdi",
            suffixes=["_srs", "_mdi"],
            tolerance=tolerance,
            direction="nearest",
        )

        # do we want to wait until we merge with MDI before dropping nans?
        dropped_rows = merged_df.copy()
        # self.merged_df = self.merged_df.dropna(subset=["datetime_srs", "datetime_hmi", "datetime_mdi"])
        merged_df = merged_df.dropna(subset=["datetime_srs"])
        merged_df = merged_df.dropna(subset=["datetime_hmi", "datetime_mdi"], how="all")
        dropped_rows = dropped_rows[~dropped_rows.index.isin(merged_df.index)].copy()

        logger.info(f"merged_df: \n{merged_df[['datetime_srs', 'datetime_hmi', 'datetime_mdi']]}")
        logger.info(f"dropped_rows: \n{dropped_rows[['datetime_srs', 'datetime_hmi', 'datetime_mdi']]}")
        logger.info(f"dates dropped: \n{dropped_rows['datetime_srs'].unique()}")

        return merged_df, dropped_rows

    @staticmethod
    def fetch_magnetograms(
        urls: list[str] = None, base_directory_path: Path = Path(dv.MAG_RAW_DATA_DIR), max_retries: int = 5
    ) -> Results:
        """
        Download magnetograms using parfive.

        Parameters
        ----------
        urls : list[str]
            List of URLs to download. Default is None.

        base_directory_path : str or Path, optional
            Base directory path to save downloaded files. Default is `arccnet.data_generation.utils.default_variables.MAG_RAW_DATA_DIR`

        max_retries : int, optional
            Maximum number of download retries. Default is 5.

        Returns
        -------
        results : parfive.Results or None
            parfive results object of the download operation, or None if no URLs provided or invalid format.
        """
        if urls is None or not all(isinstance(url, str) for url in (urls or [])):
            logger.warning("Invalid URLs format. Expected a list of strings.")
            return None

        if not base_directory_path.exists():
            base_directory_path.mkdir(parents=True)

        # Only 1 parallel connection (`max_conn`, `max_splits`)
        # https://docs.sunpy.org/en/stable/_modules/sunpy/net/jsoc/jsoc.html#JSOCClient
        downloader = Downloader(
            max_conn=1,
            progress=True,
            overwrite=False,
            max_splits=1,
        )

        paths = []
        for url in urls:
            filename = url.split("/")[-1]  # Extract the filename from the URL
            paths.append(base_directory_path / filename)

        for aurl, fname in zip(urls, paths):
            downloader.enqueue_file(aurl, filename=fname, max_splits=1)

        results = downloader.download()

        if len(results.errors) != 0:
            logger.warning(f"results.errors: {results.errors}")
            # attempt a retry
            retry_count = 0
            while len(results.errors) != 0 and retry_count < max_retries:
                logger.info("retrying...")
                downloader.retry(results)
                retry_count += 1
            if len(results.errors) != 0:
                logger.error("Failed after maximum retries.")
            else:
                logger.info("Errors resolved after retry.")
        else:
            logger.info("No errors reported by parfive")

        return results


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")

    try:
        data_manager = DataManager(dv.DATA_START_TIME, dv.DATA_END_TIME)
    except Exception:
        logger.exception("An error occurred during execution:", exc_info=True)
