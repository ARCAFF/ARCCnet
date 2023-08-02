import random
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import astropy.units as u
from astropy.coordinates import SkyCoord

import arccnet.data_generation.utils.default_variables as dv
import sunpy.map
from arccnet.data_generation.magnetograms.instruments import HMISHARPs
from arccnet.data_generation.utils.data_logger import logger
from sunpy.util.parfive_helpers import Downloader

__all__ = ["MagnetogramProcessor", "ARExtractor", "QSExtractor", "ARDetection"]


class MagnetogramProcessor:
    """
    Process Magnetograms
    """

    def __init__(self) -> None:
        filename = Path(dv.MAG_INTERMEDIATE_DATA_CSV)

        if filename.exists():
            self.loaded_data = pd.read_csv(filename)
            file_list = list(self.loaded_data.url_hmi.dropna().unique()) + list(
                self.loaded_data.url_mdi.dropna().unique()
            )
            paths = []
            for url in file_list:
                filename = Path(url).name  # Extract the filename from the URL
                file_path = Path(dv.MAG_RAW_DATA_DIR) / filename  # Join the path and filename
                paths.append(file_path)
        else:
            raise FileNotFoundError(f"{filename} does not exist. Try running `DataManager` first.")

        base_directory_path = Path(dv.MAG_INTERMEDIATE_DATA_DIR)
        if not base_directory_path.exists():
            base_directory_path.mkdir(parents=True)

        _ = self._process_data(paths)

    def _process_data(self, files) -> None:
        # !TODO find a good way to deal with the paths

        for file in tqdm(files, desc="Processing data", unit="file"):
            processed_data = self._process_datum(file)
            # !TODO probably append the name with something
            processed_data.save(Path(dv.MAG_INTERMEDIATE_DATA_DIR) / file.name, overwrite=True)

    def _process_datum(self, file) -> None:
        # 1. Load & Rotate
        map = sunpy.map.Map(file)
        r_map = self._rotate_datum(map)

        # 2. set data off-disk to 0 (np.nan would be ideal, but deep learning)
        r_map.data[~sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(r_map))] = 0
        # !TODO understand why this isn't working with MDI (leaves a white ring around the disk)

        # !TODO
        # 3. normalise radius to fixed value
        # 4. project to a certain location in space

        return r_map

    def _rotate_datum(self, amap: sunpy.map.Map) -> sunpy.map.Map:
        """
        rotate a list of maps according to metadata

        e.g. before rotation a HMI map may have: `crota2 = 180.082565`
        """

        if "crota2" not in amap.meta:
            logger.info(f"The MetaDict for {amap.meta['t_rec']} does not have 'crota2' key.")

        rmap = amap.rotate()

        return rmap


class ARExtractor:
    def __init__(self) -> None:
        filename = Path(dv.MAG_INTERMEDIATE_DATA_CSV)

        if filename.exists():
            self.loaded_data = pd.read_csv(filename)

        columns_to_update = ["url_hmi", "url_mdi"]
        new_columns = ["processed_hmi", "processed_mdi"]

        # !TODO replace with default_variables.py
        dv_base_path = Path(dv.MAG_INTERMEDIATE_DATA_DIR)

        dv_process_fits_path = Path(dv.MAG_PROCESSED_FITS_DIR)
        if not dv_process_fits_path.exists():
            dv_process_fits_path.mkdir(parents=True)

        dv_summary_plots_path = Path(dv.MAG_PROCESSED_SUMMARYPLOTS_DIR)
        if not dv_summary_plots_path.exists():
            dv_summary_plots_path.mkdir(parents=True)

        # Iterate through the columns and update the paths
        # !TODO deal with earlier on in the codebase
        for old_column, new_column in zip(columns_to_update, new_columns):
            self.loaded_data[new_column] = self.loaded_data[old_column].map(
                lambda x: dv_base_path / Path(x).name if pd.notna(x) else x
            )

        # set empty list of cutout for hmi
        cutout_list_hmi = []
        cutout_hmi_dim = []
        bls = []
        trs = []

        self.loaded_subset = self.loaded_data[
            [
                "Latitude",
                "Longitude",
                "Number",
                "Area",
                "Z",
                "Mag Type",
                "processed_hmi",
                "datetime_hmi",
                "datetime_srs",
            ]
        ].copy()

        logger.info(self.loaded_subset)

        # drop rows with NaN (so drop none with HMI)
        # !TODO go through HMI and MDI separately
        self.loaded_subset.dropna(inplace=True)

        # group by SRS files
        grouped_data = self.loaded_subset.groupby("datetime_srs")

        for time_srs, group in grouped_data:
            summary_info = []

            my_hmi_map = sunpy.map.Map(group.processed_hmi.unique()[0])  # take the first hmi
            time_hmi = group.datetime_hmi.unique()[0]

            logger.info(
                f"the srs time is {time_srs}, and the hmi time is {time_hmi}. The size of the group is len(group) {len(group)}"
            )

            for _, row in group.iterrows():
                # extract the lat/long and NOAA AR Number (for saving)
                lat, lng, numbr = row[["Latitude", "Longitude", "Number"]]
                logger.info(f" >>> {lat}, {lng}, {numbr}")

                _cd = SkyCoord(
                    lng * u.deg,
                    lat * u.deg,
                    obstime=time_hmi,
                    frame=sunpy.coordinates.frames.HeliographicStonyhurst,
                )

                transformed = _cd.transform_to(my_hmi_map.coordinate_frame)

                # Performed in arcseconds
                #
                # tr_x = transformed.helioprojective.Tx + 200 * u.arcsec
                # tr_y = transformed.helioprojective.Ty + 100 * u.arcsec
                # bl_x = transformed.helioprojective.Tx - 200 * u.arcsec
                # bl_y = transformed.helioprojective.Ty - 100 * u.arcsec
                # top_right = SkyCoord(tr_x, tr_y, frame=my_hmi_map.coordinate_frame)
                # bottom_left = SkyCoord(bl_x, bl_y, frame=my_hmi_map.coordinate_frame)
                # my_hmi_submap = my_hmi_map.susbmap(bottom_left, top_right=top_right)

                # Perform in pixel coordinates
                ar_centre = transformed.to_pixel(my_hmi_map.wcs)
                top_right = [ar_centre[0] + (dv.X_EXTENT - 1) / 2, ar_centre[1] + (dv.Y_EXTENT - 1) / 2] * u.pix
                bottom_left = [ar_centre[0] - (dv.X_EXTENT - 1) / 2, ar_centre[1] - (dv.Y_EXTENT - 1) / 2] * u.pix
                my_hmi_submap = my_hmi_map.submap(bottom_left, top_right=top_right)

                summary_info.append([top_right, bottom_left, numbr, my_hmi_submap.data.shape, ar_centre])

                # the y range should always be the same.... x may change
                # assert my_hmi_submap.data.shape[0] == dv.Y_EXTENT

                # !TODO see
                # https://gitlab.com/frontierdevelopmentlab/living-with-our-star/super-resolution-maps-of-solar-magnetic-field/-/blob/master/source/prep.py?ref_type=heads
                # logger.info(f"saving {dv_process_fits_path}/{time_srs}_{numbr}.fits")
                my_hmi_submap.save(dv_process_fits_path / f"{time_srs}_{numbr}.fits", overwrite=True)

                cutout_list_hmi.append(dv_process_fits_path / f"{time_srs}_{numbr}.fits")
                cutout_hmi_dim.append(my_hmi_submap.data.shape)

                bls.append(bottom_left)
                trs.append(top_right)

                del my_hmi_submap  # delete the submap

            # Plotting and saving
            # !TODO move to a new function
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(projection=my_hmi_map)
            my_hmi_map.plot_settings["norm"].vmin = -1500
            my_hmi_map.plot_settings["norm"].vmax = 1500
            my_hmi_map.plot(axes=ax, cmap="hmimag")

            for _, (tr, bl, num, shape, arc) in enumerate(summary_info):
                if shape == (dv.Y_EXTENT, dv.X_EXTENT):
                    rectangle_cr = "red"
                    rectangle_ls = "-"
                else:
                    rectangle_cr = "black"
                    rectangle_ls = "-."

                my_hmi_map.draw_quadrangle(
                    bl,
                    axes=ax,
                    top_right=tr,
                    edgecolor=rectangle_cr,
                    linestyle=rectangle_ls,
                    linewidth=1,
                    label=str(num),
                )

                ax.text(
                    arc[0],
                    arc[1] + (dv.Y_EXTENT / 2) + 5,
                    num,
                    **{"size": "x-small", "color": "black", "ha": "center"},
                )

            plt.savefig(dv_summary_plots_path / f"{time_srs}.png", dpi=300)

        self.loaded_subset.loc[:, "hmi_cutout"] = cutout_list_hmi
        self.loaded_subset.loc[:, "hmi_cutout_dim"] = cutout_hmi_dim
        self.loaded_subset.loc[:, "bottom_left"] = bls
        self.loaded_subset.loc[:, "top_right"] = trs

        self.loaded_subset.to_csv(Path(dv.MAG_PROCESSED_DIR) / "processed.csv")

        # clean data
        dv_final_path = Path(dv.DATA_DIR_FINAL)
        if not dv_final_path.exists():
            dv_final_path.mkdir(parents=True)

        # clean data
        # 1. Ensure the data is (400, 800)
        self.loaded_subset_cleaned = self.loaded_subset[
            self.loaded_subset["hmi_cutout_dim"] == (dv.Y_EXTENT, dv.X_EXTENT)
        ]
        # Drop NaN, Reset Index, Save to `arcutout_clean.csv`
        self.loaded_subset_cleaned = self.loaded_subset_cleaned.dropna()
        self.loaded_subset_cleaned = self.loaded_subset_cleaned.reset_index()
        self.loaded_subset_cleaned.to_csv(Path(dv.DATA_DIR_FINAL) / "arcutout_clean.csv")  # need to reset index


class QSExtractor:
    def __init__(self, num_attempts=50):
        filename = Path(dv.MAG_PROCESSED_DIR) / "processed.csv"

        if filename.exists():
            self.loaded_data = pd.read_csv(filename)

        self.loaded_data["datetime_srs"] = pd.to_datetime(self.loaded_data["datetime_srs"])
        grouped_data = self.loaded_data.groupby("datetime_srs")

        qs_df = pd.DataFrame(columns=["datetime_srs", "datetime_hmi", "qs_fits"])

        all_qs = []
        for time_srs, group in grouped_data:
            my_hmi_map = sunpy.map.Map(group.processed_hmi.unique()[0])  # take the first hmi
            time_hmi = group.datetime_hmi.unique()[0]

            vals = []
            for _, row in group.iterrows():
                # extract the lat/long and NOAA AR Number (for saving)
                lat, lng, numbr = row[["Latitude", "Longitude", "Number"]]
                logger.info(f" >>> {lat}, {lng}, {numbr}")

                ar_centre = (
                    SkyCoord(
                        lng * u.deg,
                        lat * u.deg,
                        obstime=time_hmi,
                        frame=sunpy.coordinates.frames.HeliographicStonyhurst,
                    )
                    .transform_to(my_hmi_map.coordinate_frame)
                    .to_pixel(my_hmi_map.wcs)
                )

                # all active region centres
                vals.append(ar_centre)

            logger.info(
                f"the srs time is {time_srs}, and the hmi time is {time_hmi}. The size of the group is len(group) {len(group)}"
            )

            logger.info(group)

            qs_reg = []
            for i in range(0, num_attempts):
                # create random location
                rand_1 = random.uniform(-1000, 1000) * u.arcsec
                rand_2 = random.uniform(-500, 500) * u.arcsec

                # convert to pixel coordinates
                _cd = SkyCoord(
                    rand_1,
                    rand_2,
                    frame=my_hmi_map.coordinate_frame,
                ).to_pixel(my_hmi_map.wcs)

                # check _cd is far enough from other vals
                tt = list(
                    map(
                        lambda v: self.is_point_far_from_point(
                            _cd[0], _cd[1], v[0], v[1], dv.X_EXTENT * 1.2, dv.Y_EXTENT * 1.2
                        ),
                        vals,
                    )
                )

                if all(tt):  # len of tt?
                    top_right = [_cd[0] + (dv.X_EXTENT - 1) / 2, _cd[1] + (dv.Y_EXTENT - 1) / 2] * u.pix
                    bottom_left = [_cd[0] - (dv.X_EXTENT - 1) / 2, _cd[1] - (dv.Y_EXTENT - 1) / 2] * u.pix
                    my_hmi_submap = my_hmi_map.submap(bottom_left, top_right=top_right)

                    fig = plt.figure(figsize=(5, 5))
                    ax = fig.add_subplot(projection=my_hmi_submap)
                    my_hmi_submap.plot_settings["norm"].vmin = -1500
                    my_hmi_submap.plot_settings["norm"].vmax = 1500
                    my_hmi_submap.plot(axes=ax, cmap="hmimag")

                    fn = (
                        Path(  # need to make this manually
                            "/Users/pjwright/Documents/work/ARCCnet/data/03_processed/mag/qs_fits"
                        )
                        / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_QS_{i}.fits"
                    )

                    my_hmi_submap.save(
                        fn,
                        overwrite=True,
                    )

                    qs_temp = pd.DataFrame(
                        {
                            "datetime_hmi": group.datetime_hmi.unique()[0],
                            "datetime_srs": group.datetime_srs.unique()[0],
                            "qs_fits": str(fn),
                        },
                        index=[0],
                    )

                    # print(qs_df)
                    # print(qs_temp)
                    # test this
                    # qs_df = qs_df.append(
                    #     pd.DataFrame({"datetime_srs": group.datetime_srs.unique()[0], "qs_fits": str(fn)}, index=[0]),
                    #     ignore_index=True,
                    # )

                    qs_df = pd.concat([qs_df, qs_temp], ignore_index=True)

                    del my_hmi_submap
                    vals.append(_cd)
                    qs_reg.append(_cd)

                print("vals", vals)

            all_qs.append(qs_reg)

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(projection=my_hmi_map)
            my_hmi_map.plot_settings["norm"].vmin = -1500
            my_hmi_map.plot_settings["norm"].vmax = 1500
            my_hmi_map.plot(axes=ax, cmap="hmimag")

            for value in vals:
                top_right = [value[0] + (dv.X_EXTENT - 1) / 2, value[1] + (dv.Y_EXTENT - 1) / 2] * u.pix
                bottom_left = [value[0] - (dv.X_EXTENT - 1) / 2, value[1] - (dv.Y_EXTENT - 1) / 2] * u.pix
                my_hmi_submap = my_hmi_map.submap(bottom_left, top_right=top_right)

                if my_hmi_submap.data.shape == (dv.Y_EXTENT, dv.X_EXTENT):
                    rectangle_cr = "red"
                    rectangle_ls = "-"
                else:
                    rectangle_cr = "black"
                    rectangle_ls = "-."

                if value in qs_reg:
                    rectangle_cr = "blue"

                my_hmi_map.draw_quadrangle(
                    bottom_left,
                    axes=ax,
                    top_right=top_right,
                    edgecolor=rectangle_cr,
                    linestyle=rectangle_ls,
                    linewidth=1,
                )

            plt.savefig(
                Path("/Users/pjwright/Documents/work/ARCCnet/data/03_processed/mag/qs_summary_plots")
                / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_QS.png",
                dpi=300,
            )

            print(qs_df)
            print(len(all_qs))

            qs_df.to_csv("/Users/pjwright/Documents/work/ARCCnet/data/03_processed/mag/qs_fits.csv")

    def is_point_far_from_point(self, x, y, x1, y1, threshold_x, threshold_y):
        # test this code
        return abs(x - x1) > threshold_x or abs(y - y1) > threshold_y


class ARDetection:
    def __init__(self):
        filename = Path(dv.MAG_INTERMEDIATE_DATA_CSV)

        if filename.exists():
            self.loaded_data = pd.read_csv(filename)

        columns_to_update = ["url_hmi", "url_mdi"]
        new_columns = ["processed_hmi", "processed_mdi"]

        # !TODO replace with default_variables.py
        dv_base_path = Path(dv.MAG_INTERMEDIATE_DATA_DIR)

        dv_process_fits_path = Path(dv.MAG_PROCESSED_FITS_DIR)
        if not dv_process_fits_path.exists():
            dv_process_fits_path.mkdir(parents=True)

        dv_summary_plots_path = Path(dv.MAG_PROCESSED_SUMMARYPLOTS_DIR)
        if not dv_summary_plots_path.exists():
            dv_summary_plots_path.mkdir(parents=True)

        # Iterate through the columns and update the paths
        for old_column, new_column in zip(columns_to_update, new_columns):
            self.loaded_data[new_column] = self.loaded_data[old_column].map(
                lambda x: dv_base_path / Path(x).name if pd.notna(x) else x
            )

        # 1. Get SHARPs data
        self.start_date = dv.DATA_START_TIME
        self.mid_date = datetime(2011, 1, 1)
        self.end_date = datetime(2012, 1, 1)  # dv.DATA_END_TIME
        #   1a. sort df by datetime
        #   1b. extract min/max time
        sharps_data = HMISHARPs()
        #   1c. JSOC query to get df.
        meta1 = sharps_data.fetch_metadata(self.start_date, self.mid_date)
        meta2 = sharps_data.fetch_metadata(self.mid_date, self.end_date)
        meta = pd.concat([meta1, meta2]).drop_duplicates()

        logger.info(
            f"SHARP Keys: \n{meta[['T_REC','T_OBS','DATE-OBS','DATE__OBS','datetime','magnetogram_fits', 'url']]}"
        )  # the date-obs or date-avg

        print(list(meta.columns))
        sharp_keys = meta[["magnetogram_fits", "datetime", "url"]].add_suffix("_sharp")
        # !TODO match with a df of HMI images to get the fulldisk too
        # !TODO test this
        print(list(self.loaded_data.columns))
        self.loaded_data = self.loaded_data[["magnetogram_fits_hmi", "datetime_hmi", "url_hmi"]]
        self.loaded_data["datetime_hmi"] = pd.to_datetime(self.loaded_data["datetime_hmi"])
        self.loaded_data = self.loaded_data.dropna().reset_index()
        #
        self.merged_df = pd.merge(
            self.loaded_data, sharp_keys, left_on="datetime_hmi", right_on="datetime_sharp"
        )  # no tolerance as should be exact

        print(list(self.merged_df))

        logger.info(
            f"Merged Keys: \n{self.merged_df[['datetime_hmi','datetime_sharp','url_sharp', 'url_hmi']]}"
        )  # the date-obs or date-avg
        # #   1d. Do we need to pull down data? Yesxe
        urls = list(self.merged_df.url_sharp.dropna().unique())

        # # copied from `DataManager.fetch_magnetograms`
        # # obviously remove this... but for now...
        # #
        base_directory_path = Path(dv.MAG_RAW_SHARP_DATA_DIR)
        if not base_directory_path.exists():
            base_directory_path.mkdir(parents=True)
        #
        downloader = Downloader(
            max_conn=1,
            progress=True,
            overwrite=False,
            max_splits=1,
        )
        #
        paths = []
        for url in urls:
            filename = url.split("/")[-1]  # Extract the filename from the URL
            paths.append(base_directory_path / filename)
        #
        for aurl, fname in zip(urls, paths):
            downloader.enqueue_file(aurl, filename=fname, max_splits=1)
        #
        results = downloader.download()
        #
        if len(results.errors) != 0:
            logger.warn(f"results.errors: {results.errors}")
            # attempt a retry
            retry_count = 0
            while len(results.errors) != 0 and retry_count < 3:
                logger.info("retrying...")
                downloader.retry(results)
                retry_count += 1
            if len(results.errors) != 0:
                logger.error("Failed after maximum retries.")
            else:
                logger.info("Errors resolved after retry.")
        else:
            logger.info("No errors reported by parfive")

        self.results = results
        # 2. Set SHARP Regions into a df
        bottom_left_list = []
        top_right_list = []

        bottom_left_list_px = []
        top_right_list_px = []
        # !TODO groupby fd HMI image.
        for sharp_file, fd_file in tqdm(zip(self.merged_df.url_sharp, self.merged_df.url_hmi)):
            # full-disk map
            a_fd_map = sunpy.map.Map(
                Path("/Users/pjwright/Documents/work/ARCCnet/data/02_intermediate/mag/fits/") / Path(fd_file).name
            )
            # sharp map
            a_sharp_map = sunpy.map.Map(base_directory_path / Path(sharp_file).name)
            a_sharp_map = a_sharp_map.rotate()
            # Get the bottom-left and top-right coordinates
            #   2a. Extent of rectangle (arcseconds, pixels)
            #   2a. Extent of smooth bounding curve (arcseconds, pixels)
            # !TODO want to store the data in a QTable?
            bl = (a_sharp_map.bottom_left_coord.Tx.value, a_sharp_map.bottom_left_coord.Ty.value)
            tr = (a_sharp_map.top_right_coord.Tx.value, a_sharp_map.top_right_coord.Ty.value)
            # !TODO check if NOAA AR lands inside the image...
            # !TODO reproject these into the full-disk map
            bl_transformed = a_sharp_map.bottom_left_coord.transform_to(a_fd_map.coordinate_frame).to_pixel(
                a_fd_map.wcs
            )
            tr_transformed = a_sharp_map.top_right_coord.transform_to(a_fd_map.coordinate_frame).to_pixel(a_fd_map.wcs)

            bottom_left_list.append(bl)
            top_right_list.append(tr)
            bottom_left_list_px.append(bl_transformed)
            top_right_list_px.append(tr_transformed)

        # Add the new "bottom_left" and "top_right" columns to self.meta DataFrame
        self.merged_df["bottom_left_TxTy_arcsec"] = bottom_left_list
        self.merged_df["top_right_TxTy_arcsec"] = top_right_list
        self.merged_df["bottom_left_TxTy_px"] = bottom_left_list_px
        self.merged_df["top_right_TxTy_px"] = top_right_list_px

        # 4. Save df of ARs (NOAA matched SHARP along with NOAA classification)
        # !TODO merge with NOAA AR info (on lat/lng)
        self.merged_df.to_csv(Path(dv.DATA_DIR_PROCESSED) / "hmi_sharp_cutouts.csv")


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")
    # _ = MagnetogramProcessor()
    # _ = ARExtractor()
    _ = QSExtractor()
    # _ = ARDetection()
