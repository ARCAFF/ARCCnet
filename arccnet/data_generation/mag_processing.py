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
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["MagnetogramProcessor", "ARExtractor", "QSExtractor"]


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
        for old_column, new_column in zip(columns_to_update, new_columns):
            self.loaded_data[new_column] = self.loaded_data[old_column].map(
                lambda x: dv_base_path / Path(x).name if pd.notna(x) else x
            )

        # set empty list of cutout for hmi
        cutout_list_hmi = []
        cutout_hmi_dim = []
        rsun = []
        dsun = []
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

                rsun.append(my_hmi_submap.meta["rsun_obs"])  # want to move it
                dsun.append(my_hmi_submap.meta["dsun_obs"])
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
        self.loaded_subset.loc[:, "rsun_obs"] = rsun
        self.loaded_subset.loc[:, "dsun_obs"] = dsun
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
    def __init__(self, start_date=datetime(2010, 1, 1), end_date=datetime(2022, 12, 31)):
        self.start = start_date
        self.end = end_date

        filename = Path(dv.MAG_PROCESSED_DIR) / "processed.csv"

        if filename.exists():
            self.loaded_data = pd.read_csv(filename)

        self.loaded_data["datetime_srs"] = pd.to_datetime(self.loaded_data["datetime_srs"])

        self.subset_df = self.loaded_data[
            (self.loaded_data["datetime_srs"] >= self.start) & (self.loaded_data["datetime_srs"] <= self.end)
        ]

        grouped_data = self.subset_df.groupby("datetime_srs")

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

                vals.append(ar_centre)

            print(vals)
            logger.info(
                f"the srs time is {time_srs}, and the hmi time is {time_hmi}. The size of the group is len(group) {len(group)}"
            )

            qs_reg = []
            for i in range(0, 50):
                # create random location
                rand_1 = random.uniform(-1000, 1000) * u.arcsec
                rand_2 = random.uniform(-500, 500) * u.arcsec

                # convert to pixel coordinates
                _cd = SkyCoord(
                    rand_1,
                    rand_2,
                    frame=my_hmi_map.coordinate_frame,
                ).to_pixel(my_hmi_map.wcs)

                print(_cd)

                tt = []

                # check _cd is far enough from other vals
                for v in vals:
                    tt.append(
                        self.is_point_far_from_point(_cd[0], _cd[1], v[0], v[1], dv.X_EXTENT * 1.2, dv.Y_EXTENT * 1.2)
                    )

                if all(tt):
                    print("far enough from other points")
                    top_right = [_cd[0] + (dv.X_EXTENT - 1) / 2, _cd[1] + (dv.Y_EXTENT - 1) / 2] * u.pix
                    bottom_left = [_cd[0] - (dv.X_EXTENT - 1) / 2, _cd[1] - (dv.Y_EXTENT - 1) / 2] * u.pix
                    my_hmi_submap = my_hmi_map.submap(bottom_left, top_right=top_right)

                    fig = plt.figure(figsize=(5, 5))
                    ax = fig.add_subplot(projection=my_hmi_submap)
                    my_hmi_submap.plot_settings["norm"].vmin = -1500
                    my_hmi_submap.plot_settings["norm"].vmax = 1500
                    my_hmi_submap.plot(axes=ax, cmap="hmimag")

                    my_hmi_submap.save(
                        Path(
                            "/Users/pjwright/Documents/work/ARCCnet/data/03_processed/mag/qs_fits"
                        )  # need to make this manually
                        / f"{time_srs.year}-{time_srs.month}-{time_srs.day}_QS_{i}.fits",
                        overwrite=True,
                    )

                    del my_hmi_submap
                    vals.append(_cd)
                    qs_reg.append(_cd)

                print("vals", vals)

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

    # def is_value_outside_rectangle(self, x, y, x1, y1, x2, y2):
    #     # Check if the value is outside the rectangle
    #     return x < x1 or x > x2 or y < y1 or y > y2

    def is_point_far_from_point(self, x, y, x1, y1, threshold_x, threshold_y):
        # test this code
        return abs(x - x1) > threshold_x or abs(y - y1) > threshold_y


if __name__ == "__main__":
    logger.info(f"Executing {__file__} as main program")
    _ = MagnetogramProcessor()
    _ = ARExtractor()
    _ = QSExtractor()
