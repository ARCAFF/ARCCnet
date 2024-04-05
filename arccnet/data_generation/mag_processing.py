import random
import multiprocessing
from typing import Union
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy.map
from tqdm import tqdm

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, QTable, vstack
from astropy.time import Time

from arccnet.data_generation.data_manager import Result as MagResult
from arccnet.data_generation.utils.data_logger import logger
from arccnet.data_generation.utils.utils import is_point_far_from_point, save_compressed_map

matplotlib.use("Agg")

__all__ = ["MagnetogramProcessor", "RegionExtractor"]


class MagnetogramProcessor:
    """
    Process Magnetograms.

    This class provides methods to process magnetogram data using multiprocessing.
    """

    def __init__(
        self,
        table: QTable,
        save_path: Path,
        column_name: str,
    ) -> None:
        logger.debug("Instantiated `MagnetogramProcessor`")

        self._table = QTable(table)
        if not save_path.exists():
            save_path.mkdir(parents=True)
        self._save_path = save_path
        self._column_name = column_name

        # need to fix these for QTable
        file_list = pd.Series(self._table[~self._table[self._column_name].mask][self._column_name]).dropna().unique()
        paths = [Path(path) if isinstance(path, str) else np.nan for path in file_list]

        # check if paths exist
        existing_paths = [path for path in paths if path.exists()]  # check if the raw files exist
        if len(existing_paths) < len(paths):
            missing_paths = [str(path) for path in paths if path not in existing_paths]
            logger.warn(f"The following paths do not exist: {', '.join(missing_paths)}")

        self.paths = paths

    @property
    def table(self):
        return self._table

    @property
    def save_path(self):
        return self._save_path

    @property
    def column_name(self):
        return self._column_name

    def process(
        self, use_multiprocessing: bool = True, merge_col_prefix: str = "processed_", overwrite: bool = True
    ) -> dict:
        r"""
        Process data using multiprocessing.
        """

        processed_paths = []  # list of processed filepaths

        logger.info(
            f"processing of {len(self.paths)} paths (multiprocessing = {use_multiprocessing}, overwrite = {overwrite})"
        )
        if use_multiprocessing:
            # Use tqdm to create a progress bar for multiprocessing
            with multiprocessing.Pool() as pool:
                for processed_path in tqdm(
                    pool.imap_unordered(
                        self._multiprocess_and_save_data_wrapper,
                        [(path, self.save_path, overwrite) for path in self.paths],
                    ),
                    total=len(self.paths),
                    desc="Processing",
                ):
                    processed_paths.append(processed_path)
                    # pass
        else:
            for path in tqdm(self.paths, desc="Processing"):
                processed_path = self._process_and_save_data(path, self.save_path, overwrite)
                processed_paths.append(processed_path)

        self._processed_path_mapping = {path.name: path for path in processed_paths}

        merged_table = self._add_processed_paths(self._processed_path_mapping, col_prefix=merge_col_prefix)
        return merged_table

    def _add_processed_paths(self, filename_mapping, col_prefix):
        new_table = self._table.copy()
        # new_table["processed_path"] = None
        length = len(new_table)
        new_table["temp_processed_path"] = MaskedColumn(data=[Path()] * length, mask=[True] * length)
        # Create a masked version of the 'processed_path' column with the desired mask
        # masked_processed_path = MaskedColumn(
        #     new_table[col_prefix + self._column_name], mask=(new_table[col_prefix + self._column_name] is None)
        # )

        # Update 'processed_path' directly within the masked column
        for i, path in enumerate(new_table[self._column_name]):
            if not new_table[self._column_name].mask[i]:
                filename = Path(path).name
                if filename in filename_mapping:
                    new_table["temp_processed_path"][i] = filename_mapping[filename]
                    # masked_processed_path[i] = filename_mapping[filename]

        # Replace the 'processed_path' column with the masked version
        new_table[col_prefix + self._column_name] = new_table["temp_processed_path"].astype(str)
        new_table.remove_column("temp_processed_path")

        return MagResult(new_table)

    def _multiprocess_and_save_data_wrapper(self, args):
        r"""
        Wrapper method to process and save data using `_process_and_save_data`.

        This method takes a tuple of arguments containing the file path and the output directory,
        and then calls the `_process_and_save_data` method with the provided arguments.

        Parameters
        ----------
        args : `tuple`
            A tuple containing the file path and output directory.

        Returns
        -------
        `Path`
            A path for the processed file

        See Also:
        --------
        _process_and_save_data, process
        """
        file, output_dir, overwrite = args
        return self._process_and_save_data(file, output_dir, overwrite)

    def _process_and_save_data(self, file: Path, output_dir: Path, overwrite: bool) -> Path:
        r"""
        Process data and save compressed map.

        Parameters
        ----------
        file : `Path`
            Data file path.

        output_dir : `Path`
            Directory to save processed data.

        Returns
        -------
        None
        """
        output_file = output_dir / file.name  # !TODO prefix the file.name?

        if not output_file.exists() or overwrite:
            processed_data = self._process_datum(file)
            save_compressed_map(processed_data, path=output_file, overwrite=True)

        return output_file

    def _process_datum(self, file) -> sunpy.map.Map:
        r"""
        Process a single data file.

        Processing Steps:
            1. Load and rotate
            2. Set off-disk data to 0
            # !TODO
            4. Normalise radius to a fixed value
            5. Project to a certain location in space

        Parameters
        ----------
        file : `Path`
            Data file path.

        Returns
        -------
        rotated_map : `sunpy.map.Map`
            Processed sunpy map.
        """
        #!TODO remove 'BLANK' keyword
        # v3925 WARNING: VerifyWarning: Invalid 'BLANK' keyword in header.
        # The 'BLANK' keyword is only applicable to integer data, and will be ignored in this HDU.
        # [astropy.io.fits.hdu.image]
        # 1. Load & Rotate
        single_map = sunpy.map.Map(file)
        # 2. set data off-disk to 0 (np.nan would be ideal, but deep learning)
        single_map.data[~sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(single_map))] = 0.0
        rotated_map = self._rotate_datum(single_map)
        # 4. !TODO normalise radius to fixed value
        # 5. !TODO project to a certain location in space
        return rotated_map

    def _rotate_datum(self, amap: sunpy.map.Map) -> sunpy.map.Map:
        r"""
        Rotate a map according to metadata.

        Args:
        amap : `sunpy.map.Map`
            An input sunpy map.

        Parameters
        ----------
        rotated_map : `sunpy.map.Map`
            Rotated sunpy map.

        Notes
        -----
        before rotation a HMI map may have: `crota2 = 180.082565`, for example.
        """
        return amap.rotate()


class RegionBox:
    r"""
    Parameters
    ----------
    top_right : `tuple[float, float]`
        pixel coordinates of the top right of the bounding box

    bottom_left : `tuple[float, float]`
        pixel coordinates of the top right of the bounding box

    shape : `tuple[int, int]`
        shape in pixels of the region

    ar_pos_pixels : `tuple[int, int]`
        pixel coordinates of the active region centre

    identifier: `int`
        an identifier for the region, e.g. NOAA AR Number. Default is None

    filepath: `Path`
        filepath of the region. Default is None

    """

    def __init__(
        self,
        top_right: tuple[float, float],
        bottom_left: tuple[float, float],
        shape: tuple[int, int],
        ar_pos_pixels: tuple[int, int],
        identifier: int = None,
        filepath: Path = None,
    ):
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.identifier = identifier
        self.shape = shape
        self.ar_pos_pixels = ar_pos_pixels
        self.filepath = filepath


class ARBox(RegionBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QSBox(RegionBox):
    def __init__(self, sunpy_map: sunpy.map.Map, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if sunpy.map.coordinate_is_on_solar_disk(sunpy_map.center):
            latlon = sunpy_map.center.transform_to(sunpy.coordinates.frames.HeliographicStonyhurst)
            self.center = sunpy_map.center
            self.latitude = latlon.lat.value
            self.longitude = latlon.lon.value
        else:
            self.center = np.nan
            self.latitude = np.nan
            self.longitude = np.nan


class ARClassification(QTable):
    r"""
    Result object defines both the result and download status.

    The value of the 'path' is used to encode if the corresponding file was downloaded or not.

    Notes
    -----
    Under the hood uses QTable and Masked columns to define if a file was downloaded or not

    """
    required_column_types = {
        "target_time": Time,
        "number": int,
        "latitude": u.deg,
        "longitude": u.deg,
        "processed_path_image": str,
        "filtered": bool,
        "filter_reason": str,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not set(self.colnames).issuperset(set(self.required_column_types.keys())):
            raise ValueError(
                f"{self.__class__.__name__} must contain " f"{list(self.required_column_types.keys())} columns"
            )

    @classmethod
    def augment_table(cls, base_table):
        if not isinstance(base_table, cls):
            raise ValueError("base_table must be an instance of ARClassification")

        # Check if additional columns already exist
        existing_columns = set(base_table.colnames).intersection(
            [
                "top_right_cutout",
                "bottom_left_cutout",
                "path_image_cutout",
                "dim_image_cutout",
                "sum_ondisk_nans",
                "quicklook_path",
                "region_type",
            ]
        )
        if existing_columns:
            raise ValueError(f"Columns {existing_columns} already exist in base_table.")

        # Create a copy of the base table
        new_table = cls(base_table)

        # Add the new columns to the table
        length = len(base_table)
        # Create masked columns with specified values as masks
        new_table["top_right_cutout"] = MaskedColumn(data=[(0, 0)] * length * u.pix, mask=[(True, True)] * length)
        new_table["bottom_left_cutout"] = MaskedColumn(data=[(0, 0)] * length * u.pix, mask=[(True, True)] * length)
        new_table["path_image_cutout"] = MaskedColumn(data=[Path()] * length, mask=[True] * length)
        new_table["dim_image_cutout"] = MaskedColumn(data=[(0, 0)] * length * u.pix, mask=[(True, True)] * length)
        new_table["sum_ondisk_nans"] = MaskedColumn(data=[-1] * length, dtype=np.int64, mask=[True] * length)
        new_table["quicklook_path"] = MaskedColumn(data=[Path()] * length, mask=[True] * length)
        new_table["region_type"] = MaskedColumn(data=["XX"] * length, mask=[True] * length)

        return new_table


# maybe want to pass a result?
class RegionExtractor:
    def __init__(
        self,
        table: QTable,
    ) -> None:
        # self._table = ARClassification(table[~table["processed_path_image"].mask])  # this doesn't deal with the None...
        self._table = ARClassification(table)  # this doesn't deal with the None...

    def extract_regions(self, cutout_size, data_path, summary_plot_path, qs_random_attempts=10, qs_max_iter=20):
        result_table = QTable(ARClassification.augment_table(self._table))
        table_by_target_time = result_table.group_by("target_time")

        ar_table = result_table[:0].copy()
        qs_table = result_table[:0][
            "target_time",
            "number",
            "path_image_cutout",
            "top_right_cutout",
            "bottom_left_cutout",
            "dim_image_cutout",
            "longitude",
            "latitude",
            "processed_path_image",
            "sum_ondisk_nans",
            "quicklook_path",
            "region_type",
            "filtered",
            "filter_reason",
        ].copy()

        # iterate through groups
        for tbtt in tqdm(table_by_target_time.groups):
            if len(np.unique(tbtt["processed_path_image"])) != 1:
                raise ValueError("len(image_file) is not 1")

            if np.any(tbtt["processed_path_image"].mask):
                continue

            # if np.any(tbtt["filtered"] == True):
            # only want to run if all regions in an image are not `not_ar,invalid_magnetic_class,invalid_mcintosh_class,`
            # e.g. all regions in an image have bad_lat_rate/bad_lon_rate?
            condition_met = (tbtt["filtered"] is True) & (
                tbtt["filter_reason"] != "not_ar,invalid_magnetic_class,invalid_mcintosh_class,"
            )
            if np.all(condition_met):
                for tbttrw in tbtt:
                    ar_table.add_row(tbttrw)
                continue

            # probably need to split the tbtt into filtered etc. to keep them in
            rows = tbtt[tbtt["filtered"] == False].copy()  # noqa
            rows_filtered = tbtt[tbtt["filtered"] == True].copy()  # noqa

            if len(rows) == 0:  # only filtered rows
                for rw in rows_filtered:
                    ar_table.add_row(rw)
                continue

            if tbtt["processed_path_image"][0] == "None":
                continue

            image_file = rows["processed_path_image"][0]
            image_map = sunpy.map.Map(image_file)
            quicklook_filename = (
                summary_plot_path
                / f"{image_map.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_{image_map.instrument.replace(' ', '_')}.png"
            )
            time_catalog = rows["target_time"][0].to_datetime()

            # set nan values in the map to zero
            # workaround for issues seen in processing
            data = image_map.data
            on_disk_nans = np.isnan(data)
            # if on_disk_nans.sum() > 0:
            #     indices = np.where(on_disk_nans)
            #     data[indices] = 0.0

            regions = []

            # add active regions to regions list
            active_regions = self._activeregion_extraction(rows, image_map, cutout_size, path=data_path)
            regions.extend(active_regions)

            # ... update the table
            for r, reg in zip(rows, regions):
                r["top_right_cutout"] = reg.top_right
                r["bottom_left_cutout"] = reg.bottom_left
                r["sum_ondisk_nans"] = on_disk_nans.sum()
                r["dim_image_cutout"] = reg.shape
                r["path_image_cutout"] = reg.filepath
                r["quicklook_path"] = quicklook_filename
                r["region_type"] = "AR"

                ar_table.add_row(r)

            for rw in rows_filtered:
                ar_table.add_row(rw)

            # if quiet_sun, attempt to extract `num_random_attempts` regions and append
            if qs_random_attempts > 0:
                quiet_regions = self._quietsun_extraction(
                    sunpy_map=image_map,
                    cutout_size=cutout_size,
                    num_random_attempts=qs_random_attempts,
                    max_iter=qs_max_iter,
                    existing_regions=regions,
                    path=data_path,
                )
                regions.extend(quiet_regions)
                for qsreg in quiet_regions:
                    # update dataframe
                    new_row = {
                        "target_time": Time(time_catalog),
                        "number": qsreg.identifier,
                        "path_image_cutout": qsreg.filepath,
                        "top_right_cutout": qsreg.top_right,
                        "bottom_left_cutout": qsreg.bottom_left,
                        "dim_image_cutout": qsreg.shape * u.pix,
                        "longitude": qsreg.longitude * u.deg,
                        "latitude": qsreg.latitude * u.deg,
                        "processed_path_image": image_file,
                        "sum_ondisk_nans": on_disk_nans.sum(),
                        "quicklook_path": quicklook_filename,
                        "region_type": "QS",
                    }
                    qs_table.add_row(new_row)

            # pass regions to plotting
            self.summary_plots(regions, image_map, cutout_size[1], quicklook_filename)

            del image_map

        # not sure about this, but want to convert to strings, not leave as objects
        # Add a region_type, vstack, and sort by time.
        ttt = ARClassification(ar_table)
        ttt.replace_column("path_image_cutout", [str(p) for p in ttt["path_image_cutout"]])
        ttt.replace_column("quicklook_path", [str(p) for p in ttt["quicklook_path"]])

        qst = ARClassification(qs_table)
        qst.replace_column("path_image_cutout", [str(p) for p in qst["path_image_cutout"]])
        qst.replace_column("quicklook_path", [str(p) for p in qst["quicklook_path"]])

        all_regions = ARClassification(vstack([QTable(ttt), QTable(qst)]))
        all_regions.sort("target_time")

        return ttt, qst, all_regions

    def _activeregion_extraction(self, group, sunpy_map, cutout_size, path) -> list[ARBox]:
        """
        given a table `group` that share the same `sunpy_map`, return ARBox objects with a determined cutout_size
        """
        ar_objs = []
        xsize, ysize = cutout_size
        for row in group:
            """
            iterate through group, extracting active regions from lat/lon into image pixels
            """
            top_right, bottom_left, ar_pos_pixels = extract_region_lonlat(
                sunpy_map,
                row["latitude"],
                row["longitude"],
                xsize=xsize,
                ysize=ysize,
            )

            hmi_smap = sunpy_map.submap(bottom_left, top_right=top_right)

            output_filename = (
                path
                / f"{sunpy_map.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_AR-{row['number']}_{sunpy_map.instrument.replace(' ', '_')}.fits"
            )

            save_compressed_map(hmi_smap, path=output_filename, overwrite=True)

            # store info in ARBox
            ar_objs.append(
                ARBox(
                    top_right=top_right,
                    bottom_left=bottom_left,
                    shape=hmi_smap.data.shape * u.pix,
                    ar_pos_pixels=ar_pos_pixels,
                    identifier=row["number"],
                    filepath=output_filename,
                )
            )

            del hmi_smap
        return ar_objs

    def _quietsun_extraction(
        self,
        sunpy_map: sunpy.map.Map,
        cutout_size: tuple[u.pix, u.pix],
        num_random_attempts: int,
        max_iter: int,
        existing_regions: list[Union[ARBox, QSBox]],
        path: Path,
    ) -> list[QSBox]:
        """
        extract regions of `cutout_size`, at locations not covered by `existing_regions`
        """
        xsize, ysize = cutout_size
        iterations = 0
        qs_df_len = 0
        qsbox_objs = []

        while qs_df_len < num_random_attempts and iterations <= max_iter:
            # there may be an existing CS algo for this,
            # it's essentially a simplified 2D bin packing problem,

            # generate random lng/lat and convert Helioprojective coordinates to pixel coordinates
            qs_center_hproj = SkyCoord(
                random.uniform(-1000, 1000) * u.arcsec,
                random.uniform(-1000, 1000) * u.arcsec,
                frame=sunpy_map.coordinate_frame,
            ).to_pixel(sunpy_map.wcs)

            # check ar_pos_hproj is far enough from other vals
            candidates = list(
                map(
                    lambda v: is_point_far_from_point(
                        qs_center_hproj[0], qs_center_hproj[1], v[0], v[1], xsize / u.pix * 1.01, ysize / u.pix * 1.01
                    ),
                    [box_info.ar_pos_pixels for box_info in existing_regions],
                )
            )

            # if far enough away from all other values
            if all(candidates):
                # generate the submap
                bottom_left, top_right = pixel_to_bboxcoords(xsize, ysize, qs_center_hproj * u.pix)
                qs_submap = sunpy_map.submap(bottom_left, top_right=top_right)

                # save to file
                output_filename = (
                    path
                    / f"{sunpy_map.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_QS-{qs_df_len}_{sunpy_map.instrument.replace(' ', '_')}.fits"
                )

                # create QS BBox object
                qs_region = QSBox(
                    sunpy_map=qs_submap,
                    top_right=top_right,
                    bottom_left=bottom_left,
                    shape=qs_submap.data.shape,
                    ar_pos_pixels=qs_center_hproj,
                    identifier=qs_df_len,
                    filepath=output_filename,
                )

                # only keep those with the center on disk
                if qs_region.center is np.nan:
                    iterations += 1
                    continue

                save_compressed_map(qs_submap, path=output_filename, overwrite=True)

                existing_regions.append(qs_region)
                qsbox_objs.append(qs_region)

                del qs_submap  # unsure if necessary; was having memories issues

                qs_df_len += 1
                iterations += 1

        return qsbox_objs

    @u.quantity_input
    def summary_plots(
        self,
        regions: list[Union[ARBox, QSBox]],
        sunpy_map: sunpy.map.Map,
        ysize: u.pix,
        output_filename: Path,
    ) -> None:
        fig = plt.figure(figsize=(5, 5))

        # there may be an issue with this cmap and vmin/max (different gray values as background)
        ax = fig.add_subplot(projection=sunpy_map)
        sunpy_map.plot_settings["norm"].vmin = -1499
        sunpy_map.plot_settings["norm"].vmax = 1499
        sunpy_map.plot(axes=ax, cmap="hmimag")

        text_objects = []

        for box_info in regions:
            if isinstance(box_info, ARBox):
                rectangle_cr = "red"
            elif isinstance(box_info, QSBox):
                rectangle_cr = "blue"
            else:
                raise ValueError("Unsupported box type")

            # deal with boxes off the edge
            sunpy_map.draw_quadrangle(
                box_info.bottom_left,
                axes=ax,
                top_right=box_info.top_right,
                edgecolor=rectangle_cr,
                linewidth=1,
            )

            text = ax.text(
                box_info.ar_pos_pixels[0],
                box_info.ar_pos_pixels[1] + ysize / u.pix / 2 + ysize / u.pix / 10,
                box_info.identifier,
                **{"size": "x-small", "color": "black", "ha": "center"},
            )

            text_objects.append(text)

        plt.savefig(
            output_filename,
            dpi=300,
        )
        plt.close("all")

        for text in text_objects:
            text.remove()

        return 0


@u.quantity_input
def extract_region_lonlat(sunpy_map, latitude: u.deg, longitude: u.deg, xsize: u.pix, ysize: u.pix) -> sunpy.map.Map:
    r"""

    Parameters
    ----------
    map : `sunpy.map.Map`

    latitude : `u.deg`

    longitude : `u.deg`

    xsize : `int`, `u.pix`
        x extent of region to extract (in pixels)

    ysize : `int`, `u.pix`
        y extend of region to extract (in pixels)


    Returns
    -------
    `tuple[float], tuple[float], tuple[float]]`
        locations of the top right, bottom left and active region center
    """
    ar_pos_pixels = latlon_to_map_pixels(latitude, longitude, sunpy_map)
    bottom_left, top_right = pixel_to_bboxcoords(xsize, ysize, ar_pos_pixels * u.pix)

    return top_right, bottom_left, ar_pos_pixels


@u.quantity_input
def pixel_to_bboxcoords(xsize: u.pix, ysize: u.pix, box_center: u.pix):
    r"""
    Given the box center, and xsize, ysize, return the bottom left and top right coordinates in pixels
    """
    # remove u.pix
    xsize = xsize.value
    ysize = ysize.value
    box_center = box_center.value

    top_right = [box_center[0] + (xsize - 1) / 2, box_center[1] + (ysize - 1) / 2] * u.pix
    bottom_left = [
        box_center[0] - (xsize - 1) / 2,
        box_center[1] - (ysize - 1) / 2,
    ] * u.pix

    return bottom_left, top_right


@u.quantity_input
def latlon_to_map_pixels(
    latitude: u.deg,
    longitude: u.deg,
    sunpy_map: sunpy.map.Map,
    frame=sunpy.coordinates.frames.HeliographicStonyhurst,
):
    r"""
    Given lat/lon in degrees, convert to pixel locations
    """
    ar_pos_hgs = SkyCoord(
        longitude,
        latitude,
        obstime=sunpy_map.date,
        frame=frame,
    )
    transformed = ar_pos_hgs.transform_to(sunpy_map.coordinate_frame)
    ar_pos_pixels = transformed.to_pixel(sunpy_map.wcs)
    return ar_pos_pixels


def map_pixels_to_latlon(sunpy_map: sunpy.map.Map):
    r"""
    provide pixels, get out latlon
    """
    pass
