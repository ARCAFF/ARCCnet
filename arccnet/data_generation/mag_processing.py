import copy
import random
import multiprocessing
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
from arccnet.data_generation.utils.utils import is_point_far_from_point, save_compressed_map
from arccnet.utils.logging import logger

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
        if len(self._table) > 0:
            file_list = (
                pd.Series(self._table[~self._table.mask[self._column_name]][self._column_name]).dropna().unique()
            )
        else:
            file_list = []
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
        self,
        use_multiprocessing: bool = True,
        merge_col_prefix: str = "processed_",
        overwrite: bool = True,
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
                for path in tqdm(
                    pool.imap(
                        _multiprocess_and_save_data_wrapper,
                        [(path, self.save_path, overwrite) for path in self.paths],
                    ),
                    total=len(self.paths),
                    desc="Processing",
                ):
                    processed_paths.append(path)
        else:
            for path in tqdm(self.paths, desc="Processing"):
                processed_path = _process_and_save_data(path, self.save_path, overwrite)
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
            if not new_table.mask[self._column_name][i]:
                filename = Path(path).name
                if filename in filename_mapping:
                    new_table["temp_processed_path"][i] = filename_mapping[filename]
                    # masked_processed_path[i] = filename_mapping[filename]

        # Replace the 'processed_path' column with the masked version
        new_table[col_prefix + self._column_name] = new_table["temp_processed_path"].astype(str)
        new_table.remove_column("temp_processed_path")

        return MagResult(new_table)


def _multiprocess_and_save_data_wrapper(args):
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
    return _process_and_save_data(file, output_dir, overwrite)


def _process_and_save_data(file: Path, output_dir: Path, overwrite: bool) -> Path:
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
        processed_data = _process_datum(file)
        save_compressed_map(processed_data, path=output_file, overwrite=True)

    return output_file


def _process_datum(file: Path) -> sunpy.map.Map:
    r"""
    Process a single data file.

    Processing Steps:
        1. Load and rotate
        2. Set off-disk data to 0
        3. !TODO additional steps?
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
    rotated_map = _rotate_datum(single_map)
    # 4. !TODO normalise radius to fixed value
    # 5. !TODO project to a certain location in space
    return rotated_map


def _rotate_datum(amap: sunpy.map.Map) -> sunpy.map.Map:
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
    return amap.rotate(missing=np.nan if isinstance(amap.data, np.floating) else 0)


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
        self.region_type = "AR"


class IABox(RegionBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.region_type = "IA"


class IIBox:
    def __init__(
        self,
        identifier: int = None,
    ):
        self.identifier = identifier
        self.region_type = "II"


class FilteredBox(RegionBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filepath = None
        self.region_type = "FB"


class QSBox(RegionBox):
    def __init__(self, sunpy_map: sunpy.map.Map, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.region_type = "QS"

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
        # "processed_path_image": str,
        "filtered": bool,
        "filter_reason": str,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not set(self.colnames).issuperset(set(self.required_column_types.keys())):
            raise ValueError(
                f"{self.__class__.__name__} must contain {list(self.required_column_types.keys())} columns"
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

        qs_rows = [
            "target_time",
            "number",
            "path_image_cutout",
            "top_right_cutout",
            "bottom_left_cutout",
            "dim_image_cutout",
            "longitude",
            "latitude",
            "processed_path_image_mag",
            "sum_ondisk_nans",
            "quicklook_path",
            "region_type",
            "filtered",
            "filter_reason",
        ]

        qs_table = copy.deepcopy(result_table[:0])[qs_rows]

        # # iterate through groups
        ar_table_all = None

        for tbtt in tqdm(table_by_target_time.groups):
            if len(np.unique(tbtt["processed_path_image_mag"])) != 1 and len(
                np.unique(tbtt["processed_path_image_cont"])
            ):
                raise ValueError("len(image_file) is not 1")

            if np.any(tbtt["processed_path_image_mag"].mask):
                continue

            # if np.any(tbtt["filtered"] == True):
            # only want to run if all regions in an image are not `not_ar,invalid_magnetic_class,invalid_mcintosh_class,`
            # e.g. all regions in an image have bad_lat_rate/bad_lon_rate?
            condition_met = (tbtt["filtered"] is True) & (
                tbtt["filter_reason"] != "not_ar,invalid_magnetic_class,invalid_mcintosh_class,"
            )
            if np.all(condition_met):
                ar_table = tbtt.copy()
                if ar_table_all is None:
                    ar_table_all = copy.deepcopy(ar_table)
                else:
                    ar_table_all = ARClassification(vstack([QTable(ar_table_all), QTable(ar_table)]))
                continue

            # probably need to split the tbtt into filtered etc. to keep them in
            rows = tbtt[tbtt["filtered"] == False].copy()  # noqa
            rows_filtered = tbtt[tbtt["filtered"] == True].copy()  # noqa

            if len(rows) == 0:  # only filtered rows
                ar_table = rows_filtered.copy()
                if ar_table_all is None:
                    ar_table_all = copy.deepcopy(ar_table)
                else:
                    ar_table_all = ARClassification(vstack([QTable(ar_table_all), QTable(ar_table)]))
                continue

            if tbtt["processed_path_image_mag"][0] == "None":
                continue

            mag_file = rows["processed_path_image_mag"][0]
            conf_file = rows["processed_path_image_cont"][0]
            mag_map = sunpy.map.Map(mag_file)
            cont_map = sunpy.map.Map(conf_file)
            quicklook_filename = (
                summary_plot_path
                / f"{mag_map.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_{mag_map.detector.replace(' ', '_')}.png"
            )
            time_catalog = rows["target_time"][0].to_datetime()

            # set nan values in the map to zero
            # workaround for issues seen in processing
            data = mag_map.data
            on_disk_nans = np.isnan(data)
            # if on_disk_nans.sum() > 0:
            #     indices = np.where(on_disk_nans)
            #     data[indices] = 0.0

            regions = []

            # add active regions to regions list
            valid_regions = self._validregion_extraction(rows, mag_map, cont_map, cutout_size, path=data_path)
            rows_filtered_labels, rows_filtered_unlabeled, filtered_regions = self._filteredregion_extraction(
                rows_filtered, mag_map, cutout_size, path=data_path
            )
            regions.extend(valid_regions)

            # ... update the table
            if len(rows) != len(regions):
                raise ValueError("len(rows) != len(regions)")
            for r, reg in zip(rows, regions):
                r["top_right_cutout"] = reg.top_right
                r["bottom_left_cutout"] = reg.bottom_left
                r["sum_ondisk_nans"] = on_disk_nans.sum()
                r["dim_image_cutout"] = reg.shape
                r["path_image_cutout"] = reg.filepath
                r["quicklook_path"] = quicklook_filename
                r["region_type"] = reg.region_type

            ar_table = copy.deepcopy(rows)

            if (len(rows_filtered_labels) + len(rows_filtered_unlabeled)) != len(rows_filtered):
                raise ValueError("len(filtered labeled rows) + len(filtered unlabeled rows) != len(all filtered rows)")
            if len(rows_filtered_labels) != len(filtered_regions):
                raise ValueError("length mismatch of `rows_filtered_labels`, `filtered_regions`")
            for rf, rw in zip(rows_filtered_labels, filtered_regions):
                rf["top_right_cutout"] = rw.top_right
                rf["bottom_left_cutout"] = rw.bottom_left
                rf["sum_ondisk_nans"] = on_disk_nans.sum()
                rf["dim_image_cutout"] = rw.shape
                rf["path_image_cutout"] = rw.filepath
                rf["quicklook_path"] = quicklook_filename
                rf["region_type"] = rw.region_type
                ar_table.add_row(rf)

            # these are rows that aren't associated with a region box
            for ul_row in rows_filtered_unlabeled:
                if ul_row["id"] == "II":
                    ul_row["region_type"] = "II"
                else:
                    # set a placeholder
                    ul_row["region_type"] = "FX"
                ar_table.add_row(ul_row)

            regions.extend(filtered_regions)
            # if quiet_sun, attempt to extract `num_random_attempts` regions and append
            if qs_random_attempts > 0:
                quiet_regions = self._quietsun_extraction(
                    mag_map=mag_map,
                    cont_map=cont_map,
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
                        "processed_path_image_mag": mag_file,
                        # "processed_path_image_cutout": conf_file,
                        "sum_ondisk_nans": on_disk_nans.sum(),
                        "quicklook_path": quicklook_filename,
                        "region_type": qsreg.region_type,
                    }
                    qs_table.add_row(new_row)

            # pass regions to plotting
            self.summary_plots(regions, mag_map, cont_map, cutout_size[1], quicklook_filename)

            del mag_map

            if ar_table_all is None:
                ar_table_all = copy.deepcopy(ar_table)
            else:
                ar_table_all = ARClassification(vstack([QTable(ar_table_all), QTable(ar_table)]))

        # not sure about this, but want to convert to strings, not leave as objects
        # Add a region_type, vstack, and sort by time.
        art = ARClassification(ar_table_all)
        # !TODO Could be an issue with the "--" making the way into the paths
        art["path_image_cutout"] = MaskedColumn(
            data=[str(p) for p in art["path_image_cutout"]], mask=art["path_image_cutout"].mask, fill_value=""
        )
        art["quicklook_path"] = MaskedColumn(
            data=[str(p) for p in art["quicklook_path"]], mask=art["quicklook_path"].mask, fill_value=""
        )

        qst = ARClassification(qs_table)
        # qst.replace_column("path_image_cutout", [str(p) for p in qst["path_image_cutout"]])
        qst["path_image_cutout"] = MaskedColumn(
            data=[str(p) for p in qst["path_image_cutout"]], mask=qst["path_image_cutout"].mask, fill_value=""
        )
        # qst.replace_column("quicklook_path", [str(p) for p in qst["quicklook_path"]])
        qst["quicklook_path"] = MaskedColumn(
            data=[str(p) for p in qst["quicklook_path"]], mask=qst["quicklook_path"].mask, fill_value=""
        )

        all_regions = ARClassification(vstack([QTable(art), QTable(qst)]))
        all_regions.sort("target_time")

        return art, qst, all_regions

    def _validregion_extraction(self, group, mag_map, cont_map, cutout_size, path) -> list[ARBox, IABox]:
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
                mag_map,
                row["latitude"],
                row["longitude"],
                xsize=xsize,
                ysize=ysize,
            )

            mag_submap = mag_map.submap(bottom_left, top_right=top_right)
            cont_submap = cont_map.submap(bottom_left, top_right=top_right)
            det = mag_submap.detector if mag_submap.detector != "" else mag_submap.instrument
            output_mag_filename = (
                path / f"{mag_submap.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_{row['id']}-{row['number']}_"
                f"mag_{det.replace(' ', '_')}.fits"
            )
            det = cont_submap.detector if cont_submap.detector != "" else cont_submap.instrument
            output_cont_filename = (
                path / f"{cont_submap.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_{row['id']}-{row['number']}_"
                f"cont_{det.replace(' ', '_')}.fits"
            )

            # store info in ARBox
            if row["id"] == "I":
                ar_objs.append(
                    ARBox(
                        top_right=top_right,
                        bottom_left=bottom_left,
                        shape=mag_submap.data.shape * u.pix,
                        ar_pos_pixels=ar_pos_pixels,
                        identifier=row["number"],
                        filepath=output_mag_filename,
                    )
                )
            elif row["id"] == "IA":
                ar_objs.append(
                    IABox(
                        top_right=top_right,
                        bottom_left=bottom_left,
                        shape=mag_submap.data.shape * u.pix,
                        ar_pos_pixels=ar_pos_pixels,
                        identifier=str(row["number"]),
                        filepath=output_mag_filename,
                    )
                )
            else:
                raise NotImplementedError(f"id == {row['id']} is not implemented.")

            save_compressed_map(mag_submap, path=output_mag_filename, overwrite=True)
            save_compressed_map(cont_submap, path=output_cont_filename, overwrite=True)

            del mag_submap

        return ar_objs

    def _filteredregion_extraction(self, group, sunpy_map, cutout_size, path) -> list[FilteredBox]:
        """
        Given a table `group` that share the same `sunpy_map`, return ARBox objects with a determined cutout_size
        """
        region_objs = []
        valid_rows = []  # List to keep track of rows with valid region_objs
        invalid_rows = []

        xsize, ysize = cutout_size
        for row in group:
            """
            Iterate through group, extracting active regions from lat/lon into image pixels
            """
            top_right, bottom_left, ar_pos_pixels = extract_region_lonlat(
                sunpy_map,
                row["latitude"],
                row["longitude"],
                xsize=xsize,
                ysize=ysize,
            )

            try:
                filtered_smap = sunpy_map.submap(bottom_left, top_right=top_right)
                region_objs.append(
                    FilteredBox(
                        top_right=top_right,
                        bottom_left=bottom_left,
                        shape=filtered_smap.data.shape * u.pix,
                        ar_pos_pixels=ar_pos_pixels,
                        identifier=str(row["id"]) + "-" + str(row["number"]),
                    )
                )

                valid_rows.append(row)  # Append to valid_rows
                del filtered_smap
            except Exception as e:
                invalid_rows.append(row)
                logger.warn(e)

        return valid_rows, invalid_rows, region_objs

    def _quietsun_extraction(
        self,
        mag_map: sunpy.map.Map,
        cont_map: sunpy.map.Map,
        cutout_size: tuple[u.pix, u.pix],
        num_random_attempts: int,
        max_iter: int,
        existing_regions: list[ARBox | QSBox],
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
                frame=mag_map.coordinate_frame,
            ).to_pixel(mag_map.wcs)

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
                qs_mag_submap = mag_map.submap(bottom_left, top_right=top_right)
                # save to file
                det = qs_mag_submap.detector if qs_mag_submap.detector != "" else qs_mag_submap.instrument
                output_mag_filename = (
                    path / f"{qs_mag_submap.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_QS-{qs_df_len}_"
                    f"mag_{det.replace(' ', '_')}.fits"
                )
                qs_cont_submap = cont_map.submap(bottom_left, top_right=top_right)
                det = qs_cont_submap.detector if qs_cont_submap.detector != "" else qs_cont_submap.instrument
                output_cont_filename = (
                    path / f"{qs_cont_submap.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_QS-{qs_df_len}_"
                    f"cont_{det.replace(' ', '_')}.fits"
                )

                # create QS BBox object
                qs_region = QSBox(
                    sunpy_map=qs_mag_submap,
                    top_right=top_right,
                    bottom_left=bottom_left,
                    shape=qs_mag_submap.data.shape,
                    ar_pos_pixels=qs_center_hproj,
                    identifier=qs_df_len,
                    filepath=output_mag_filename,
                )

                # only keep those with the center on disk
                if qs_region.center is np.nan:
                    iterations += 1
                    continue

                save_compressed_map(qs_mag_submap, path=output_mag_filename, overwrite=True)
                save_compressed_map(qs_cont_submap, path=output_cont_filename, overwrite=True)

                existing_regions.append(qs_region)
                qsbox_objs.append(qs_region)

                del qs_mag_submap, qs_cont_submap  # unsure if necessary; was having memories issues

                qs_df_len += 1
                iterations += 1

        return qsbox_objs

    @u.quantity_input
    def summary_plots(
        self,
        regions: list[ARBox | QSBox],
        mag_map: sunpy.map.Map,
        cont_map: sunpy.map.Map,
        ysize: u.pix,
        output_filename: Path,
    ) -> None:
        fig = plt.figure(figsize=(5, 10))

        # there may be an issue with this cmap and vmin/max (different gray values as background)
        mag_ax = fig.add_subplot(211, projection=mag_map)
        mag_map.plot_settings["norm"].vmin = -1499
        mag_map.plot_settings["norm"].vmax = 1499
        mag_map.plot(axes=mag_ax, cmap="hmimag")
        mag_map.draw_grid(axes=mag_ax)

        cont_ax = fig.add_subplot(212, projection=cont_map)
        cont_map.plot(axes=cont_ax, vmin=0.95, vmax=1.05)
        cont_map.draw_grid(axes=cont_ax)

        text_objects = []

        for box_info in regions:
            if isinstance(box_info, ARBox):
                rectangle_cr = "red"
            elif isinstance(box_info, QSBox):
                rectangle_cr = "blue"
            elif isinstance(box_info, FilteredBox):
                rectangle_cr = "black"
            elif isinstance(box_info, IABox):
                rectangle_cr = "darkslategrey"
            else:
                raise ValueError("Unsupported box type")

            # deal with boxes off the edge
            mag_map.draw_quadrangle(
                box_info.bottom_left, axes=mag_ax, top_right=box_info.top_right, edgecolor=rectangle_cr, linewidth=1
            )

            cont_map.draw_quadrangle(
                box_info.bottom_left, axes=cont_ax, top_right=box_info.top_right, edgecolor=rectangle_cr, linewidth=1
            )

            text = mag_ax.text(
                box_info.ar_pos_pixels[0],
                box_info.ar_pos_pixels[1] + ysize / u.pix / 2 + ysize / u.pix / 10,
                box_info.identifier,
                **{"size": "x-small", "color": "black", "ha": "center"},
            )
            text_objects.append(text)

            text = cont_ax.text(
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
