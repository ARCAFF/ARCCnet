from pathlib import Path
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from tqdm import tqdm

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, QTable
from astropy.time import Time

from arccnet.data_generation.utils.data_logger import logger

matplotlib.use("Agg")

__all__ = ["RegionDetection", "DetectionBox"]


@dataclass
class DetectionBox:
    fulldisk_path: Path
    cutout_path: Path
    bottom_left_coord_px: tuple[float, float]
    top_right_coord_px: tuple[float, float]


class RegionDetectionTable(QTable):
    r"""
    Region Detection QTable object.

    """
    required_column_types = {
        "target_time": Time,
        "processed_path": str,
        "path_arc": str,
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
            raise ValueError("base_table must be an instance of RegionDetectionTable")

        # Check if additional columns already exist
        existing_columns = set(base_table.colnames).intersection(["top_right", "bottom_left"])
        if existing_columns:
            raise ValueError(f"Columns {existing_columns} already exist in base_table.")

        # Create a copy of the base table
        new_table = cls(base_table)

        # Add the new columns to the table
        length = len(base_table)
        # Create masked columns with specified values as masks
        new_table["top_right_cutout"] = MaskedColumn(data=[(0, 0)] * length * u.pix, mask=[(True, True)] * length)
        new_table["bottom_left_cutout"] = MaskedColumn(data=[(0, 0)] * length * u.pix, mask=[(True, True)] * length)

        return new_table


class RegionDetection:
    def __init__(self, table: QTable, col_group_path="processed_path", col_cutout_path="path_arc"):
        r"""
        Initialize a RegionDetection instance

        Parameters
        ----------
        table : `QTable`
            QTable object with col_group and col_cutout

        col_group_path : `str`
            column in `table` to group data by e.g full-disk data that maps to multiple cutouts

        col_cutout_path : `str`
            column in `table` for the cutout paths.

        """

        self._loaded_data = RegionDetectionTable(table)
        self._result_table = QTable(RegionDetectionTable.augment_table(self._loaded_data))
        self._col_group = col_group_path
        self._col_cutout = col_cutout_path

    def get_bboxes(
        self,
    ) -> list[DetectionBox]:
        r"""
        Extract detection boxes from the input DataFrame.

        Returns
        -------
        `list[DetectionBox]`
            List of DetectionBox instances
        """
        grouped_data = QTable(self._loaded_data).group_by(self._col_group)

        bboxes = []
        for group in tqdm(grouped_data.groups, total=len(grouped_data.groups), desc="Processing"):
            fulldisk_path = group["processed_path"][0]
            fulldisk_map = sunpy.map.Map(Path(fulldisk_path))

            for row in group:
                cutout_map = sunpy.map.Map(row[self._col_cutout])

                # rotate with missing, the value to use for pixels in the output map that are beyond the extent of the input map,
                # set to zero. the default is `np.nan`
                cutout_map = cutout_map.rotate(missing=0)
                bl_transformed = (
                    cutout_map.bottom_left_coord.transform_to(fulldisk_map.coordinate_frame).to_pixel(fulldisk_map.wcs)
                    * u.pix
                )
                tr_transformed = (
                    cutout_map.top_right_coord.transform_to(fulldisk_map.coordinate_frame).to_pixel(fulldisk_map.wcs)
                    * u.pix
                )

                bboxes.append(
                    DetectionBox(
                        fulldisk_path=Path(fulldisk_path),
                        cutout_path=Path(row[self._col_cutout]),
                        bottom_left_coord_px=bl_transformed,
                        top_right_coord_px=tr_transformed,
                    )
                )

            del cutout_map

        del fulldisk_map

        return self.update_loaded_data(bboxes), bboxes

    def update_loaded_data(
        self,
        bboxes: list[DetectionBox],
    ) -> QTable:
        r"""
        Update the loaded QTable with detection box information.

        Parameters
        ----------
        bboxes : `list[DetectionBox]`
            List of DetectionBox instances

        Returns
        -------
        `QTable`
            Updated QTable with coordinates
        """
        updated_table = self._result_table.copy()  # Assuming self._loaded_data is a QTable

        for bbox in bboxes:
            # Find rows in self._loaded_data that match the fulldisk_path and cutout_path
            matching_rows = np.where(
                (updated_table[self._col_group] == str(bbox.fulldisk_path))
                & (updated_table[self._col_cutout] == str(bbox.cutout_path))
            )

            if len(matching_rows[0]) == 1:
                # Update the masked columns directly using the row index
                updated_table[matching_rows[0][0]]["top_right_cutout"] = bbox.top_right_coord_px
                updated_table[matching_rows[0][0]]["bottom_left_cutout"] = bbox.bottom_left_coord_px
            else:
                logger.warn(f"{len(matching_rows)} rows matched with {bbox.fulldisk_path} and {bbox.cutout_path}")

        return RegionDetectionTable(updated_table)

    def summary_plots(
        self,
        table: RegionDetectionTable,
        summary_plot_path: Path,
    ) -> None:
        data = QTable(table)

        # quick and dirty. remove
        col = MaskedColumn(data=[Path()] * len(data), mask=[True] * len(data))
        data.add_column(col, name="quicklook_path")

        grouped_data = data.group_by("processed_path")

        logger.info("region detection ")
        for group in tqdm(grouped_data.groups, total=len(grouped_data.groups), desc="Plotting"):
            fulldisk_path = group["processed_path"][0]
            instrument = group["instrument"][0]

            fulldisk_map = sunpy.map.Map(Path(fulldisk_path))

            output_filename = (
                summary_plot_path / f"{fulldisk_map.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_{instrument}.png"
            )  # need to add to the table

            for row in group:
                row["quicklook_path"] = output_filename

            logger.info(group)

            self._summary_plot(group, fulldisk_map, output_filename)

        return grouped_data

    @staticmethod
    def _summary_plot(
        table: RegionDetectionTable,
        sunpy_map: sunpy.map.Map,
        output_filename: Path,
    ):
        """
        assumes a RegionDetectionTable that has been grouped by `processed_path`
        """
        # set up plot
        fig = plt.figure(figsize=(5, 5))
        # there may be an issue with this cmap and vmin/max (different gray values as background)
        ax = fig.add_subplot(projection=sunpy_map)
        sunpy_map.plot_settings["norm"].vmin = -1499
        sunpy_map.plot_settings["norm"].vmax = 1499
        sunpy_map.plot(axes=ax, cmap="hmimag")

        for row in table:
            # deal with boxes off the edge
            sunpy_map.draw_quadrangle(
                row["bottom_left_cutout"],
                axes=ax,
                top_right=row["top_right_cutout"],
                edgecolor="black",
                linewidth=1,
            )

            ax.plot_coord(
                SkyCoord(row["longitude"], row["latitude"], frame=sunpy.coordinates.frames.HeliographicStonyhurst),
                marker="o",
                linestyle="None",
                markeredgecolor="k",
                markersize=4,
                label=f'NOAA {row["NOAA"]}',
            )

            ax.legend()

        plt.savefig(
            output_filename,
            dpi=300,
        )

        plt.close("all")

        return
