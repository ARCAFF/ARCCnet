from pathlib import Path
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from tqdm import tqdm

import astropy.units as u
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
        grouped_data = table.group_by("processed_path")

        for group in tqdm(grouped_data.groups, total=len(grouped_data.groups), desc="Processing"):
            self._summary_plot(group, summary_plot_path)

        return

    @staticmethod
    def _summary_plot(
        table: RegionDetectionTable,
        summary_plot_path: Path,
    ):
        """
        assumes a RegionDetectionTable that has been grouped by `processed_path`

        """
        fulldisk_path = table["processed_path"][0]
        instrument = table["instrument"][0]

        if instrument == "HMI":
            num_col_name = "record_HARPNUM_arc"
            identifier = "HARP"
        elif instrument == "MDI":
            num_col_name = "record_TARPNUM_arc"
            identifier = "TARP"
        else:
            raise NotImplementedError()

        fulldisk_map = sunpy.map.Map(Path(fulldisk_path))

        # save to file
        output_filename = (
            summary_plot_path
            / f"{fulldisk_map.date.to_datetime().strftime('%Y%m%d_%H%M%S')}_{instrument}_{identifier}.png"
        )

        # set up plot
        fig = plt.figure(figsize=(5, 5))
        # there may be an issue with this cmap and vmin/max (different gray values as background)
        ax = fig.add_subplot(projection=fulldisk_map)
        fulldisk_map.plot_settings["norm"].vmin = -1499
        fulldisk_map.plot_settings["norm"].vmax = 1499
        fulldisk_map.plot(axes=ax, cmap="hmimag")

        text_objects = []

        for row in table:
            # deal with boxes off the edge
            fulldisk_map.draw_quadrangle(
                row["bottom_left_cutout"],
                axes=ax,
                top_right=row["top_right_cutout"],
                edgecolor="black",
                linewidth=1,
            )

            text = ax.text(
                row["bottom_left_cutout"][0].value
                + (row["top_right_cutout"][0].value - row["bottom_left_cutout"][0].value) / 2,
                row["bottom_left_cutout"][1].value
                + (row["top_right_cutout"][1].value - row["bottom_left_cutout"][1].value) / 2,
                row[num_col_name],
                **{"size": "x-small", "color": "black", "ha": "center", "va": "center"},
            )

            text_objects.append(text)

        plt.savefig(
            output_filename,
            dpi=300,
        )

        plt.close("all")

        for text in text_objects:
            text.remove()

        return
