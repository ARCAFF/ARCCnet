from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import sunpy.map
from pandas import DataFrame
from tqdm import tqdm

from arccnet import config
from arccnet.data_generation.utils.data_logger import logger

__all__ = ["RegionDetection", "DetectionBox"]


@dataclass
class DetectionBox:
    fulldisk_path: Path
    cutout_path: Path
    bottom_left_coord_px: tuple[float, float]
    top_right_coord_px: tuple[float, float]


class RegionDetection:
    def __init__(self, filename: Path = None):
        """
        Initialize a RegionDetection instance

        Parameters
        ----------
        filename : Path, optional
            Path to the input CSV file. If no path is provided,
            this defaults to HMI-SHARPs: dv.MAG_INTERMEDIATE_HMISHARPS_DATA_CSV.
        """
        full_disk_path_column = "download_path"
        cutout_path_column = "download_path_arc"

        if filename is None:
            filename = config["paths"]["mag_intermediate_hmisharps_data_csv"]

        self.loaded_data = pd.read_csv(filename)

        self.bboxes = self.get_bboxes(self.loaded_data, full_disk_path_column, cutout_path_column)
        self.regiondetection_df = self.update_loaded_data(
            self.loaded_data, full_disk_path_column, cutout_path_column, self.bboxes
        )

    def get_bboxes(
        self,
        df: DataFrame,
        col_group: str,
        col_cutout: str,
    ) -> list[DetectionBox]:
        """
        Extract detection boxes from the input DataFrame.

        Parameters
        ----------
        df : `DataFrame`
            Input DataFrame

        col_group : `str`
            Column name for group

        col_cutout : `str`
            Column name for cutout URLs

        Returns
        -------
        `list[DetectionBox]`
            List of DetectionBox instances
        """
        grouped_data = df.groupby(col_group)
        bboxes = []
        for fulldisk_path, group in tqdm(grouped_data, total=len(grouped_data), desc="Processing"):
            fulldisk_map = sunpy.map.Map(Path(fulldisk_path))

            for _, row in group.iterrows():
                cutout_map = sunpy.map.Map(row[col_cutout])

                # rotate with missing, the value to use for pixels in the output map that are beyond the extent of the input map,
                # set to zero. the default is `np.nan`
                cutout_map = cutout_map.rotate(missing=0)
                bl_transformed = cutout_map.bottom_left_coord.transform_to(fulldisk_map.coordinate_frame).to_pixel(
                    fulldisk_map.wcs
                )
                tr_transformed = cutout_map.top_right_coord.transform_to(fulldisk_map.coordinate_frame).to_pixel(
                    fulldisk_map.wcs
                )

                bboxes.append(
                    DetectionBox(
                        fulldisk_path=Path(fulldisk_path),
                        bottom_left_coord_px=(bl_transformed[0].item(), bl_transformed[1].item()),
                        top_right_coord_px=(tr_transformed[0].item(), tr_transformed[1].item()),
                        cutout_path=Path(row[col_cutout]),
                    )
                )

            del cutout_map

        del fulldisk_map

        return bboxes

    def update_loaded_data(
        self,
        df: DataFrame,
        col_group: str,
        col_cutout: str,
        bboxes: list[DetectionBox],
    ) -> DataFrame:
        """
        Update the loaded DataFrame with detection box information.

        Parameters
        ----------
        df : `DataFrame`
            Input DataFrame

        col_group : `str`
            Column name for the group

        col_cutout : `str`
            Column name for cutout

        bboxes : `list[DetectionBox]`
            List of DetectionBox instances

        Returns
        -------
        `DataFrame`
            Updated DataFrame with coordinates
        """
        updated_df = df.copy()

        for bbox in bboxes:  # Assuming self.df contains the list of DetectionBox instances
            # Find rows in self.loaded_data that match the fulldisk_path
            matching_row = updated_df[
                (updated_df[col_group].apply(lambda x: Path(x)) == bbox.fulldisk_path)
                & (updated_df[col_cutout].apply(lambda x: Path(x)) == bbox.cutout_path)
            ]

            if len(matching_row) == 1:
                # Check if the columns exist in the DataFrame and add them if needed
                coord_cols = ["bottom_left_coord_px", "top_right_coord_px"]
                for col in coord_cols:
                    if col not in updated_df.columns:
                        updated_df[col] = None
                        logger.info(f"Column '{col}' added")

                updated_df.at[matching_row.index[0], coord_cols[0]] = bbox.bottom_left_coord_px
                updated_df.at[matching_row.index[0], coord_cols[1]] = bbox.top_right_coord_px
            else:
                logger.warn(
                    f"{len(matching_row)} rows matched with {bbox.fulldisk_path.name} and {bbox.cutout_path.name} "
                )

        return updated_df
