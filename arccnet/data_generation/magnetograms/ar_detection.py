from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import sunpy.map
from tqdm import tqdm

import arccnet.data_generation.utils.default_variables as dv
from arccnet.data_generation.utils.data_logger import logger


@dataclass
class DetectionBox:
    fulldisk_path: Path
    cutout_path: Path
    bottom_left_coord_px: tuple[float, float]
    top_right_coord_px: tuple[float, float]


class ARDetection:
    def __init__(self, filename: Path = dv.MAG_INTERMEDIATE_HMISHARPS_DATA_CSV):
        """
        Initialize an instance of ARDetection.

        Parameters
        ----------
        filename : Path, optional
            Path to the input CSV file, by default dv.MAG_INTERMEDIATE_HMISHARPS_DATA_CSV
        """
        group_url = "url"
        cutout_url = "url_arc"

        base_directory_path = Path(dv.MAG_RAW_SHARP_DATA_DIR)
        self.loaded_data = pd.read_csv(filename)

        self.bboxes = self.get_bboxes(self.loaded_data, group_url, cutout_url, base_directory_path)
        self.updated_df = self.update_loaded_data(self.loaded_data, group_url, cutout_url, self.bboxes)

        # !TODO
        # plot function (don't need to save)
        # save df

    def get_bboxes(self, df, group_url, cutout_url, base_directory_path) -> list[DetectionBox]:
        """
        Extract detection boxes from the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        group_url : str
            Column name for group URLs.

        cutout_url : str
            Column name for cutout URLs.

        base_directory_path : Path
            Base directory path.

        Returns
        -------
        list[DetectionBox]
            List of DetectionBox instances.
        """
        grouped_data = df.groupby(group_url)
        bboxes = []
        for url_hmi, group in tqdm(grouped_data, total=len(grouped_data), desc="Processing"):
            fd_map = sunpy.map.Map(Path(dv.MAG_INTERMEDIATE_DATA_DIR) / Path(url_hmi).name)

            for _, row in group.iterrows():
                ar_map = sunpy.map.Map(base_directory_path / Path(row[cutout_url]).name)
                ar_map = ar_map.rotate()  # Move this elsewhere

                bl_transformed = ar_map.bottom_left_coord.transform_to(fd_map.coordinate_frame).to_pixel(fd_map.wcs)
                tr_transformed = ar_map.top_right_coord.transform_to(fd_map.coordinate_frame).to_pixel(fd_map.wcs)

                bboxes.append(
                    DetectionBox(
                        fulldisk_path=Path(dv.MAG_INTERMEDIATE_DATA_DIR) / Path(row[group_url]).name,
                        bottom_left_coord_px=(bl_transformed[0].item(), bl_transformed[1].item()),
                        top_right_coord_px=(tr_transformed[0].item(), tr_transformed[1].item()),
                        cutout_path=base_directory_path / Path(row[cutout_url]).name,
                    )
                )

            del ar_map

        del fd_map

        return bboxes

    def update_loaded_data(self, df, group_url, cutout_url, bboxes: list[DetectionBox]) -> pd.DataFrame:
        """
        Update the loaded DataFrame with detection box information.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        group_url : str
            Column name for group URLs.

        cutout_url : str
            Column name for cutout URLs.

        bboxes : list[DetectionBox]
            List of DetectionBox instances.

        Returns
        -------
        pd.DataFrame: Updated DataFrame
        """
        updated_df = df.copy()

        for bbox in bboxes:  # Assuming self.df contains the list of DetectionBox instances
            # Find rows in self.loaded_data that match the fulldisk_path
            matching_row = updated_df[
                (updated_df[group_url].apply(lambda x: Path(x).name) == bbox.fulldisk_path.name)
                & (updated_df[cutout_url].apply(lambda x: Path(x).name) == bbox.cutout_path.name)
            ]

            if len(matching_row) == 1:
                matching_row = updated_df[
                    (updated_df["url"].apply(lambda x: Path(x).name) == bbox.fulldisk_path.name)
                    & (updated_df["url_arc"].apply(lambda x: Path(x).name) == bbox.cutout_path.name)
                ]

                # Check if the columns exist in the DataFrame and add them if needed
                if "bottom_left_coord_px" not in updated_df.columns:
                    updated_df["bottom_left_coord_px"] = None
                    print("Column 'bottom_left_coord_px' added")
                if "top_right_coord_px" not in updated_df.columns:
                    updated_df["top_right_coord_px"] = None
                    print("Column 'top_right_coord_px' added")

                updated_df.at[matching_row.index[0], "bottom_left_coord_px"] = bbox.bottom_left_coord_px
                updated_df.at[matching_row.index[0], "top_right_coord_px"] = bbox.top_right_coord_px
            else:
                logger.warn(
                    f"{len(matching_row)} rows matched with {bbox.fulldisk_path.name} and {bbox.cutout_path.name} "
                )

        return updated_df


if __name__ == "__main__":
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    ar = ARDetection()

    rows = ar.updated_df.copy()
    random_datetime = rows.datetime_srs.unique()[1]
    rows = rows[rows["datetime_srs"] == random_datetime]

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Load the first image
    fd_map = sunpy.map.Map(Path(dv.MAG_INTERMEDIATE_DATA_DIR) / Path(rows.url.iloc[0]).name)
    plt.imshow(fd_map.data, cmap="gray", vmin=-20, vmax=20, origin="lower")
    # why is the magnetogram background mottled?

    for _, r in rows.iterrows():
        # Calculate rectangle parameters
        width = r["top_right_coord_px"][0] - r["bottom_left_coord_px"][0]
        height = r["top_right_coord_px"][1] - r["bottom_left_coord_px"][1]

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (r["bottom_left_coord_px"][0], r["bottom_left_coord_px"][1]),
            width,
            height,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )

        # Add the Rectangle patch to the axes
        ax.add_patch(rect)

    # Show the plot
    plt.show()
