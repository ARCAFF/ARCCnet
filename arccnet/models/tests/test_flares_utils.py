from pathlib import Path

import pandas as pd

from arccnet.models.flares import utils as flare_utils


def test_check_fits_file_existence_resolves_paths(tmp_path: Path):
    data_folder = tmp_path
    dataset_folder = "dataset"
    fits_dir = data_folder / dataset_folder / "data" / "cutout_classification" / "fits"
    fits_dir.mkdir(parents=True)

    hmi_file = fits_dir / "20100407_235819_I-11060_HMI_SIDE1.fits"
    mdi_file = fits_dir / "19961212_235945_I-8003_MDI.fits"
    hmi_file.write_bytes(b"test-hmi")
    mdi_file.write_bytes(b"test-mdi")

    df = pd.DataFrame(
        {
            "path_image_cutout_hmi": [
                "20100407_235819_I-11060_HMI_SIDE1.fits",
                "",
                "/mnt/ARCAFF/v0.3.0/04_final/data/cutout_classification/fits/20100407_235819_I-11060_HMI_SIDE1.fits",
                "",
                "",
            ],
            "path_image_cutout_mdi": [
                "",
                "19961212_235945_I-8003_MDI.fits",
                "",
                "",
                "missing.fits",
            ],
        }
    )

    mapped_df, missing_idx = flare_utils.check_fits_file_existence(df, str(data_folder), dataset_folder)

    assert missing_idx == [3]
    assert mapped_df["file_exists"].tolist() == [True, True, True, False, False]

    assert mapped_df.loc[0, "path_image_cutout_hmi"] == hmi_file.name
    assert mapped_df.loc[1, "path_image_cutout_mdi"] == mdi_file.name
    assert mapped_df.loc[2, "path_image_cutout_hmi"] == hmi_file.name

    assert mapped_df.loc[0, "resolved_path"] == str(hmi_file)
    assert mapped_df.loc[1, "resolved_path"] == str(mdi_file)
    assert mapped_df.loc[2, "resolved_path"] == str(hmi_file)
    assert mapped_df.loc[4, "resolved_path"] is None
