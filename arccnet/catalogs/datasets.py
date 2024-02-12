import numpy as np

from astropy.table import QTable

from arccnet.data_generation.utils.utils import grouped_stratified_split

# CLI to tar and compress
# tar -cf - fits quicklook cutout-magnetic-catalog-v20231027.parq cutout-mcintosh-catalog-v20231027.parq | pigz > arccnet-cutout-dataset-v20231027.tar.gz


def create_cutout_classification_datasets(cutout_catalog):
    # Set seed
    np.random.seed(42)

    cutout_cat = QTable.read(cutout_catalog.parq)

    # have data for
    clean_cutout_cat = cutout_cat[
        ((~cutout_cat["processed_path_image_hmi"].mask) | (~cutout_cat["processed_path_image_mdi"].mask))
    ]

    # Can't convert to pandas
    cols = clean_cutout_cat.colnames
    [
        cols.remove(n)
        for n in [
            "top_right_cutout_hmi",
            "bottom_left_cutout_hmi",
            "dim_image_cutout_hmi",
            "top_right_cutout_mdi",
            "bottom_left_cutout_mdi",
            "dim_image_cutout_mdi",
        ]
    ]

    clean_cutout_df = clean_cutout_cat[cols].to_pandas()

    # Make QS a new class in both magnetic and hale
    clean_cutout_df.loc[clean_cutout_df.region_type == "QS", "magnetic_class"] = "QS"
    clean_cutout_df.loc[clean_cutout_df.region_type == "QS", "mcintosh_class"] = "QS"

    # Will only tar up the local fits not source so need to change to be relative to that folder
    def make_paths_relative(cat, col):
        relpahts = ["/".join(p.split("/")[-2:]) for p in cat[col][~cat[col].mask]]
        cat[col][~cat[col].mask] = relpahts

    make_paths_relative(clean_cutout_cat, "path_image_cutout_hmi")
    make_paths_relative(clean_cutout_cat, "quicklook_path_hmi")
    make_paths_relative(clean_cutout_cat, "path_image_cutout_mdi")
    make_paths_relative(clean_cutout_cat, "quicklook_path_mdi")

    # Make train / test split
    mag_train, mag_test = grouped_stratified_split(clean_cutout_df, group_col="number", class_col="magnetic_class")
    hale_train, hale_test = grouped_stratified_split(clean_cutout_df, group_col="number", class_col="mcintosh_class")

    mag_train_cat = clean_cutout_cat[mag_train]
    clean_cutout_cat[mag_test]

    hale_train_cat = clean_cutout_cat[hale_train]
    clean_cutout_cat[hale_test]

    mag_train_cat.write("cutout-magnetic-catalog-v20231027.parq")
    hale_train_cat.write("cutout-mcintosh-catalog-v20231027.parq")


# CLI to tar and compress
# tar -cf - fits quicklook cutout-magnetic-catalog-v20231027.parq cutout-mcintosh-catalog-v20231027.parq | pigz > arccnet-cutout-dataset-v20231027.tar.gz


def create_detection_datasets(detection_catalog):
    np.random.seed(42)

    extraction_cat = QTable.read(detection_catalog)
    extraction_cat.rename_column("NOAA", "number")

    # Can't convert to pandas
    cols = extraction_cat.colnames
    [cols.remove(n) for n in ["top_right_cutout", "bottom_left_cutout"]]

    extraction_df = extraction_cat.to_pandas()

    # Will only tar up the local fits not source so need to change to be relative to that folder
    def make_paths_relative(cat, col):
        relpahts = ["/".join(p.split("/")[-2:]) for p in cat[col]]
        cat[col] = relpahts

    make_paths_relative(extraction_cat, "processed_path")

    # Make train / test split
    mag_train, mag_test = grouped_stratified_split(extraction_df, group_col="number", class_col="magnetic_class")
    hale_train, hale_test = grouped_stratified_split(extraction_df, group_col="number", class_col="mcintosh_class")

    mag_train_cat = extraction_cat[mag_train]
    extraction_cat[mag_test]

    hale_train_cat = extraction_cat[hale_train]
    extraction_cat[hale_test]

    mag_train_cat.write("cutout-magnetic-catalog-v20231027.parq")
    hale_train_cat.write("cutout-mcintosh-catalog-v20231027.parq")


# CLI to tar and compress
# tar -cf - fits quicklook cutout-magnetic-catalog-v20231027.parq cutout-mcintosh-catalog-v20231027.parq | pigz > arccnet-cutout-dataset-v20231027.tar.gz
