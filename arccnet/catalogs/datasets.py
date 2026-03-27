import numpy as np

from astropy.table import QTable

from arccnet.data_generation.utils.utils import grouped_stratified_split

# CLI to tar and compress
# tar -cf - fits quicklook cutout-magnetic-catalog-v20231027.parq cutout-mcintosh-catalog-v20231027.parq | pigz > arccnet-cutout-dataset-v20231027.tar.gz


def create_cutout_classification_datasets(cutout_catalog_path, random_seed=42):
    # Set seed
    np.random.seed(random_seed)

    cutout_cat = QTable.read(cutout_catalog_path)

    filtered_cat = cutout_cat[~(cutout_cat["filtered_mdi"] | cutout_cat["filtered_hmi"])]

    # Can only convert 1D to pands
    cols = [name for name in filtered_cat.colnames if len(filtered_cat[name].shape) <= 1]

    filtered_cat_df = filtered_cat[cols].to_pandas()

    # Make new QS and PLG classes in both magnetic and hale
    filtered_cat_df.loc[filtered_cat_df.region_type == "QS", "magnetic_class"] = "QS"
    filtered_cat_df.loc[filtered_cat_df.region_type == "QS", "mcintosh_class"] = "QS"

    filtered_cat_df.loc[filtered_cat_df.region_type == "IA", "magnetic_class"] = "PLG"  # plage
    filtered_cat_df.loc[filtered_cat_df.region_type == "IA", "mcintosh_class"] = "PLG"

    # Will only tar up the local fits not source so need to change to be relative to that folder
    def make_paths_relative(cat, col):
        relpahts = ["/".join(p.split("/")[-2:]) for p in cat[col][~cat[col].mask]]
        cat[col][~cat[col].mask] = relpahts

    make_paths_relative(filtered_cat, "path_image_cutout_hmi")
    make_paths_relative(filtered_cat, "quicklook_path_hmi")
    make_paths_relative(filtered_cat, "path_image_cutout_mdi")
    make_paths_relative(filtered_cat, "quicklook_path_mdi")

    # Make train / test split
    mag_train, mag_test = grouped_stratified_split(
        filtered_cat_df, group_col="number", class_col="magnetic_class", train_size=0.8, test_size=0.2
    )
    hale_train, hale_test = grouped_stratified_split(
        filtered_cat_df, group_col="number", class_col="mcintosh_class", train_size=0.8, test_size=0.2
    )

    mag_train_cat = filtered_cat[mag_train]
    # mag_test_cat = filtered_cat[mag_test]

    hale_train_cat = filtered_cat[hale_train]
    # hale_test_cat = filtered_cat[hale_test]

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
    # def make_paths_relative(cat, col):
    #     relpahts = ["/".join(p.split("/")[-2:]) for p in cat[col]]
    #     cat[col] = relpahts
    #
    # make_paths_relative(extraction_cat, "processed_path")

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
