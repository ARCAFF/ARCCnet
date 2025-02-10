import numpy as np
import pandas as pd
import pytest

import arccnet.models.dataset_utils as ut_d
from arccnet.models import labels


@pytest.fixture
def sample_dataframe():
    data = {
        "processed_path_image_hmi": ["path1", "", "path3", "path4", "", "path6"],
        "longitude_hmi": [30, -70, 45, 80, -50, 50],
        "longitude_mdi": [np.nan, 60, np.nan, 90, np.nan, 60],
        "label": ["A", "B", "C", "A", "B", "C"],
    }
    return pd.DataFrame(data)


def test_undersample_group_filter(monkeypatch, sample_dataframe):
    """
    Test the `undersample_group_filter` function for correct location assignment, label mapping and encoding,
    proper undersampling of the majority class, and filtering of 'rear' locations.

    This test verifies that:
        1. Locations are accurately assigned as 'front' or 'rear' based on longitude thresholds.
        2. Original labels ('A', 'B', 'C') are correctly mapped to grouped labels ('group1', 'group2').
        3. Grouped labels are properly encoded into numerical indices.
        4. The majority class ('group1') is undersampled to match the size of the minority class ('group2') plus a buffer.
        5. Rows with 'rear' locations are excluded from the undersampled DataFrame.
        6. When undersampling is disabled, all 'front' locations are retained without altering class distribution.

    Steps:
        1. Setup:
            - Define a label mapping where 'A' and 'B' map to 'group1', and 'C' maps to 'group2'.
            - Mock the `labels.LABEL_TO_INDEX` dictionary to ensure consistent encoding (`{'group1': 0, 'group2': 1}`).

        2. Undersampling Enabled (`undersample=True`):
            - Invoke `undersample_group_filter` with the sample DataFrame.
            - Assert that each row is assigned the correct location ('front' or 'rear').
            - Verify that labels are correctly mapped to grouped labels.
            - Ensure that grouped labels are accurately encoded.
            - Confirm that the majority class ('group1') is undersampled to 2 instances.
            - Check that the minority class ('group2') retains 2 instances.
            - Ensure that all 'rear' locations are excluded from the undersampled DataFrame.

        3. Undersampling Disabled (`undersample=False`):
            - Invoke `undersample_group_filter` with undersampling turned off.
            - Assert that only 'front' locations are retained.
            - Verify that the number of 'front' rows matches the expected count (5 rows).

    Assertions:
        - Location Assignment: Validates that the 'location' column matches the expected list of locations.
        - Label Mapping: Ensures that the 'grouped_labels' column accurately reflects the label mapping.
        - Label Encoding: Confirms that the 'encoded_labels' column correctly encodes the grouped labels.
        - Undersampling Counts: Checks that the undersampled DataFrame has the correct number of instances for each group.
        - Filtering Rear Locations: Verifies that no 'rear' locations exist in the undersampled DataFrame.
        - Row Count Without Undersampling: Ensures the correct number of 'front' rows are present when undersampling is disabled.
    """
    label_mapping = {"A": "group1", "B": "group1", "C": "group2"}

    # Mock labels.LABEL_TO_INDEX to align with expected encoded labels
    mock_label_to_index = {"group1": 0, "group2": 1}
    monkeypatch.setattr(labels, "LABEL_TO_INDEX", mock_label_to_index)

    # Call the function with undersampling enabled
    df_modified, df_undersampled = ut_d.undersample_group_filter(
        df=sample_dataframe.copy(),
        label_mapping=label_mapping,
        long_limit_deg=60,
        undersample=True,
        buffer_percentage=0.1,
    )

    # Corrected expected_locations
    expected_locations = ["front", "front", "front", "rear", "front", "front"]
    assert all(df_modified["location"] == expected_locations), "Location assignment incorrect."

    # Test label mapping
    expected_grouped_labels = ["group1", "group1", "group2", "group1", "group1", "group2"]
    assert all(df_modified["grouped_labels"] == expected_grouped_labels), "Label mapping incorrect."

    # Test label encoding
    expected_encoded_labels = [0, 0, 1, 0, 0, 1]
    assert all(df_modified["encoded_labels"] == expected_encoded_labels), "Label encoding incorrect."

    # Test undersampling
    #    Original group counts: group1 has 3, group2 has 2
    #    Second largest class count is 2 (group2), buffer is 0.1 -> n_samples = 2 * 1.1 = 2.2 -> 2
    #    So group1 should be undersampled to 2 samples
    group1_undersampled_count = df_undersampled[df_undersampled["grouped_labels"] == "group1"].shape[0]
    group2_count = df_undersampled[df_undersampled["grouped_labels"] == "group2"].shape[0]
    assert group1_undersampled_count == 2, "Undersampling of majority class incorrect."
    assert group2_count == 2, "Minority class count incorrect."

    # Test that 'rear' locations are filtered out in the undersampled dataframe
    assert all(df_undersampled["location"] == "front"), "'rear' locations were not properly filtered out."

    # Call the function with undersampling disabled
    df_modified_no_undersample, df_no_undersample = ut_d.undersample_group_filter(
        df=sample_dataframe.copy(),
        label_mapping=label_mapping,
        long_limit_deg=60,
        undersample=False,
        buffer_percentage=0.1,
    )

    # Test that all front locations are present without undersampling
    expected_front_indices = df_modified_no_undersample[df_modified_no_undersample["location"] == "front"].index
    assert set(df_no_undersample.index) == set(expected_front_indices), "Filtering without undersampling incorrect."

    # Test that all front locations are present
    assert df_no_undersample.shape[0] == 5, "Incorrect number of rows after filtering without undersampling."
