import os
import tempfile
from itertools import chain

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from arccnet.data_generation.utils.utils import check_column_values, grouped_stratified_split, save_df_to_html


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "ID": ["I", "I", "II"],
            "Value": [10, 20, 30],
        }
    )


@pytest.fixture
def valid_values():
    return {
        "ID": ["I", "II"],
        "Value": [10, 20, 30],
    }


# save_df_to_html


def test_save_df_to_html(sample_dataframe):
    with tempfile.NamedTemporaryFile(delete=False) as file:
        save_df_to_html(sample_dataframe, file.name)
        assert os.path.isfile(file.name)

    # remove temporary file
    os.remove(file.name)
    assert not os.path.isfile(file.name)


def test_save_df_to_html_invalid_filename(sample_dataframe):
    with pytest.raises(ValueError):
        save_df_to_html(sample_dataframe, 123)


def test_save_df_to_html_invalid_dataframe():
    with pytest.raises(ValueError):
        save_df_to_html("not a dataframe", "test.html")


# check_column_values


def test_check_column_values(sample_dataframe, valid_values):
    result = check_column_values(sample_dataframe, valid_values, return_catalog=False)
    assert result is None


def test_check_column_values_invalid_columns(sample_dataframe):
    valid_values = {"InvalidColumn": [1, 2, 3]}
    with pytest.raises(ValueError):
        check_column_values(sample_dataframe, valid_values, return_catalog=False)


def test_check_column_values_invalid_values(sample_dataframe, caplog):
    import logging

    valid_values = {"ID": ["Invalid"]}
    with caplog.at_level(logging.ERROR):  # Set the log level to capture ERROR level logs
        check_column_values(sample_dataframe, valid_values, return_catalog=False)

    assert (
        "Invalid `ID`; `ID` = ['I', 'II']" in caplog.text
    )  # Check if the log output contains the expected error message


# !TODO throw a ValueError if the SRS values for Mcintosh classes are invalid
# def test_check_column_values_invalid_values(sample_dataframe):
#     with pytest.raises(ValueError):
#         valid_values = {"ID": ["Invalid"]}
#         check_column_values(sample_dataframe, valid_values, return_catalog=False)


def test_check_column_values_return_catalog(sample_dataframe, valid_values):
    result = check_column_values(sample_dataframe, valid_values, return_catalog=True)
    assert isinstance(result, pd.DataFrame)
    # Additional assertions on the returned catalog


@pytest.mark.parametrize(
    "classes",
    ([1] * 10 + [2] * 40 + [2] * 200 + [4] * 750, ["a"] * 10 + ["b"] * 40 + ["c"] * 200 + ["d"] * 750),
    ids=["numeric", "strings"],
)
def test_grouped_stratified(classes):
    groups = list(chain.from_iterable([[i] * 10 for i in range(100)]))
    np.random.seed(42)
    np.random.shuffle(classes)
    data = list(zip(classes, groups))
    cls_col = "class"
    grp_col = "group"
    df = pd.DataFrame(data, columns=[cls_col, grp_col])

    train_indices, test_indices = grouped_stratified_split(df, class_col=cls_col, group_col=grp_col, random_state=42)

    # make sure no group appears in both sets
    train_groups = df[grp_col].iloc[train_indices]
    test_groups = df[grp_col].iloc[test_indices]
    groups_intersection = set(train_groups).intersection(set(test_groups))
    assert len(groups_intersection) == 0

    # make sure distribution of sets classes are close to dist of original
    class_dist = df[cls_col].value_counts(normalize=True)
    train_class_dist = df[cls_col].iloc[train_indices].value_counts(normalize=True)
    test_class_dist = df[cls_col].iloc[test_indices].value_counts(normalize=True)
    assert_allclose(class_dist, train_class_dist, atol=0.02)  # 2% error
    assert_allclose(class_dist, test_class_dist, atol=0.02)  # 2% error

    # make sure all data is used
    assert_array_equal(np.arange(1000), np.sort(np.hstack([train_indices, test_indices])))
