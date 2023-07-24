import os
import tempfile
from random import shuffle
from itertools import chain

import pandas as pd
import pytest
from numpy.testing import assert_allclose

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


def test_grouped_stratified_split():
    list(range(100))
    groups = list(chain.from_iterable([[i] * 10 for i in range(10)]))
    classes = [0] * 20 + [1] * 30 + [2] * 50
    shuffle(classes)
    data = list(zip(classes, groups))
    cls_col = "class"
    grp_col = "group"
    df = pd.DataFrame(data, columns=[cls_col, grp_col])

    splits = list(grouped_stratified_split(df, cls_col, grp_col))
    train_indices, test_indices = splits[0]

    train_groups = df[grp_col].iloc[train_indices]
    test_groups = df[grp_col].iloc[test_indices]
    groups_intersection = set(train_groups).intersection(set(test_groups))
    assert len(groups_intersection) == 0

    class_dist = df[cls_col].value_counts(normalize=True)
    train_class_dist = df[cls_col].iloc[train_indices].value_counts(normalize=True)
    test_class_dist = df[cls_col].iloc[test_indices].value_counts(normalize=True)
    assert_allclose(class_dist, train_class_dist, rtol=0.1)
    assert_allclose(class_dist, test_class_dist, rtol=0.15)
