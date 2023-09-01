from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import sunpy.map
from sklearn.model_selection import StratifiedGroupKFold

import astropy.io

from arccnet.data_generation.utils.data_logger import logger

__all__ = [
    "is_point_far_from_point",
    "make_relative",
    "save_compressed_map",
    "round_to_midnight",
    "save_df_to_html",
    "check_column_values",
    "grouped_stratified_split",
]


def is_point_far_from_point(x, y, x1, y1, threshold_x, threshold_y):
    return abs(x - x1) > abs(threshold_x) or abs(y - y1) > abs(threshold_y)


def make_relative(base_path, path):
    return Path(path).relative_to(Path(base_path))


def save_compressed_map(amap: sunpy.map.Map, path: Path, **kwargs) -> None:
    """
    Save a compressed map.
    If "bscale" and "bzero" exist in the metadata, remove before saving.
    See: https://github.com/sunpy/sunpy/issues/7139

    Parameters
    ----------
    amap : sunpy.map.Map
        the sunpy map object to be saved
    path : Path
        the path to save the file to

    Returns
    -------
    None
    """
    if "bscale" in amap.meta:
        del amap.meta["bscale"]

    if "bzero" in amap.meta:
        del amap.meta["bzero"]

    amap.save(path, hdu_type=astropy.io.fits.CompImageHDU, **kwargs)


def round_to_midnight(dt: datetime):
    # Calculate the next midnight
    next_midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    # Calculate the previous midnight
    previous_midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # Calculate time differences
    time_to_next_midnight = next_midnight - dt
    time_to_previous_midnight = dt - previous_midnight

    # Compare time differences and round to the closest midnight
    if time_to_next_midnight < time_to_previous_midnight:
        return next_midnight
    else:
        return previous_midnight


def save_df_to_html(df: pd.DataFrame, filename: str) -> None:
    """
    Save the provided `df` to an HTML file with the specified `filename`.

    Parameters
    ----------
    df : `pandas.DataFrame`
        a `pandas.DataFrame` to save to the HTML file

    filename : str
        the HTML filename

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `filename` is not a string or `df` is not a DataFrame

    """

    if not isinstance(filename, str):
        raise ValueError("The `filename` must be a string")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The provided object is not a `pandas.DataFrame`")

    with open(filename, "w") as file:
        file.write(df.to_html())


def check_column_values(catalog: pd.DataFrame, valid_values: dict, return_catalog=True) -> pd.DataFrame:
    """
    Check column values against known (valid) values.

    First check if the columns in `valid_values` are present in the
    `catalog` DataFrame and verify that the corresponding values in those
    columns match the known valid values.

    Parameters
    ----------
    catalog : pandas.DataFrame
        a `pandas.DataFrame` that contains a set of columns

    valid_values : dict
        a dictionary containing the column names and valid values.
        The dictionary keys must be a subset of the `catalog.columns`

    return_catalog : bool
        return the catalog? Default is True

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any columns in `valid_values` are not present in the `catalog`.

    Examples
    --------
    >>> catalog = pd.DataFrame({'ID': ['I', 'I', 'II'], 'Value': [10, 20, 30]})
    >>> valid_values = {'ID': ['I', 'II'], 'Value': [10, 20, 30]}
    >>> check_column_values(catalog, valid_values, return_catalog=False)
    """

    # Check that the columns in `valid_values` are in `catalog``
    invalid_columns = set(valid_values.keys()) - set(catalog.columns)
    if invalid_columns:
        raise ValueError(f"Columns {list(invalid_columns)} in `valid_values` are not present in `catalog`.")

    # Checking values against the `valid_values`
    for col, vals in valid_values.items():
        result = catalog[col].isin(vals)
        invalid_vals = catalog.loc[~result, col].unique().tolist()
        if invalid_vals:
            msg = f"Invalid `{col}`; `{col}` = {invalid_vals}"
            logger.error(msg)
            # raise ValueError(msg) # !TODO reinstate ValueError

    # if catalog["ID"].nunique() != 1 or catalog["ID"].unique()[0] != "I":
    #     raise ValueError("Invalid 'ID' values.")

    if return_catalog:
        return catalog


def grouped_stratified_split(
    df, *, class_col, group_col, train_size=0.7, test_size=0.3, shuffle=True, random_state=None
) -> tuple[np.ndarray[int], np.ndarray[int]]:
    r"""
    Return grouped stratified splits for given data with train test sizes

    Not super efficient or exact but saves time having to implement from scratch. Abuse StratifiedGroupKFold and
    n_splits to get approximately desired sizes and then only return 1st split.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data
    class_col : `str`
        Name of the column containing classes
    group_col
        Name of the column containing groups
    test_size: `float`
        Size of test set
    train_size: `float`
        Size of train set
    random_state : `int` or `RandomState` instance, default=None
        Random state info passed on to StratifiedGroupKFold
    shuffle : `boolean` default True
        If the data should be shuffled
    Returns
    -------
    Train and test indices
    """
    if train_size + test_size != 1.0:
        raise ValueError("Train and test size must sum to 1.0")

    train_index = train_size * 10
    test_index = train_size * 10
    if not train_index.is_integer() or not test_index.is_integer():
        raise ValueError("Train and test size must be given in multiples of 0.1")

    n_splits = int(1 / test_size)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    splits = list(sgkf.split(df.index.tolist(), df[class_col], df[group_col]))
    train, test = splits[0]

    return train, test
