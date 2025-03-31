from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import sunpy.map
from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold
from sunpy.map import Map

import astropy.units as u
from astropy.io.fits import CompImageHDU
from astropy.table import Table
from astropy.time import Time

from arccnet.utils.logging import logger

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

    if "BLANK" in amap.meta:
        del amap.meta["BLANK"]

    if "bscale" in amap.meta:
        del amap.meta["bscale"]

    if "bzero" in amap.meta:
        del amap.meta["bzero"]

    amap.save(path, **kwargs)  # , hdu_type=astropy.io.fits.CompImageHDU, **kwargs)


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


def time_split(df: DataFrame, *, time_column: str, train: int, test: int, validate: int):
    r"""
    Split data set based on time

    First t 'train' days go to train, next v 'validate' days go to validate, and finally next t 'test' days go
    to test, this process is repeated until all data is assigned. The concept here is the validation data acts as a
    buffer to between the test and train set

    Parameters
    ----------
    df :
        Input dataset
    time_column :
        Name of column containing time data to use for splitting
    train :
        Number of consecutive days to assign to train set
    test
        Number of consecutive days to assign to test set
    validate
        Number of consecutive days to assign to validation set

    Returns
    -------

    """
    full_cycle = train + test + validate

    duration = df[time_column].max() - df[time_column].min()
    full_cycles, remain = divmod(duration.days, full_cycle)

    train_indices = []
    test_indices = []
    val_indices = []

    train_start = df[time_column].min()

    for cycle in range(full_cycles + 1):
        train_end = train_start + pd.DateOffset(days=train - 1)
        val_start = train_end + pd.DateOffset(days=1)
        val_end = val_start + pd.DateOffset(days=validate - 1)
        test_start = val_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(days=test - 1)

        train_indices.append(df[df[time_column].between(train_start, train_end)].index.values)
        val_indices.append(df[df[time_column].between(val_start, val_end)].index.values)
        test_indices.append(df[df[time_column].between(test_start, test_end)].index.values)

        train_start = test_end + pd.DateOffset(days=1)

    train_indices = np.hstack(train_indices)
    test_indices = np.hstack(test_indices)
    val_indices = np.hstack(val_indices)

    return train_indices, test_indices, val_indices


def _get_info(path):
    map = Map(path)
    info = {
        "Observatory": map.observatory,
        "Instrument": map.instrument,
        "Detector": map.detector,
        "Measurement": map.measurement,
        "Wavelength": map.wavelength,
        "Date": map.date,
        "Path": path,
    }
    del map
    return info


def build_local_files(path):
    r"""
    Build local fits file table based on recursive search from given path.


    Parameters
    ----------
    path : `str` or `pathlib.Path`
        Directory to start search from

    Returns
    -------
    fits_file_info : `astropy.table.Table`
        Table of fits file info
    """
    root = Path(path)
    files = root.rglob("*.fits")

    with ProcessPoolExecutor() as executor:
        file_info = executor.map(_get_info, files, chunksize=1000)

    fits_file_info = Table(list(file_info))
    return fits_file_info


def reproject(input_map: Map, traget_map: Map, **reproject_kwargs) -> Map:
    r"""
    Reproject the input map to the target map WCS.

    Copy back over some import meta information removed when using reproject.

    Parameters
    ----------
    input_map
        Input map to reproject
    traget_map
        Target map

    Returns
    -------

    """
    output_map = input_map.reproject_to(traget_map.wcs, **reproject_kwargs)
    # Created by manually looking at the headers and removing anything related to WCS, dates, data values and observer
    # fmt: off
    KEY_MAP  = {('SDO', 'AIA'): {'bld_vers', 'lvl_num', 'trecstep', 'trecepoc', 'trecroun', 'origin', 'telescop',
                                 'instrume', 'camera', 'img_type', 'exptime', 'expsdev', 'int_time', 'wavelnth',
                                 'waveunit', 'wave_str', 'fsn', 'fid', 'quallev0', 'quality', 'flat_rec', 'nspikes',
                                 'mpo_rec', 'inst_rot', 'imscl_mp', 'x0_mp', 'y0_mp', 'asd_rec', 'sat_y0', 'sat_z0',
                                 'sat_rot', 'acs_mode', 'acs_eclp', 'acs_sunp', 'acs_safe', 'acs_cgt', 'orb_rec',
                                 'roi_sum', 'roi_nax1', 'roi_nay1', 'roi_llx1', 'roi_lly1', 'roi_nax2', 'roi_nay2',
                                 'roi_llx2', 'roi_lly2', 'pixlunit', 'dn_gain', 'eff_area', 'eff_ar_v', 'tempccd',
                                 'tempgt', 'tempsmir', 'tempfpad', 'ispsname', 'isppktim', 'isppktvn', 'aivnmst',
                                 'aimgots', 'asqhdr', 'asqtnum', 'asqfsn', 'aiahfsn', 'aecdelay', 'aiaecti', 'aiasen',
                                 'aifdbid', 'aimgotss', 'aifcps', 'aiftswth', 'aifrmlid', 'aiftsid', 'aihismxb',
                                 'aihis192', 'aihis348', 'aihis604', 'aihis860', 'aifwen', 'aimgshce', 'aectype',
                                 'aecmode', 'aistate', 'aiaecenf', 'aifiltyp', 'aimshobc', 'aimshobe', 'aimshotc',
                                 'aimshote', 'aimshcbc', 'aimshcbe', 'aimshctc', 'aimshcte', 'aicfgdl1', 'aicfgdl2',
                                 'aicfgdl3', 'aicfgdl4', 'aifoenfl', 'aimgfsn', 'aimgtyp', 'aiawvlen', 'aiagp1',
                                 'aiagp2', 'aiagp3', 'aiagp4', 'aiagp5', 'aiagp6', 'aiagp7', 'aiagp8', 'aiagp9',
                                 'aiagp10', 'agt1svy', 'agt1svz', 'agt2svy', 'agt2svz', 'agt3svy', 'agt3svz',
                                 'agt4svy', 'agt4svz', 'aimgshen', 'keywddoc', 'recnum', 'blank', 'drms_id',
                                 'primaryk', 'comment', 'history', 'keycomments'},
                ('SDO', 'HMI'): {'telescop', 'instrume', 'wavelnth', 'camera', 'bunit', 'origin', 'content', 'quality',
                                 'quallev1', 'bld_vers', 'hcamid', 'source', 'trecepoc', 'trecstep', 'trecunit',
                                 'cadence', 'datasign', 'hflid', 'hcftid', 'qlook', 'cal_fsn', 'lutquery', 'tsel',
                                 'tfront', 'tintnum', 'sintnum', 'distcoef', 'rotcoef', 'odicoeff', 'orocoeff',
                                 'polcalm', 'codever0', 'codever1', 'codever2', 'codever3', 'calver64', 'recnum',
                                 'blank', 'checksum', 'datasum', 'waveunit', 'detector', 'history', 'comment',
                                 'keycomments'},
    }
    # fmt: on
    keys = KEY_MAP.get((input_map.observatory, input_map.detector), [])
    meta_to_update = {key: input_map.meta[key] for key in keys}
    output_map.meta.update(meta_to_update)
    return output_map


def round_to_daystart(time: Time) -> Time:
    r"""
    Round time to given interval start of day


    Parameters
    ----------
    time :
        Times to round
    interval :
        Interval

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> time = Time('2000-01-01')
    >>> times = time + [0, 6, 12, 18, 24, 30, 36, 48] *u.h
    <Time object: scale='utc' format='isot' value=['2000-01-01T01:00:00.000' '2000-01-01T02:00:00.000'
    '2000-01-01T03:00:00.000' '2000-01-01T04:00:00.000'
    '2000-01-01T05:00:00.000' '2000-01-01T06:00:00.000'
    '2000-01-01T07:00:00.000' '2000-01-01T08:00:00.000'
    '2000-01-01T09:00:00.000' '2000-01-01T10:00:00.000'
    Returns
    -------

    """
    day_start = Time(time.strftime("%Y-%m-%d"))
    next_day = day_start + 1 * u.day
    diff = time - day_start
    diff.to_value(u.hour)
    rounded = np.where(diff.to_value(u.hour) < 12, day_start, next_day)
    return Time(rounded)


### Code run to patch up current data needs to be integrated into pipeline


def _aia_to_hmi(aia_path, hmi_map, hmi_path):
    aia_path = Path(aia_path)
    aia_map = Map(aia_path)
    print("!!!!!!!!!!!!!!!!!!!! reprojecting !!!!!!!!!!!!!!!!")
    aia_repro = reproject(aia_map, hmi_map)
    print("!!!!!!!!!!!!!!!!!!!! done reprojecting !!!!!!!!!!!!!!!!")
    aia_repro.meta["repotar"] = str(hmi_path.name)
    outpath = hmi_path.parent.parent.parent.parent / "euv" / "fits" / "aia"
    outpath.mkdir(parents=True, exist_ok=True)
    outfile = outpath / f"{aia_path.stem}_reprojected.fits"
    print(f"!!!!!!!!!!!!!!!!!!!! Output file !!!!!!!!!!!!!!!!: {outfile}")
    aia_repro.save(outfile, hdu_type=CompImageHDU)


def reproject_aia_to_hmi(hmi_paths, fits_info, executor):
    for path in list(hmi_paths):
        print(f"HMI path {path}")
        hmi_path = Path("/mnt/ARCAFF/v0.2.2/ARCCnet/" + path)
        hmi_map = Map(hmi_path)
        day = round_to_daystart(hmi_map.date)
        print(day)
        fits = fits_info[fits_info["Day"] == day]
        for row in fits[fits["Detector"] == "AIA"]:
            print(f"AIA and HMI paths: {row['Path_str']}, {hmi_path}")
            executor.submit(_aia_to_hmi, row["Path_str"], hmi_map, hmi_path)


# with ProcessPoolExecutor(max_workers=32) as executor:
#   reproject_aia_to_hmi(hmi_paths, sdo_only, executor)
