from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, YearLocator
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, transform_with_sun_center
from sunpy.map import Map, make_fitswcs_header

import astropy.units as u
from astropy.coordinates import SkyCoord

rng = np.random.RandomState(1338)

__all__ = [
    "visualise_groups_classes",
    "visualise_data_split",
    "plot_set_distributions",
    "plot_srs_coverage",
    "plot_srs_map",
    "plot_filtered_srs_trace",
]


def visualise_groups_classes(df, *, class_col, group_col, axes=None):
    r"""
    Visualise distribution of groups and classes.

    Inspired by sklearn https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data
    class_col : `str`
        Name of column contain classes
    group_col
        Name of column containing groups
    Returns
    -------

    """
    if axes is None:
        fig, axes = plt.subplots()

    groups = df[group_col].astype("category").cat.codes
    classes = df[class_col].astype("category").cat.codes

    # Visualize dataset groups
    axes.scatter(range(len(groups)), [4.5] * len(groups), c=groups % 20, marker="_", lw=50, cmap="tab20")
    axes.scatter(range(len(groups)), [2.5] * len(groups), c=classes, marker="_", lw=50, cmap="Paired")

    axes.scatter(
        range(len(groups)), [0.5] * len(groups), c=df["datetime"] - df["datetime"][0], marker="_", lw=50, cmap="viridis"
    )
    axes.set(
        ylim=[-1, 6],
        yticks=[0.5, 2.5, 4.5],
        yticklabels=["Time", "Class", "Number"],
        xlabel="Sample index",
    )


def visualise_data_split(df, *, train_idxs, test_idxs, class_col, group_col, axes=None, lw=10):
    r"""

    Parameters
    ----------
    df
    train_idxs
    test_idxs
    class_col
    group_col
    ax
    lw

    Returns
    -------

    """
    if axes is None:
        axes = plt.gca()

    groups = df[group_col].astype("category").cat.codes
    classes = df[class_col].astype("category").cat.codes

    # Fill in indices with the training/test groups
    indices = np.array([np.nan] * len(df))
    indices[train_idxs] = 1
    indices[test_idxs] = 0

    # Visualize the results
    axes.scatter(
        range(len(indices)),
        [0.5] * len(indices),
        c=indices,
        marker="_",
        lw=lw,
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.2,
    )

    # Plot the data classes and groups at the end
    axes.scatter(range(len(df)), [1.5] * len(df), c=classes, marker="_", lw=lw, cmap="Paired")

    axes.scatter(range(len(df)), [2.5] * len(df), c=groups % 20, marker="_", lw=lw, cmap="tab20")

    axes.scatter(
        range(len(groups)), [3.5] * len(groups), c=df["datetime"] - df["datetime"][0], marker="_", lw=lw, cmap="viridis"
    )

    # Formatting
    yticklabels = ["Split", "Class", "Number", "Time"]
    axes.set(
        yticks=np.arange(1 + 3) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylim=[1 + 3.2, -0.2],
        xlim=[-500, len(df) + 500],
        title=class_col,
    )
    return axes


def plot_set_distributions(df, *, train_idxs, test_idxs, class_col):
    r"""
    Plot class distributions for all data, train, and test sets.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data
    train_idxs : `list` or `np.ndarray[int]`
        Train indices
    test_idxs : `list` or `np.ndarray[int]`
        Test indices
    class_col : `str`:
        Name of column containing classes

    Returns
    -------
    `Axes`
        Axes containing the plots of the distributions

    """
    dists_df = pd.concat(
        [
            df[class_col].value_counts(normalize=True),
            df[class_col].iloc[train_idxs].value_counts(normalize=True),
            df[class_col].iloc[test_idxs].value_counts(normalize=True),
        ],
        axis=1,
    )
    dists_df.columns = ["Original", "Train", "Test"]
    ax = dists_df.plot(kind="bar")
    ax.set_yscale("log")
    ax.set_title(class_col)


def plot_srs_coverage(processed_catalog, figsize=(6, 4), dpi=300):
    pcat_df = processed_catalog.to_pandas()
    min_date = pcat_df.time.min()
    max_date = pcat_df.time.max()
    srs_coverage = np.full((max_date.year - min_date.year + 1, 366), -1)

    for g, v in pcat_df.groupby("time"):
        if str(v["path"].values[0]) == ".":
            val = 0

        else:
            if v["loaded_successfully"].values[0] == True:  # noqa: E712
                val = 2
            else:
                val = 1

        # -1 default, 0 no file found,  1 parse error, 2 ok
        srs_coverage[g.year - min_date.year, g.day_of_year - 1] = val
    fig, axes = plt.subplots(1, 1)
    cm = plt.get_cmap("viridis", 3)
    cm.set_under("white")
    mat = axes.imshow(srs_coverage, cmap=cm, aspect="auto", interpolation="none", vmin=0)
    axes.set_yticks(range(max_date.year - min_date.year + 1), range(min_date.year, max_date.year + 1))
    axes.set_xlabel("Day of Year")
    axes.set_ylabel("Year")
    axes.set_title(f"SRS Coverage {min_date.date()} to {max_date.date()}")
    cax = fig.colorbar(mat, ticks=[0.33, 1, 1.66])
    cax.set_ticklabels(["Missing", "Parse Error", "Ok"])
    return fig, axes


def plot_srs_map(processed_catalog, filtered_only=False, figsize=(6, 4), dpi=300):
    pcat_df = processed_catalog.to_pandas()

    all_ars_mask = pcat_df.id == "I"
    ars = pcat_df[all_ars_mask]
    if filtered_only:
        ars = ars[ars.filtered == True]  # noqa: E712

    data = np.full((10, 10), np.nan)

    # Define a reference coordinate and create a header using sunpy.map.make_fitswcs_header
    hgs_zz = SkyCoord(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime="2013-12-06", frame=HeliographicStonyhurst)
    skycoord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime="2013-12-06", observer=hgs_zz, frame=Helioprojective)

    # Scale set to the following for solar limb to be in the field of view
    header = make_fitswcs_header(data, skycoord, scale=[220, 220] * u.arcsec / u.pixel)

    # Use sunpy.map.Map to create the blank map
    blank_map = Map(data, header)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(projection=blank_map)
    blank_map.plot(axes=ax)
    blank_map.draw_limb(axes=ax, color="k", linewidth=0.1)
    blank_map.draw_grid(axes=ax, color="k", linewidth=0.1)

    coords = SkyCoord(
        ars.longitude.values * u.deg, ars.latitude.values * u.deg, obstime=["2013-12-06"], frame=HeliographicStonyhurst
    )

    vis = np.abs(coords.lon) < 90 * u.deg

    with transform_with_sun_center():
        # coords = SkyCoord([-100, 100]*u.deg, [0]*7*u.deg,
        #               obstime=['2013-12-06']*7, frame=frames.HeliographicStonyhurst)
        ax.scatter_coord(coords[vis], s=0.5, edgecolor="none", label="Front", alpha=0.25)
        ax.scatter_coord(coords[~vis], s=0.5, color="r", edgecolor="none", label="Rear", alpha=0.25)

    ax.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    ax.tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    ax.set_title("SRS 1996-01-01 - 2022-12-31", fontsize=8)
    ax.legend(fontsize=6, frameon=False, markerscale=4)

    return fig, ax


def plot_filtered_srs_trace(processed_catalog, numbers=None, figsize=(6, 4), dpi=300):
    if numbers is None:
        numbers = dict()

    pcat_df = processed_catalog.to_pandas()

    all_ars_mask = pcat_df.id == "I"
    ars = pcat_df[all_ars_mask]

    data = np.full((10, 10), np.nan)

    # Define a reference coordinate and create a header using sunpy.map.make_fitswcs_header
    hgs_zz = SkyCoord(0 * u.deg, 0 * u.deg, 1 * u.AU, obstime="2013-12-06", frame=HeliographicStonyhurst)
    skycoord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime="2013-12-06", observer=hgs_zz, frame=Helioprojective)

    # Scale set to the following for solar limb to be in the field of view
    header = make_fitswcs_header(data, skycoord, scale=[220, 220] * u.arcsec / u.pixel)

    # Use sunpy.map.Map to create the blank map
    blank_map = Map(data, header)

    fig = plt.figure()
    ax = fig.add_subplot(projection=blank_map)
    blank_map.plot(axes=ax)
    blank_map.draw_limb(axes=ax, color="k", linewidth=0.1)
    blank_map.draw_grid(axes=ax, color="k", linewidth=0.1)

    for number, label in numbers.items():
        mask = ars.number == number

        coords = SkyCoord(
            ars[mask].longitude.values * u.deg,
            ars[mask].latitude.values * u.deg,
            obstime=["2013-12-06"],
            frame=HeliographicStonyhurst,
        )

        with transform_with_sun_center():
            # coords = SkyCoord([-100, 100]*u.deg, [0]*7*u.deg,
            #               obstime=['2013-12-06']*7, frame=frames.HeliographicStonyhurst)
            ax.plot_coord(coords, marker=".", label=f"{number} {label}")

    ax.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    ax.tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    ax.set_title("SRS 1996-01-01 - 2022-12-31", fontsize=8)
    ax.legend(fontsize=6, frameon=False, markerscale=2, loc="upper right")

    return fig, ax


def plot_hmi_mdi_availability(
    hmi_table: pd.DataFrame,
    mdi_table: pd.DataFrame,
    start_time: datetime = datetime(1995, 1, 1),
    end_time: datetime = datetime.now(),
    **kwargs,
):
    """
    given HMI and MDI `DataFrame` objects, plot availability from `start_time` until `end_time`.
    """

    hmi_availability = hmi_table[["target_time", "url"]]
    mdi_availability = mdi_table[["target_time", "url"]]

    fig, ax = plt.subplots(figsize=(8, 2), **kwargs)

    # Decide where to plot the line in y space, here: "1"
    hmis = [0.8 if not pd.isna(url) else np.nan for url in hmi_availability["url"]]
    mdis = [1.2 if not pd.isna(url) else np.nan for url in mdi_availability["url"]]

    ax.scatter(mdi_availability["target_time"], mdis, s=10**2, marker="|", color="red", label="SoHO/MDI")
    ax.scatter(hmi_availability["target_time"], hmis, s=10**2, marker="|", color="blue", label="SDO/HMI")

    # Set some limits on the y-axis
    ax.set_ylim(0, 2)
    ax.set_xlim(start_time, end_time)

    # Remove y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Set a fixed locator for the x-axis (yearly ticks)
    years = plt.matplotlib.dates.YearLocator(base=1)
    ax.xaxis.set_major_locator(years)

    # Use a fixed formatter for the x-axis
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))

    # Rotate x-axis labels for better readability
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(40)

    # Add a legend
    ax.legend()

    return fig, ax


def plot_col_scatter(
    df_list: list[pd.DataFrame],
    column: str,
    start_time: datetime = datetime(1995, 1, 1),
    end_time: datetime = datetime.now(),
    colors: list = ["red", "blue"],
):
    fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)  # Create 2 rows with a shared x-axis

    # Create a YearLocator and DateFormatter to set x-axis ticks to every year
    year_locator = YearLocator()
    year_formatter = DateFormatter("%Y")

    for i, df in enumerate(df_list):
        axs[i].scatter(df["target_time"], df[column], marker=".", s=0.5, color=colors[i])
        axs[i].set_xlim(start_time, end_time)

        # Customize y-axis labels (optional)
        axs[i].set_ylabel(f"{column}")

        # Set x-axis locator and formatter
        axs[i].xaxis.set_major_locator(year_locator)
        axs[i].xaxis.set_major_formatter(year_formatter)

        # Rotate x-axis labels for better readability
        for label in axs[i].get_xticklabels():
            label.set_ha("center")
            label.set_rotation(40)

    plt.subplots_adjust(hspace=0.1)  # Adjust vertical spacing between subplots

    return fig, axs


def plot_col_scatter_single(
    df_list: list[pd.DataFrame],
    column: str,
    start_time: datetime = datetime(1995, 1, 1),
    end_time: datetime = datetime.now(),
    colors: list = ["red", "blue"],
):
    fig, ax = plt.subplots(figsize=(8, 2))

    # Scatter plot for the first row
    for i, df in enumerate(df_list):
        ax.scatter(df["target_time"], df[column], marker=".", s=0.1, color=colors[i])

    ax.set_ylabel(f"{column}")

    # Set a fixed locator for the x-axis (yearly ticks)
    years = plt.matplotlib.dates.YearLocator(base=1)
    ax.xaxis.set_major_locator(years)

    # Use a fixed formatter for the x-axis
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))

    # Rotate x-axis labels for better readability
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(40)

    ax.set_xlim(start_time, end_time)

    return fig, ax


def plot_map(map_one, figsize=(5, 4)):
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 1, 1, projection=map_one)
    map_one.plot(cmap="hmimag")

    vmin, vmax = -1499, 1499
    map_one.plot_settings["norm"].vmin = vmin
    map_one.plot_settings["norm"].vmax = vmax

    return fig, ax1


def plot_maps(map_one, map_two, figsize=(10, 4)):
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 2, 1, projection=map_one)
    map_one.plot(cmap="hmimag")

    ax2 = fig.add_subplot(1, 2, 2, projection=map_two)
    map_two.plot(cmap="hmimag")

    vmin, vmax = -1499, 1499
    map_one.plot_settings["norm"].vmin = vmin
    map_one.plot_settings["norm"].vmax = vmax
    map_two.plot_settings["norm"].vmin = vmin
    map_two.plot_settings["norm"].vmax = vmax

    return fig, [ax1, ax2]


def plot_maps_regions(map_one, regions_one, map_two, regions_two, **kwargs):
    fig = plt.figure(figsize=(10, 4))

    # Assign different projections to each subplot
    ax0 = fig.add_subplot(1, 2, 1, projection=map_one)
    ax1 = fig.add_subplot(1, 2, 2, projection=map_two)

    # Set the colormap limits for both maps
    vmin, vmax = -1499, 1499
    map_one.plot_settings["norm"].vmin = vmin
    map_one.plot_settings["norm"].vmax = vmax
    map_two.plot_settings["norm"].vmin = vmin
    map_two.plot_settings["norm"].vmax = vmax

    # Plot HMI and MDI maps on the respective subplots
    map_one.plot(axes=ax0, cmap="hmimag")
    map_two.plot(axes=ax1, cmap="hmimag")

    # Loop through region_table and draw quadrangles for both maps
    for row in regions_one:
        print(row["bottom_left_cutout"])
        map_one.draw_quadrangle(row["bottom_left_cutout"], axes=ax0, top_right=row["top_right_cutout"], **kwargs)
    for row in regions_two:
        print(row["bottom_left_cutout"])
        map_two.draw_quadrangle(row["bottom_left_cutout"], axes=ax1, top_right=row["top_right_cutout"], **kwargs)

    return fig, [ax0, ax1]
