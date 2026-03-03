# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: py_3.11
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from datetime import datetime
from functools import partial

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from p_tqdm import p_map

from arccnet import load_config
from arccnet.models import dataset_utils as ut_d
from arccnet.models import preprocessing_common as pp_common
from arccnet.notebooks.analysis.EDA_utils import (
    analyze_quality_flags,
    create_solar_grid,
    load_and_analyze_fits_pair,
    process_row,
)
from arccnet.visualisation import utils as ut_v

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
config = load_config()

# %%
data_folder = "/ARCAFF/data"
dataset_folder = "arcnet-v20251017/04_final"
df_file_name = "data/cutout_classification/region_classification.parq"
dataset_title = "arccnet v20251017"

# %%
df, _, filtered_ql_df = ut_d.make_dataframe(data_folder, dataset_folder, df_file_name)
ut_v.make_classes_histogram(df["label"], figsz=(18, 6), text_fontsize=11, title=dataset_title)
plt.show()
df

# %%
mdi_color = "royalblue"
hmi_color = "tomato"

df_MDI = df[df["path_image_cutout_hmi"] == ""].copy()
df_HMI = df[df["path_image_cutout_mdi"] == ""].copy()

# %% [markdown]
# # Quality Flags

# %%
quality_mdi_df = analyze_quality_flags(df_MDI, "SOHO/MDI")
quality_mdi_df

# %%
quality_hmi_df = analyze_quality_flags(df_HMI, "SDO/HMI")
quality_hmi_df


# %% [markdown]
# ## Data Filtering

# %%
# Use shared preprocessing to keep notebook behavior aligned with training pipelines.
df_clean = ut_d.cleanup_df(df, log_level=None)
hmi_available_orig = pp_common.nonempty_path_mask(df["path_image_cutout_hmi"])
mdi_available_orig = pp_common.nonempty_path_mask(df["path_image_cutout_mdi"])
hmi_available_clean = pp_common.nonempty_path_mask(df_clean["path_image_cutout_hmi"])
mdi_available_clean = pp_common.nonempty_path_mask(df_clean["path_image_cutout_mdi"])

print("DATA FILTERING Stats")
print("-" * 40)
hmi_orig, hmi_clean = int(hmi_available_orig.sum()), int(hmi_available_clean.sum())
mdi_orig, mdi_clean = int(mdi_available_orig.sum()), int(mdi_available_clean.sum())
total_orig, total_clean = len(df), len(df_clean)
hmi_pct = (hmi_clean / hmi_orig * 100) if hmi_orig else 0.0
mdi_pct = (mdi_clean / mdi_orig * 100) if mdi_orig else 0.0
total_pct = (total_clean / total_orig * 100) if total_orig else 0.0

print(f"HMI: {hmi_clean:,}/{hmi_orig:,} ({hmi_pct:.1f}% retained)")
print(f"MDI: {mdi_clean:,}/{mdi_orig:,} ({mdi_pct:.1f}% retained)")
print(f"Total: {total_clean:,}/{total_orig:,} ({total_pct:.1f}% retained)")
print("-" * 40)


# %%
# Path Analysis
hmi_missing = ~pp_common.nonempty_path_mask(df_clean["path_image_cutout_hmi"])
mdi_missing = ~pp_common.nonempty_path_mask(df_clean["path_image_cutout_mdi"])
both_missing = hmi_missing & mdi_missing

# Count statistics
stats = {
    "total": len(df_clean),
    "hmi_none": (df_clean["path_image_cutout_hmi"] == "None").sum(),
    "hmi_empty": (df_clean["path_image_cutout_hmi"] == "").sum(),
    "mdi_none": (df_clean["path_image_cutout_mdi"] == "None").sum(),
    "mdi_empty": (df_clean["path_image_cutout_mdi"] == "").sum(),
    "both_missing": both_missing.sum(),
    "at_least_one": (~hmi_missing | ~mdi_missing).sum(),
    "both_exist": (~hmi_missing & ~mdi_missing).sum(),
}

print("PATH ANALYSIS:")
print("-" * 40)
print(f"Total rows in df_clean: {stats['total']:,}")
print(f"HMI paths - None: {stats['hmi_none']:,}, Empty: {stats['hmi_empty']:,}")
print(f"MDI paths - None: {stats['mdi_none']:,}, Empty: {stats['mdi_empty']:,}")
print(f"Both paths missing: {stats['both_missing']:,} ({stats['both_missing'] / max(stats['total'], 1) * 100:.1f}%)")
print(f"At least one path exists: {stats['at_least_one']:,}")
print(f"Both paths exist: {stats['both_exist']:,}")


# %% [markdown]
# # Location of ARs on the Sun

# %%
AR_IA_lbs = ["Alpha", "Beta", "IA", "Beta-Gamma-Delta", "Beta-Gamma", "Beta-Delta", "Gamma-Delta", "Gamma"]
AR_IA_df = df_clean[df_clean["label"].isin(AR_IA_lbs)].reset_index(drop=True)

# %%
ut_v.make_classes_histogram(AR_IA_df["label"], figsz=(12, 7), text_fontsize=11, title=f"{dataset_title} ARs", y_off=100)
plt.show()


# %%
def plot_histogram(ax, data, degree_ticks, title, color="#4C72B0"):
    """Plot histogram with degree labels."""
    rad_ticks = np.deg2rad(degree_ticks)
    ax.hist(data, bins=rad_ticks, color=color, edgecolor="black")
    ax.set_xticks(rad_ticks)
    ax.set_xticklabels([f"{deg}°" for deg in degree_ticks])
    ax.set_xlabel(f"{title} (degrees)")
    ax.set_ylabel("Frequency")


# Get coordinates
lonV = np.deg2rad(pp_common.select_longitude_series(AR_IA_df).to_numpy(dtype=np.float64))
latV = np.deg2rad(pp_common.select_latitude_series(AR_IA_df).to_numpy(dtype=np.float64))
degree_ticks = np.arange(-90, 91, 15)

# Plot histograms
with plt.style.context("seaborn-v0_8-darkgrid"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_histogram(ax1, lonV, degree_ticks, "Longitude")
    plot_histogram(ax2, latV, degree_ticks, "Latitude")
    plt.tight_layout()
    plt.show()


# %%
results = ut_d.filter_by_location(AR_IA_df, lon_limit_deg=65)

# Calculate y, z coordinates for plotting
yV = np.cos(latV) * np.sin(lonV)
zV = np.sin(latV)

# Create solar disc visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_artist(plt.Circle((0, 0), 1, edgecolor="gray", facecolor="none"))
create_solar_grid(ax)

# Plot data points using masks
front_mask, rear_mask = results["mask_front"], results["mask_rear"]
ax.scatter(yV[rear_mask], zV[rear_mask], s=1, alpha=0.2, color=hmi_color, label="Rear")
ax.scatter(yV[front_mask], zV[front_mask], s=1, alpha=0.2, color=mdi_color, label="Front")

# Configure plot
ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
ax.axis("off")
ax.legend(fontsize=12)
plt.show()

# Print statistics
front_count = int(front_mask.sum())
rear_count = int(rear_mask.sum())
total_count = front_count + rear_count
print(f"Rear ARs: {rear_count:,}")
print(f"Front ARs: {front_count:,}")
print(f"Percentage of rear ARs: {100 * rear_count / total_count:.2f}%")

ut_v.make_classes_histogram(results["df_front"]["label"], title="Front ARs", y_off=10, figsz=(11, 5))
ut_v.make_classes_histogram(results["df_rear"]["label"], title="Rear ARs", y_off=10, figsz=(11, 5))
plt.show()

# %% [markdown]
# # Time Distribution
# %%
mdi_df = AR_IA_df[AR_IA_df["path_image_cutout_mdi"] != ""]
hmi_df = AR_IA_df[AR_IA_df["path_image_cutout_hmi"] != ""]

# Get time series data
mdi_dates, hmi_dates = mdi_df["dates"].values, hmi_df["dates"].values
mdi_counts, hmi_counts = mdi_df["dates"].value_counts().sort_index(), hmi_df["dates"].value_counts().sort_index()

# Setup plot
tick_dates = [datetime(year, 1, 1) for year in range(1996, 2025, 2)]

with plt.style.context("seaborn-v0_8-darkgrid"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1]})

    # Top panel: Bar chart
    ax1.bar(mdi_counts.index, mdi_counts.values, width=0.8, color=mdi_color, alpha=0.9, label="MDI")
    ax1.bar(hmi_counts.index, hmi_counts.values, width=0.8, color=hmi_color, alpha=0.9, label="HMI")
    ax1.set(ylabel="n° of ARs per day", ylim=[0, 20], yticks=np.arange(0, 20 + 2, 2))
    ax1.tick_params(axis="y", labelsize=14)
    ax1.legend(loc="upper left", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Bottom panel: Timeline
    for dates, color, y_range in zip([mdi_dates, hmi_dates], [mdi_color, hmi_color], [(0.2, 0.8), (1.2, 1.8)]):
        ax2.vlines(dates, *y_range, color=color, alpha=0.9, linewidth=0.5)

    ax2.set(ylim=[0, 2], yticks=[])
    ax2.grid(True, linestyle="--", alpha=0.75)

    # X-axis formatting
    ax2.xaxis_date()
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xticks(tick_dates)
    plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=14)
    ax2.set_xlabel("Time", fontsize=16)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# # McIntosh Classification

# %%
AR_df = df[df["magnetic_class"] != ""].copy()

ut_v.make_classes_histogram(
    AR_df["mcintosh_class"],
    horizontal=True,
    figsz=(10, 18),
    y_off=20,
    x_rotation=0,
    ylabel="Number of Active Regions",
    title="McIntosh Class Distribution",
    ylim=5900,
)
plt.show()

# %%

# McIntosh classification components
AR_df = df[df["magnetic_class"] != ""].copy()
for comp in ["Z_component", "p_component", "c_component"]:
    AR_df[comp] = AR_df["mcintosh_class"].str[{"Z_component": 0, "p_component": 1, "c_component": 2}[comp]]

# Plot parameters and histograms
params = [
    ("Z_component", (10, 6), "Z McIntosh Component"),
    ("p_component", (9, 6), "p McIntosh Component"),
    ("c_component", (6, 6), "c McIntosh Component"),
]

for component, figsz, title in params:
    ut_v.make_classes_histogram(AR_df[component], y_off=50, figsz=figsz, title=title)
# %%
mappings = {
    # Merge D, E, F into LG (LargeGroup)
    "Z_component": {"A": "A", "B": "B", "C": "C", "D": "LG", "E": "LG", "F": "LG", "H": "H"},
    # Merge s and h into sym & a and k into asym
    "p_component": {"x": "x", "r": "r", "s": "sym", "h": "sym", "a": "asym", "k": "asym"},
    # Merge i and c into frag
    "c_component": {"x": "x", "o": "o", "i": "frag", "c": "frag"},
}

# Apply mappings and plot
for comp, mapping in mappings.items():
    AR_df[f"{comp}_grouped"] = AR_df[comp].map(mapping)
    ut_v.make_classes_histogram(
        AR_df[f"{comp}_grouped"],
        y_off=50,
        figsz={"Z_component": (8, 6), "p_component": (6, 6), "c_component": (5, 6)}[comp],
        title=f"{comp.split('_')[0].upper()} McIntosh Component (Grouped)",
    )
    plt.show()


# %%
print("------ McIntosh Classes ------")
mcintosh_classes = sorted(AR_df["mcintosh_class"].unique())
for letter in sorted({cls[0] for cls in mcintosh_classes}):
    classes = [cls for cls in mcintosh_classes if cls.startswith(letter)]
    print(f"{letter}: {', '.join(classes)}")
print(f"\nn° of classes: {len(AR_df['mcintosh_class'].unique())}")
print("\n------ Grouped McIntosh Classes ------")
grouped_classes = list(
    (AR_df["Z_component_grouped"] + AR_df["p_component_grouped"] + AR_df["c_component_grouped"]).unique()
)
grouped_classes_sorted = sorted(grouped_classes)
for letter in sorted({cls[0] for cls in grouped_classes_sorted}):
    classes = [cls for cls in grouped_classes_sorted if cls.startswith(letter)]
    print(f"{letter}: {', '.join(classes)}")
print(f"\nn° of classes: {len(grouped_classes)}")
# %% [markdown]
# # Pixel Values


# %% [markdown]
# ### Single Image


# %%
idx = 144

data = load_and_analyze_fits_pair(idx, AR_IA_df, data_folder, dataset_folder)

# Create subplot and display
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"{data['row']['label']} - {data['row']['mcintosh_class']} - {data['row']['dates']}", fontsize=12, y=0.95)

# Display both images with colorbars
for ax, img_data, title, stats in zip(
    [ax1, ax2],
    [data["mag_data"], data["cont_data"]],
    ["Magnetogram", "Continuum"],
    [data["mag_stats"], data["cont_stats"]],
):
    im = ax.imshow(img_data, cmap="gray", origin="lower")
    ax.set_title(f"{title}\nMean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.show()

print(f"Label: {data['row']['label']} - {data['row']['mcintosh_class']}")
print(f"Date: {data['row']['dates']}")
print(f"{'Statistic':<12} {'Magnetogram':<12} {'Continuum':<12}")
print("-" * 36)
print(f"{'Mean':<12} {data['mag_stats']['mean']:<12.2f} {data['cont_stats']['mean']:<12.2f}")
print(f"{'Std Dev':<12} {data['mag_stats']['std']:<12.2f} {data['cont_stats']['std']:<12.2f}")
print(f"{'Min':<12} {data['mag_stats']['min']:<12.2f} {data['cont_stats']['min']:<12.2f}")
print(f"{'Max':<12} {data['mag_stats']['max']:<12.2f} {data['cont_stats']['max']:<12.2f}")

# %% [markdown]
# ### All images statistics


# %%
process_row_fn = partial(process_row, df_clean=AR_IA_df, data_folder=data_folder, dataset_folder=dataset_folder)
results = p_map(process_row_fn, range(len(AR_IA_df)))

# %%
flat_stats = []
for entry in results:
    if entry is not None:
        idx = entry["index"]
        row = {
            "index": idx,
            "label": entry["label"],
            "mcintosh_class": entry["mcintosh_class"],
        }
        row.update({f"mag_{k}": v for k, v in entry["mag_stats"].items()})
        row.update({f"cont_{k}": v for k, v in entry["cont_stats"].items()})
        flat_stats.append(row)
stats_df = pd.DataFrame(flat_stats)
stats_df.describe()

# %%
# Find the indices of the 10 highest mag_mean values
top10_indices = stats_df["mag_mean"].nlargest(10).index
# Get the corresponding rows
top10_rows = stats_df.loc[top10_indices]
top10_rows

# %%
stats_config = [
    ("mag_mean", "Magnetogram Mean", "royalblue"),
    ("mag_std", "Magnetogram Std Dev", "royalblue"),
    ("mag_min", "Magnetogram Min", "royalblue"),
    ("mag_max", "Magnetogram Max", "royalblue"),
    ("cont_mean", "Continuum Mean", "tomato"),
    ("cont_std", "Continuum Std Dev", "tomato"),
    ("cont_min", "Continuum Min", "tomato"),
    ("cont_max", "Continuum Max", "tomato"),
]

# Create histograms
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, (col, title, color) in enumerate(stats_config):
    ax = axes.flat[i]
    ax.hist(stats_df[col], bins=50, color=color, alpha=0.7, edgecolor="black", linewidth=0.5, log=True)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Value", fontsize=14)
    if i % 4 == 0:  # First column gets y-label
        ax.set_ylabel("Frequency", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

plt.tight_layout()
plt.show()

# %%
# Create boxplots by class
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
all_labels = stats_df["label"].unique()

# Create a custom color palette
palette = {label: colors[i % len(colors)] for i, label in enumerate(all_labels)}

for col, title, _ in stats_config:
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=stats_df, x="label", y=col, hue="label", palette=palette, ax=ax, legend=False)
    ax.set_title(f"{title} by Active Region Class", fontsize=18)
    ax.tick_params(axis="x", rotation=45, labelsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Value", fontsize=16)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(ha="right")
    plt.tight_layout()
    plt.show()
# %%
