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
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from p_tqdm import p_map

from astropy.table import Table

from arccnet.models import labels
from arccnet.models.fulldisk import utils as fd_utils
from arccnet.visualisation import utils as ut_v

sns.set_style("darkgrid")
pd.set_option("display.max_columns", None)
# %%
data_folder = Path(os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data/"))
dataset_root = Path("arccnet-v20251017")
dataset_folder = dataset_root / "04_final"
df_name = "data/region_detection/region_detection_noaa-xarp.parq"
dataset_title = "arccnet v20251017"

tab = Table.read(data_folder / dataset_folder / df_name)
good_quality_tab = tab[~tab["filtered"]]
bad_quality_tab = tab[tab["filtered"]]

# Convert only 1D columns to DataFrame
names_1d = [name for name in good_quality_tab.colnames if len(good_quality_tab[name].shape) <= 1]
df = good_quality_tab[names_1d].to_pandas()
# Add multidimensional columns as lists
for col in ["top_right_cutout", "bottom_left_cutout"]:
    df[col] = list(good_quality_tab[col])

# Replace all 0.0 values in magnetic_class with 'IA' for consistency
if "magnetic_class" in df.columns:
    df["magnetic_class"] = df["magnetic_class"].apply(lambda x: "IA" if x == 0.0 or pd.isna(x) else x)

# %% [markdown]
# ## Global Min/Max Analysis for Continuum Images
# %%

print("Computing global min/max values for continuum images...")
print("=" * 80)

# Get and filter unique continuum image paths
unique_cont_images = df["processed_path_image_cont"].unique()
unique_cont_images = [p for p in unique_cont_images if isinstance(p, str)]
print(f"Total unique continuum images: {len(unique_cont_images)}")

stats_results = p_map(
    fd_utils.get_fits_statistics,
    unique_cont_images,
    [data_folder] * len(unique_cont_images),
    [dataset_root] * len(unique_cont_images),
)
global_mins, global_maxs, global_means, global_stds, nan_counts, total_pixels = map(np.array, zip(*stats_results))

# Compute global statistics
global_min = np.min(global_mins)
global_max = np.max(global_maxs)
overall_mean = np.mean(global_means)
overall_std = np.mean(global_stds)
total_nan_pixels = np.sum(nan_counts)
total_pixel_count = np.sum(total_pixels)
nan_percentage = (total_nan_pixels / total_pixel_count) * 100

print("\nGLOBAL CONTINUUM STATISTICS:")
print(f"  Global Min: {global_min:.4e}")
print(f"  Global Max: {global_max:.4e}")
print(f"  Global Range: {global_max - global_min:.4e}")
print(f"  Mean of Means: {overall_mean:.4e}")
print(f"  Mean of Stds: {overall_std:.4e}")
print(f"  NaN pixels: {total_nan_pixels:,} / {total_pixel_count:,} ({nan_percentage:.3f}%)")


# %%
# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Per-image min values
axes[0, 0].hist(global_mins, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
axes[0, 0].axvline(global_min, color="red", linestyle="--", linewidth=2, label=f"Global Min: {global_min:.4e}")
axes[0, 0].set_xlabel("Minimum Value")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Distribution of Per-Image Minimum Values")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Per-image max values
axes[0, 1].hist(global_maxs, bins=50, edgecolor="black", alpha=0.7, color="coral")
axes[0, 1].axvline(global_max, color="red", linestyle="--", linewidth=2, label=f"Global Max: {global_max:.4e}")
axes[0, 1].set_xlabel("Maximum Value")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Distribution of Per-Image Maximum Values")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Per-image mean values
axes[1, 0].hist(global_means, bins=50, edgecolor="black", alpha=0.7, color="lightgreen")
axes[1, 0].axvline(overall_mean, color="red", linestyle="--", linewidth=2, label=f"Overall Mean: {overall_mean:.4e}")
axes[1, 0].set_xlabel("Mean Value")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Distribution of Per-Image Mean Values")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# NaN percentage per image
nan_percentages = np.array(nan_counts) / np.array(total_pixels) * 100
axes[1, 1].hist(nan_percentages, bins=50, edgecolor="black", alpha=0.7, color="orange")
axes[1, 1].axvline(
    nan_percentage, color="red", linestyle="--", linewidth=2, label=f"Overall NaN%: {nan_percentage:.3f}%"
)
axes[1, 1].set_xlabel("NaN Percentage")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Distribution of NaN Percentage per Image")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Box plots for outlier detection
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].boxplot(global_mins, vert=True)
axes[0].set_ylabel("Value")
axes[0].set_title("Per-Image Minimum Values\n(Outlier Detection)")
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(global_maxs, vert=True)
axes[1].set_ylabel("Value")
axes[1].set_title("Per-Image Maximum Values\n(Outlier Detection)")
axes[1].grid(True, alpha=0.3)

axes[2].boxplot(global_means, vert=True)
axes[2].set_ylabel("Value")
axes[2].set_title("Per-Image Mean Values\n(Outlier Detection)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
fd_utils.identify_outliers(
    global_mins,
    unique_cont_images,
    "Minimum Values",
    visualize=True,
    data_folder=data_folder,
    dataset_root=dataset_root,
)
fd_utils.identify_outliers(
    global_maxs,
    unique_cont_images,
    "Maximum Values",
    visualize=True,
    data_folder=data_folder,
    dataset_root=dataset_root,
)
fd_utils.identify_outliers(
    global_means, unique_cont_images, "Mean Values", visualize=True, data_folder=data_folder, dataset_root=dataset_root
)

print("\n" + "=" * 80)
print("Recommended normalization values for CONTINUUM:")
print(f"  Use global_min = {global_min:.4e}")
print(f"  Use global_max = {global_max:.4e}")
print("=" * 80)

# %% [markdown]
# ## Global Min/Max Analysis for Magnetogram Images
# %%
print("\nComputing global min/max values for magnetogram images...")
print("=" * 80)

# Get unique magnetogram images
unique_mag_images = df["processed_path_image_mag"].unique()
# Filter out non-string paths (e.g., NaN floats)
unique_mag_images = [p for p in unique_mag_images if isinstance(p, str)]
print(f"Total unique magnetogram images: {len(unique_mag_images)}")

mag_stats_results = p_map(
    fd_utils.get_fits_statistics,
    unique_mag_images,
    [data_folder] * len(unique_mag_images),
    [dataset_root] * len(unique_mag_images),
)
mag_global_mins, mag_global_maxs, mag_global_means, mag_global_stds, mag_nan_counts, mag_total_pixels = map(
    np.array, zip(*mag_stats_results)
)

# Compute global statistics for magnetograms
mag_global_min = np.min(mag_global_mins)
mag_global_max = np.max(mag_global_maxs)
mag_overall_mean = np.mean(mag_global_means)
mag_overall_std = np.mean(mag_global_stds)
mag_total_nan_pixels = np.sum(mag_nan_counts)
mag_total_pixel_count = np.sum(mag_total_pixels)
mag_nan_percentage = (mag_total_nan_pixels / mag_total_pixel_count) * 100

print("\nGLOBAL MAGNETOGRAM STATISTICS:")
print(f"  Global Min: {mag_global_min:.4e}")
print(f"  Global Max: {mag_global_max:.4e}")
print(f"  Global Range: {mag_global_max - mag_global_min:.4e}")
print(f"  Mean of Means: {mag_overall_mean:.4e}")
print(f"  Mean of Stds: {mag_overall_std:.4e}")
print(f"  NaN pixels: {mag_total_nan_pixels:,} / {mag_total_pixel_count:,} ({mag_nan_percentage:.3f}%)")

# %%
# Visualize magnetogram distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Per-image min values
axes[0, 0].hist(mag_global_mins, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
axes[0, 0].axvline(mag_global_min, color="red", linestyle="--", linewidth=2, label=f"Global Min: {mag_global_min:.4e}")
axes[0, 0].set_xlabel("Minimum Value")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Magnetogram: Distribution of Per-Image Minimum Values")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Per-image max values
axes[0, 1].hist(mag_global_maxs, bins=50, edgecolor="black", alpha=0.7, color="coral")
axes[0, 1].axvline(mag_global_max, color="red", linestyle="--", linewidth=2, label=f"Global Max: {mag_global_max:.4e}")
axes[0, 1].set_xlabel("Maximum Value")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Magnetogram: Distribution of Per-Image Maximum Values")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Per-image mean values
axes[1, 0].hist(mag_global_means, bins=50, edgecolor="black", alpha=0.7, color="lightgreen")
axes[1, 0].axvline(
    mag_overall_mean, color="red", linestyle="--", linewidth=2, label=f"Overall Mean: {mag_overall_mean:.4e}"
)
axes[1, 0].set_xlabel("Mean Value")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Magnetogram: Distribution of Per-Image Mean Values")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# NaN percentage per image
mag_nan_percentages = np.array(mag_nan_counts) / np.array(mag_total_pixels) * 100
axes[1, 1].hist(mag_nan_percentages, bins=50, edgecolor="black", alpha=0.7, color="orange")
axes[1, 1].axvline(
    mag_nan_percentage, color="red", linestyle="--", linewidth=2, label=f"Overall NaN%: {mag_nan_percentage:.3f}%"
)
axes[1, 1].set_xlabel("NaN Percentage")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Magnetogram: Distribution of NaN Percentage per Image")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Box plots for outlier detection in magnetograms
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].boxplot(mag_global_mins, vert=True)
axes[0].set_ylabel("Value")
axes[0].set_title("Magnetogram: Per-Image Minimum Values\n(Outlier Detection)")
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(mag_global_maxs, vert=True)
axes[1].set_ylabel("Value")
axes[1].set_title("Magnetogram: Per-Image Maximum Values\n(Outlier Detection)")
axes[1].grid(True, alpha=0.3)

axes[2].boxplot(mag_global_means, vert=True)
axes[2].set_ylabel("Value")
axes[2].set_title("Magnetogram: Per-Image Mean Values\n(Outlier Detection)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
print("\nMAGNETOGRAM OUTLIER ANALYSIS:")
fd_utils.identify_outliers(
    mag_global_mins,
    unique_mag_images,
    "Minimum Values",
    visualize=True,
    data_folder=data_folder,
    dataset_root=dataset_root,
)
fd_utils.identify_outliers(
    mag_global_maxs,
    unique_mag_images,
    "Maximum Values",
    visualize=True,
    data_folder=data_folder,
    dataset_root=dataset_root,
)
fd_utils.identify_outliers(
    mag_global_means,
    unique_mag_images,
    "Mean Values",
    visualize=True,
    data_folder=data_folder,
    dataset_root=dataset_root,
)

print("\n" + "=" * 80)
print("Recommended normalization values for MAGNETOGRAM:")
print(f"  Use global_min = {mag_global_min:.4e}")
print(f"  Use global_max = {mag_global_max:.4e}")
print("=" * 80)

# %%

# %%
discarded_df = df[df["filtered"]]
selected_df = df[~df["filtered"]]
# Calculate the percentage of selected items and unique fulldisks
selected_percentage = len(selected_df) / len(df) * 100
unique_fulldisks_count = len(selected_df["processed_path_image_mag"].unique())
unique_fulldisks_total = len(df["processed_path_image_mag"].unique())
unique_fulldisks_percentage = unique_fulldisks_count / unique_fulldisks_total * 100
# Print the formatted output
print(f"             Selected FDs: {len(selected_df)} out of {len(df)} ({selected_percentage:.2f}%)")
print(
    f"Selected Unique Fulldisks: {unique_fulldisks_count} out of {unique_fulldisks_total}  "
    f"({unique_fulldisks_percentage:.2f}%)"
)
# %%
discarded_counts = discarded_df["magnetic_class"].value_counts().sort_index()
selected_counts = selected_df["magnetic_class"].value_counts().sort_index()
all_classes = sorted(set(discarded_counts.index) | set(selected_counts.index))
discarded_counts = discarded_counts.reindex(all_classes, fill_value=0)
selected_counts = selected_counts.reindex(all_classes, fill_value=0)
counts_df = pd.DataFrame({"Selected": selected_counts, "Discarded": discarded_counts}, index=all_classes)
greek_labels = labels.convert_to_greek_label(all_classes)
class_to_greek = dict(zip(all_classes, greek_labels))
counts_df.index = counts_df.index.map(class_to_greek)
counts_df["Total"] = counts_df["Selected"] + counts_df["Discarded"]
grand_total = counts_df["Total"].sum()
counts_df["Selected_pct"] = counts_df["Selected"] / grand_total * 100
counts_df["Discarded_pct"] = counts_df["Discarded"] / grand_total * 100
counts_df["Total_pct"] = counts_df["Total"] / grand_total * 100
counts_df["Selected_formatted"] = counts_df.apply(lambda x: f"{int(x['Selected'])} ({x['Selected_pct']:.2f}%)", axis=1)
counts_df["Discarded_formatted"] = counts_df.apply(
    lambda x: f"{int(x['Discarded'])} ({x['Discarded_pct']:.2f}%)", axis=1
)
counts_df["Total_formatted"] = counts_df.apply(lambda x: f"{int(x['Total'])} ({x['Total_pct']:.2f}%)", axis=1)
formatted_counts_df = counts_df[["Selected_formatted", "Discarded_formatted", "Total_formatted"]]
formatted_counts_df.columns = ["Selected", "Discarded", "Total"]
formatted_counts_df
# %%
class_series = selected_df["magnetic_class"].apply(lambda x: "IA" if (x == 0.0 or pd.isna(x)) else x)
ut_v.make_classes_histogram(
    class_series,
    y_off=20,
    figsz=(9, 5),
    title="FullDisk Dataset AR classes",
    # ylim=3400,
    transparent=True,
)
plt.show()
# %%
idx = 4567
row = selected_df.iloc[idx]
prefix = "/mnt/ARCAFF/v0.3.0"
local_root = str(data_folder / dataset_root)
# Map 0.0 and NaN to 'IA' for magnetic_class in image_labels
image_labels = selected_df[selected_df["processed_path_image_mag"] == row["processed_path_image_mag"]].copy()
image_labels["magnetic_class"] = image_labels["magnetic_class"].apply(lambda x: "IA" if (x == 0.0 or pd.isna(x)) else x)
fits_magn_path = row["processed_path_image_mag"].replace(prefix, local_root)
fits_cont_path = row["processed_path_image_cont"].replace(prefix, local_root)
fig, (ax_mag, ax_cont), _ = fd_utils.plot_full_disk_pair(
    fits_magnetogram=fits_magn_path,
    fits_continuum=fits_cont_path,
    image_labels=image_labels,
)
fd_utils.add_class_legend(
    fig,
    fd_utils.CLASS_COLOR_MAP,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=min(len(fd_utils.CLASS_COLOR_MAP), 5),
    frameon=False,
)
plt.tight_layout()
plt.show()
# %% [markdown]
# ## Cutout Sizes
# fd_utils.w_h_scatterplot(selected_df)
# %%
widths, heights, _, _ = fd_utils.compute_widths_heights(selected_df)
aspect_ratios = np.array(widths) / np.array(heights)
hist_fig = go.Figure()
# Add a histogram trace for aspect ratios
hist_fig.add_trace(go.Histogram(x=aspect_ratios, nbinsx=50, marker=dict(color="blue"), opacity=0.75))
# Update the layout for the histogram
hist_fig.update_layout(
    title="Aspect Ratios", xaxis_title="Aspect Ratio (Width/Height)", yaxis_title="Count", autosize=True
)
# Show the histogram
hist_fig.show()
# %%
lon_trshld = 70
front_df = fd_utils.filter_front_side(selected_df, longitude_threshold=lon_trshld)
back_df = selected_df[(selected_df["longitude"] >= lon_trshld) | (selected_df["longitude"] <= -lon_trshld)]
# %%
fig, ax = plt.subplots(figsize=(8, 8))
fd_utils.plot_ar_locations(front_df, ax=ax, color="#1f77b4", label="Front Side", marker_size=1, alpha=0.2)
fd_utils.plot_ar_locations(back_df, ax=ax, color="darkorange", label="Back Side", marker_size=1, alpha=0.2)
ax.set_title("ARs Location on the Sun")
ax.legend(loc="upper right", frameon=False)
plt.show()
# %%
fd_utils.w_h_scatterplot(front_df)
# %%
min_size = 0.03
img_size_dic = fd_utils.IMG_SIZE_BY_INSTRUMENT

# Calculate region sizes and filter by minimum size
cleaned_df = fd_utils.calculate_region_sizes(front_df, img_size_dic)
cleaned_df = fd_utils.filter_by_minimum_size(cleaned_df, min_size)
# %%
fd_utils.w_h_scatterplot(cleaned_df)
