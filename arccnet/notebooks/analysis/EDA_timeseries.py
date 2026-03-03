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
#     display_name: venv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Timeseries Data EDA (Current Dataset Status)
#
# This notebook focuses on the **current timeseries dataset status** and
# the **current 3-class target setup**:
#
# - `No-flare`
# - `C`
# - `M+` (M or X)
#
# It provides:
# - dataset overview and target balance
# - temporal coverage and drift plots
# - split-wise class balance checks
# - path completeness and file-availability checks
# - quick visual inspection of a representative sample
# %%
import os
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits

from arccnet.models.timeseries import config as ts_config
from arccnet.models.timeseries.manifest import build_dataset
from arccnet.models.timeseries.splitters import get_split
from arccnet.visualisation import utils as ut_v

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")

CLASS_NAMES = list(getattr(ts_config, "FLARE_CLASS_NAMES", ["No-flare", "C", "M+"]))
CLASS_COLORS = {
    "No-flare": "#4C78A8",
    "C": "#F58518",
    "M+": "#E45756",
    "M": "#E45756",
    "X": "#B279A2",
}
CHANNEL_LABELS = [str(wl) for wl in ts_config.CHANNEL_ORDER]


def resolve_timeseries_root(data_folder):
    """Resolve timeseries root across legacy and namespaced layouts."""
    env_root = os.getenv("ARCAFF_TIMESERIES_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(
        [
            Path(data_folder) / "timeseries" / "04_final" / "data",
            Path(data_folder) / "04_final" / "data",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find timeseries data root. Checked: " + ", ".join(str(c) for c in candidates))


def parse_paths_grid(raw_paths):
    """Parse manifest `paths` field into a list-of-lists."""
    if raw_paths is None:
        return []
    if isinstance(raw_paths, float) and np.isnan(raw_paths):
        return []

    if isinstance(raw_paths, str):
        try:
            parsed = json.loads(raw_paths)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(raw_paths)
    elif isinstance(raw_paths, np.ndarray):
        parsed = raw_paths.tolist()
    else:
        parsed = raw_paths

    if parsed is None:
        return []
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        return []

    out = []
    for timestep in parsed:
        if timestep is None:
            out.append([])
            continue
        if isinstance(timestep, np.ndarray):
            timestep = timestep.tolist()
        if not isinstance(timestep, (list, tuple)):
            timestep = [timestep]
        out.append(list(timestep))
    return out


def safe_ratio(numerator, denominator):
    """Division helper that avoids division-by-zero."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def annotate_bars(ax, values):
    """Write values above bars."""
    ymax = max(values) if len(values) > 0 else 0
    offset = max(1, ymax * 0.01)
    for idx, val in enumerate(values):
        ax.text(idx, val + offset, f"{int(val)}", ha="center", va="bottom", fontsize=9)


def map_flare_class_name(value, class_names):
    """Map integer class index to configured class label."""
    try:
        idx = int(value)
    except Exception:
        return f"class_{value}"
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return f"class_{idx}"


def compute_path_completeness(df, num_timesteps, num_channels):
    """Compute path-level completeness and timestep/channel availability rates."""
    channel_presence = np.zeros((num_timesteps, num_channels), dtype=np.int64)
    missing_fractions = []
    fully_observed_samples = []
    fully_observed_timesteps = []

    total_expected = num_timesteps * num_channels

    for raw_paths in df["paths"]:
        grid = parse_paths_grid(raw_paths)
        valid_entries = 0
        full_steps = 0

        for t in range(num_timesteps):
            row = grid[t] if t < len(grid) else []
            step_valid = 0

            for c in range(num_channels):
                path_entry = row[c] if c < len(row) else None
                is_valid = path_entry not in (None, "", "None")

                if is_valid:
                    valid_entries += 1
                    step_valid += 1
                    channel_presence[t, c] += 1

            if step_valid == num_channels:
                full_steps += 1

        missing_fractions.append(1.0 - safe_ratio(valid_entries, total_expected))
        fully_observed_samples.append(valid_entries == total_expected)
        fully_observed_timesteps.append(full_steps)

    out = df.copy()
    out["missing_fraction_paths"] = missing_fractions
    out["is_fully_observed_paths"] = fully_observed_samples
    out["fully_observed_timesteps"] = fully_observed_timesteps

    availability_rate = channel_presence / max(len(df), 1)
    return out, availability_rate


def estimate_on_disk_missing_rate(df, sample_rows=120, random_state=42):
    """Estimate physical file missing rate from a random subset of samples."""
    if len(df) == 0:
        return {
            "checked_paths": 0,
            "missing_paths": 0,
            "missing_rate": 0.0,
            "channel_missing_rate": np.zeros(ts_config.NUM_CHANNELS, dtype=np.float32),
        }

    n_rows = min(sample_rows, len(df))
    probe = df.sample(n=n_rows, random_state=random_state).reset_index(drop=True)

    checked_paths = 0
    missing_paths = 0
    checked_by_channel = np.zeros(ts_config.NUM_CHANNELS, dtype=np.int64)
    missing_by_channel = np.zeros(ts_config.NUM_CHANNELS, dtype=np.int64)

    for raw_paths in probe["paths"]:
        grid = parse_paths_grid(raw_paths)
        for t in range(ts_config.NUM_TIMESTEPS):
            row = grid[t] if t < len(grid) else []
            for c in range(ts_config.NUM_CHANNELS):
                path_entry = row[c] if c < len(row) else None
                if path_entry in (None, "", "None"):
                    continue

                checked_paths += 1
                checked_by_channel[c] += 1
                if not Path(str(path_entry)).exists():
                    missing_paths += 1
                    missing_by_channel[c] += 1

    channel_missing_rate = np.array(
        [safe_ratio(missing_by_channel[c], checked_by_channel[c]) for c in range(ts_config.NUM_CHANNELS)],
        dtype=np.float32,
    )

    return {
        "checked_paths": int(checked_paths),
        "missing_paths": int(missing_paths),
        "missing_rate": safe_ratio(missing_paths, checked_paths),
        "channel_missing_rate": channel_missing_rate,
    }


def load_fits_2d(path):
    """Load a FITS file into a clean 2D float32 array."""
    with fits.open(path) as hdul:
        hdu_idx = 1 if len(hdul) > 1 and hdul[1].data is not None else 0
        data = hdul[hdu_idx].data.astype(np.float32)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def pick_visualization_sample(df):
    """Pick one sample with at least one full valid timestep on disk."""
    ranked = df.sort_values(["missing_fraction_paths", "date"]).reset_index(drop=True)
    for _, row in ranked.iterrows():
        grid = parse_paths_grid(row["paths"])
        for t in range(min(ts_config.NUM_TIMESTEPS, len(grid))):
            timestep_paths = grid[t]
            if len(timestep_paths) < ts_config.NUM_CHANNELS:
                continue
            valid = True
            for path_entry in timestep_paths[: ts_config.NUM_CHANNELS]:
                if path_entry in (None, "", "None") or not Path(str(path_entry)).exists():
                    valid = False
                    break
            if valid:
                return row, t, timestep_paths[: ts_config.NUM_CHANNELS]
    return None, None, None


# %% [markdown]
# ## 1. Load Current Manifest / Build from Current Data Root

# %%
DATA_FOLDER = Path(os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data"))
TIMESERIES_ROOT = resolve_timeseries_root(DATA_FOLDER)
MANIFEST_PATH = DATA_FOLDER / "timeseries_manifest.parquet"

# Set to True when you want to force a rebuild from disk.
REBUILD_MANIFEST = True
MAX_SAMPLES = None

print(f"Data folder:      {DATA_FOLDER}")
print(f"Timeseries root:  {TIMESERIES_ROOT}")
print(f"Manifest path:    {MANIFEST_PATH}")
print(f"Configured classes: {CLASS_NAMES}")

if REBUILD_MANIFEST or not MANIFEST_PATH.exists():
    print("\nBuilding manifest from current data root...")
    df_manifest = build_dataset(TIMESERIES_ROOT, output_path=MANIFEST_PATH, max_samples=MAX_SAMPLES)
else:
    print("\nLoading existing manifest...")
    df_manifest = pd.read_parquet(MANIFEST_PATH)
    if MAX_SAMPLES is not None:
        df_manifest = df_manifest.head(MAX_SAMPLES).copy()

print(f"Loaded samples: {len(df_manifest):,}")
df_manifest.head(3)

# %% [markdown]
# ## 2. Derive EDA Columns

# %%
df_eda = df_manifest.copy()
df_eda["datetime"] = pd.to_datetime(df_eda["date"], errors="coerce")
df_eda["year"] = df_eda["datetime"].dt.year
df_eda["month"] = df_eda["datetime"].dt.month

df_eda["total_before"] = df_eda["xb"] + df_eda["mb"] + df_eda["cb"]
df_eda["total_after"] = df_eda["xa"] + df_eda["ma"] + df_eda["ca"]
df_eda["has_flare_after"] = df_eda["total_after"] > 0
df_eda["is_m_plus_target"] = (df_eda["ma"] > 0) | (df_eda["xa"] > 0)

df_eda["flare_class_name"] = df_eda["flare_class"].apply(lambda v: map_flare_class_name(v, CLASS_NAMES))
df_eda["flare_class_name"] = pd.Categorical(df_eda["flare_class_name"], categories=CLASS_NAMES, ordered=True)

df_eda, channel_availability = compute_path_completeness(
    df_eda,
    num_timesteps=ts_config.NUM_TIMESTEPS,
    num_channels=ts_config.NUM_CHANNELS,
)

summary = {
    "samples": len(df_eda),
    "unique_noaa_ars": int(df_eda["noaa_ar"].nunique()),
    "date_start": str(df_eda["datetime"].min().date()) if len(df_eda) else "N/A",
    "date_end": str(df_eda["datetime"].max().date()) if len(df_eda) else "N/A",
    "m_plus_rate_percent": 100.0 * float(df_eda["is_m_plus_target"].mean()) if len(df_eda) else 0.0,
    "fully_observed_path_samples_percent": 100.0 * float(df_eda["is_fully_observed_paths"].mean())
    if len(df_eda)
    else 0.0,
}

print("Current dataset snapshot:")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")

# %% [markdown]
# ## 3. Class Balance (Current 3-Class Setup)

# %%
class_order_3 = ["No-flare", "C", "M+"]
class_counts_3 = df_eda["flare_class_name"].value_counts().reindex(class_order_3, fill_value=0)
class_colors_3 = [CLASS_COLORS.get(name, "#4C78A8") for name in class_order_3]

ut_v.make_classes_histogram(
    df_eda["flare_class_name"],
    figsz=(8, 5),
    y_off=10,
    ylim=class_counts_3.max() * 1.1,
    fontsize=11,
    title="Class Counts (No-flare / C / M+)",
    ylabel="Samples",
    categories=class_order_3,
)
plt.show()

# Define class_colors for use in later plots (temporal coverage, etc.)
class_colors = [CLASS_COLORS.get(name, "#4C78A8") for name in CLASS_NAMES]

# %% [markdown]
# ## 3b. Class Balance (4-Class View: No-flare / C / M / X)


# %%
def map_4class_label(row):
    """Map per-sample flare counts into 4-class target labels."""
    if row.get("xa", 0) > 0:
        return "X"
    if row.get("ma", 0) > 0:
        return "M"
    if row.get("ca", 0) > 0:
        return "C"
    return "No-flare"


df_eda["flare_class_name_4"] = df_eda.apply(map_4class_label, axis=1)

class_order_4 = ["No-flare", "C", "M", "X"]
class_counts_4 = df_eda["flare_class_name_4"].value_counts().reindex(class_order_4, fill_value=0)
class_colors_4 = [CLASS_COLORS.get(name, "#4C78A8") for name in class_order_4]

ut_v.make_classes_histogram(
    df_eda["flare_class_name_4"],
    figsz=(8, 5),
    y_off=10,
    ylim=class_counts_4.max() * 1.1,
    fontsize=11,
    title="Class Counts (No-flare / C / M / X)",
    ylabel="Samples",
    categories=class_order_4,
)
plt.show()

# %% [markdown]
# ## 4. Temporal Coverage and Target Drift

# %%
monthly_total = df_eda.set_index("datetime").resample("MS").size()
monthly_class = (
    df_eda.groupby([pd.Grouper(key="datetime", freq="MS"), "flare_class_name"], observed=False)
    .size()
    .unstack(fill_value=0)
    .reindex(columns=CLASS_NAMES, fill_value=0)
)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(monthly_total.index, monthly_total.values, color="#2E86AB", linewidth=2)
axes[0].set_title("Monthly Sample Volume")
axes[0].set_ylabel("Samples / month")
axes[0].grid(True, alpha=0.25)

axes[1].stackplot(
    monthly_class.index,
    [monthly_class[name].values for name in CLASS_NAMES],
    labels=CLASS_NAMES,
    colors=class_colors,
    alpha=0.9,
)
axes[1].set_title("Monthly Class Composition")
axes[1].set_ylabel("Samples / month")
axes[1].set_xlabel("Month")
axes[1].legend(loc="upper left", ncol=len(CLASS_NAMES))
axes[1].grid(True, alpha=0.25)

plt.tight_layout()
plt.show()

# %%
heatmap = (
    df_eda.assign(year=df_eda["datetime"].dt.year, month=df_eda["datetime"].dt.month)
    .pivot_table(index="year", columns="month", values="sample_id", aggfunc="count", fill_value=0)
    .sort_index()
)

fig, ax = plt.subplots(figsize=(12, 4.8))
im = ax.imshow(heatmap.values, cmap="YlGnBu", aspect="auto")
ax.set_title("Sample Density Heatmap (Year x Month)")
ax.set_xlabel("Month")
ax.set_ylabel("Year")
ax.set_xticks(range(12))
ax.set_xticklabels([f"{m:02d}" for m in range(1, 13)])
ax.set_yticks(range(len(heatmap.index)))
ax.set_yticklabels(heatmap.index.tolist())

for y in range(heatmap.shape[0]):
    for x in range(heatmap.shape[1]):
        value = int(heatmap.values[y, x])
        if value > 0:
            rgba = im.cmap(im.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "black" if luminance > 0.5 else "white"
            ax.text(x, y, str(value), ha="center", va="center", fontsize=8, color=text_color)

plt.colorbar(im, ax=ax, label="Samples")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Split Health Check (Using Current Config Strategy)

# %%
split_kwargs = {"seed": ts_config.SEED}
if ts_config.SPLIT_STRATEGY == "noaa":
    split_kwargs.update({"train_frac": ts_config.TRAIN_FRAC, "val_frac": ts_config.VAL_FRAC})
elif ts_config.SPLIT_STRATEGY == "time":
    split_kwargs.update(
        {"train_years": ts_config.TRAIN_YEARS, "val_years": ts_config.VAL_YEARS, "test_years": ts_config.TEST_YEARS}
    )

split_data = get_split(df_eda, strategy=ts_config.SPLIT_STRATEGY, **split_kwargs)
split_frames = {
    "train": split_data["train_df"],
    "val": split_data["val_df"],
    "test": split_data["test_df"],
}

split_class_counts = pd.DataFrame(
    {
        split_name: split_df["flare_class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
        for split_name, split_df in split_frames.items()
    }
).T
split_class_pct = split_class_counts.div(split_class_counts.sum(axis=1), axis=0).fillna(0.0) * 100.0

print("Split sizes:")
for split_name, split_df in split_frames.items():
    print(f"  {split_name}: {len(split_df):,} samples | {split_df['noaa_ar'].nunique():,} NOAA ARs")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bottom = np.zeros(len(split_class_counts), dtype=np.float32)
for class_name in CLASS_NAMES:
    values = split_class_counts[class_name].values
    axes[0].bar(split_class_counts.index, values, bottom=bottom, label=class_name, color=CLASS_COLORS.get(class_name))
    bottom += values
axes[0].set_title("Split-wise Class Counts")
axes[0].set_ylabel("Samples")
axes[0].legend(title="Class", loc="upper right")
axes[0].grid(True, axis="y", alpha=0.25)

bottom = np.zeros(len(split_class_pct), dtype=np.float32)
for class_name in CLASS_NAMES:
    values = split_class_pct[class_name].values
    axes[1].bar(split_class_pct.index, values, bottom=bottom, label=class_name, color=CLASS_COLORS.get(class_name))
    bottom += values
axes[1].set_title("Split-wise Class Percentages")
axes[1].set_ylabel("Percent")
axes[1].set_ylim(0, 100)
axes[1].grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. NOAA AR Coverage and Flare Target Counts

# %%
ar_counts = df_eda["noaa_ar"].value_counts()
top_ar = ar_counts.head(20).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].barh(top_ar.index.astype(str), top_ar.values, color="#72B7B2", edgecolor="black", linewidth=0.6)
axes[0].set_title("Top 20 Most Observed NOAA ARs")
axes[0].set_xlabel("Samples")
axes[0].set_ylabel("NOAA AR")
axes[0].grid(True, axis="x", alpha=0.25)

axes[1].hist(ar_counts.values, bins=30, color="#54A24B", edgecolor="black", alpha=0.85)
axes[1].set_title("Distribution of Samples per NOAA AR")
axes[1].set_xlabel("Samples per AR")
axes[1].set_ylabel("Number of ARs")
axes[1].grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
target_specs = [
    ("ca", "C-count in 24h (Ca)", "#F58518"),
    ("ma", "M-count in 24h (Ma)", "#E45756"),
    ("xa", "X-count in 24h (Xa)", "#B279A2"),
]

for ax, (col, title, color) in zip(axes, target_specs):
    max_val = int(df_eda[col].max()) if len(df_eda) else 0
    bins = np.arange(0, max_val + 2) - 0.5
    if len(bins) < 2:
        bins = np.array([-0.5, 0.5])
    ax.hist(df_eda[col], bins=bins, color=color, edgecolor="black", alpha=0.9)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel("Samples (log)")
    ax.grid(True, axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Path Completeness and On-Disk Availability

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].hist(df_eda["missing_fraction_paths"], bins=30, color="#4C78A8", edgecolor="black", alpha=0.9)
axes[0].set_title("Missing Path Fraction per Sample")
axes[0].set_xlabel("Missing fraction (0 = complete)")
axes[0].set_ylabel("Samples")
axes[0].grid(True, axis="y", alpha=0.25)

mean_missing_by_class = (
    df_eda.groupby("flare_class_name", observed=False)["missing_fraction_paths"].mean().reindex(CLASS_NAMES).fillna(0.0)
)
axes[1].bar(
    mean_missing_by_class.index,
    mean_missing_by_class.values * 100.0,
    color=[CLASS_COLORS.get(name, "#4C78A8") for name in mean_missing_by_class.index],
    edgecolor="black",
)
axes[1].set_title("Mean Missing Path % by Class")
axes[1].set_ylabel("Missing paths (%)")
axes[1].grid(True, axis="y", alpha=0.25)

im = axes[2].imshow(channel_availability, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGnBu")
axes[2].set_title("Path Availability Rate (Timestep x Channel)")
axes[2].set_xlabel("Channel")
axes[2].set_ylabel("Timestep")
axes[2].set_xticks(range(ts_config.NUM_CHANNELS))
axes[2].set_xticklabels(CHANNEL_LABELS, rotation=45, ha="right")
axes[2].set_yticks(range(ts_config.NUM_TIMESTEPS))
axes[2].set_yticklabels([f"T{t}" for t in range(ts_config.NUM_TIMESTEPS)])
plt.colorbar(im, ax=axes[2], label="Availability rate")

plt.tight_layout()
plt.show()

# %%
existence_stats = estimate_on_disk_missing_rate(df_eda, sample_rows=120, random_state=ts_config.SEED)
print("On-disk existence probe:")
print(f"  Checked paths: {existence_stats['checked_paths']:,}")
print(f"  Missing paths: {existence_stats['missing_paths']:,}")
print(f"  Missing rate:  {existence_stats['missing_rate'] * 100:.2f}%")

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
channel_missing_pct = existence_stats["channel_missing_rate"] * 100.0
ax.bar(CHANNEL_LABELS, channel_missing_pct, color="#E45756", edgecolor="black")
ax.set_title("Estimated Missing-File Rate by Channel (sampled)")
ax.set_ylabel("Missing paths (%)")
ax.set_xlabel("Channel")
ax.set_ylim(0, max(channel_missing_pct.max() * 1.2, 1.0))
ax.grid(True, axis="y", alpha=0.25)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Visual Check: One Representative Sample

# %%
viz_row, viz_timestep, viz_paths = pick_visualization_sample(df_eda)

if viz_row is None:
    print("No fully available sample timestep found for FITS visualization.")
else:
    print("Selected sample:")
    print(f"  sample_id:   {viz_row['sample_id']}")
    print(f"  date:        {viz_row['date']}")
    print(f"  NOAA AR:     {viz_row['noaa_ar']}")
    print(f"  flare class: {viz_row['flare_class_name']}")
    print(f"  timestep:    T{viz_timestep}")

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()

    for idx, path_entry in enumerate(viz_paths):
        image = load_fits_2d(path_entry)
        wl = ts_config.CHANNEL_ORDER[idx]

        if wl == ts_config.HMI_WAVELENGTH:
            vmax = np.nanpercentile(np.abs(image), 99)
            vmin = -vmax
            cmap = "gray"
        else:
            vmin, vmax = np.nanpercentile(image, [1, 99])
            cmap = "magma"

        im = axes[idx].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axes[idx].set_title(f"{wl}", fontsize=10)
        axes[idx].axis("off")
        plt.colorbar(im, ax=axes[idx], fraction=0.045, pad=0.03)

    plt.suptitle(
        f"Sample {viz_row['sample_id']} | timestep T{viz_timestep} | channels {ts_config.NUM_CHANNELS}",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

# %%
if viz_row is not None and 171 in ts_config.CHANNEL_ORDER:
    channel_171_idx = ts_config.CHANNEL_ORDER.index(171)
    grid = parse_paths_grid(viz_row["paths"])
    temporal_paths_171 = []
    for t in range(min(ts_config.NUM_TIMESTEPS, len(grid))):
        row = grid[t]
        if channel_171_idx < len(row):
            path_entry = row[channel_171_idx]
            if path_entry not in (None, "", "None") and Path(str(path_entry)).exists():
                temporal_paths_171.append((t, path_entry))

    if len(temporal_paths_171) > 0:
        n_plot = min(6, len(temporal_paths_171))
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()
        for idx in range(6):
            axes[idx].axis("off")

        for idx in range(n_plot):
            t, path_entry = temporal_paths_171[idx]
            image = load_fits_2d(path_entry)
            vmin, vmax = np.nanpercentile(image, [1, 99])
            im = axes[idx].imshow(image, cmap="magma", vmin=vmin, vmax=vmax, origin="lower")
            axes[idx].set_title(f"171A - T{t}")
            axes[idx].axis("off")
            plt.colorbar(im, ax=axes[idx], fraction=0.045, pad=0.03)

        plt.suptitle("Temporal evolution (AIA 171A)", y=1.01)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## 9. Compact Status Summary

# %%
class_counts = df_eda["flare_class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)

print("=" * 72)
print("TIMESERIES DATASET STATUS SUMMARY")
print("=" * 72)
print(f"Samples: {len(df_eda):,}")
print(f"Date range: {df_eda['datetime'].min().date()} -> {df_eda['datetime'].max().date()}")
print(f"Unique NOAA ARs: {df_eda['noaa_ar'].nunique():,}")

print("\nClass distribution:")
for class_name in CLASS_NAMES:
    count = int(class_counts.get(class_name, 0))
    pct = safe_ratio(count, len(df_eda)) * 100.0
    print(f"  {class_name:>8}: {count:>7,} ({pct:5.1f}%)")

print("\nCompleteness:")
print(f"  Fully observed paths: {df_eda['is_fully_observed_paths'].mean() * 100:.2f}%")
print(f"  Mean missing path fraction: {df_eda['missing_fraction_paths'].mean() * 100:.2f}%")
print(f"  Estimated on-disk missing rate (sampled): {existence_stats['missing_rate'] * 100:.2f}%")
print("=" * 72)
