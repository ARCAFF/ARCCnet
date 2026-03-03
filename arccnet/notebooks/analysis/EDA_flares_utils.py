import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from arccnet.models import labels

FLARE_CLASSES = ["B", "C", "M", "X", "A"]
FLARE_COLORS = {
    "None": "#d3d3d3",  # Light gray for non-flaring
    "B": "#b3cde3",  # Light blue
    "C": "#6497b1",  # Medium blue
    "M": "#005b96",  # Dark blue
    "X": "#03396c",  # Very dark blue
    "A": "#011f4b",  # Almost black blue
}


def get_flare_data_by_magnetic_class(df):
    """
    Get flare data grouped by magnetic class.
    Returns DataFrame with flare counts per magnetic class.
    """
    mag_flare_data = df.groupby("magnetic_class")[FLARE_CLASSES].sum().reset_index()
    mag_flare_data["total_flares"] = mag_flare_data[FLARE_CLASSES].sum(axis=1)
    return mag_flare_data.sort_values("total_flares", ascending=False)


def classify_active_regions(df):
    """
    Classify active regions as flaring or non-flaring and identify highest flare class.
    Adds 'has_flares' and 'highest_flare' columns to the DataFrame.
    """
    # Mark active regions that produce flares
    df["has_flares"] = df[FLARE_CLASSES].sum(axis=1) > 0

    # Identify highest flare class for each active region
    df["highest_flare"] = "None"
    for cls in reversed(FLARE_CLASSES):
        df.loc[(df[cls] > 0), "highest_flare"] = cls

    return df


def create_stacked_bar_chart(
    df, x, y_series, colors, ax=None, show_totals=True, percentage=False, y_off=10, title=None, xlabel=None, ylabel=None
):
    """
    Create a stacked bar chart.

    Parameters:
    - df: DataFrame containing the data
    - x: Column name for x-axis categories
    - y_series: List of column names for stacked values
    - colors: Dictionary mapping y_series to colors
    - ax: Matplotlib axis to plot on
    - show_totals: Whether to show total labels
    - percentage: Whether to show percentages instead of raw values
    - title, xlabel, ylabel: Chart labels
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    data = df.copy()
    if percentage:
        totals = data[y_series].sum(axis=1)
        for col in y_series:
            data[col] = data[col] / totals * 100

    bottom = np.zeros(len(data))
    for col in y_series:
        ax.bar(data[x], data[col], bottom=bottom, label=col, color=colors.get(col, "#333333"), edgecolor="black")

        # Add value labels for bars that are large enough
        threshold = 5 if percentage else 20
        for i, value in enumerate(data[col]):
            if value > threshold:
                text_color = "white" if col in ["M", "X", "A"] else "black"
                label = f"{value:.1f}%" if percentage else str(int(value))
                ax.text(
                    i,
                    bottom[i] + value / 2,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=10,
                )

        bottom += data[col].values

    # Add total annotations
    if show_totals:
        if percentage:
            for i in range(len(data)):
                ax.text(i, 101, "100%", ha="center", va="bottom", fontsize=11)
        else:
            totals = data[y_series].sum(axis=1)
            for i, total in enumerate(totals):
                ax.text(i, total + y_off, f"Total Flares: {int(total)}", ha="center", va="bottom", fontsize=11)

    # Set labels
    if title:
        ax.set_title(title, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    ax.tick_params(axis="both", labelsize=11)
    if percentage:
        ax.set_ylim(0, 105)

    return ax


def analyze_flaring_vs_nonflaring(df):
    """Analyze flaring vs non-flaring active regions by magnetic class"""
    df = classify_active_regions(df)

    magnetic_classes = sorted(filter(lambda x: x is not None, df["magnetic_class"].unique()))

    grouped = df.groupby(["magnetic_class", "highest_flare"]).size().unstack(fill_value=0)

    available_classes = ["None"] + [cls for cls in FLARE_CLASSES if cls in grouped.columns]

    # Reindex with sorted magnetic classes
    grouped = grouped.reindex(magnetic_classes)

    grouped["total"] = grouped.sum(axis=1)

    for cls in available_classes:
        if cls not in grouped.columns:
            grouped[cls] = 0

    grouped = grouped[available_classes + ["total"]]

    # Convert magnetic classes to Greek labels for plotting
    grouped_plot = grouped.reset_index()
    grouped_plot["greek_labels"] = grouped_plot["magnetic_class"].apply(lambda x: labels.convert_to_greek_label([x])[0])

    with plt.style.context("seaborn-v0_8-darkgrid"):
        create_stacked_bar_chart(
            grouped_plot,
            "greek_labels",
            available_classes,
            colors=FLARE_COLORS,
            percentage=True,
            show_totals=False,
            title="Percentage of Active Regions by Magnetic Class and Flaring Activity",
            xlabel="Magnetic Class",
            ylabel="Percentage of Active Regions",
        )

        legend_labels = ["Non-flaring"] if "None" in available_classes else []
        for cls in FLARE_CLASSES:
            if cls in available_classes:
                legend_labels.append(f"{cls}-class flare")

        plt.show()

    # Create summary table
    summary = grouped.copy()
    summary = summary.rename(columns={"total": "Total ARs"})

    # Add flaring and non-flaring columns
    summary["Non-flaring ARs"] = summary["None"] if "None" in summary.columns else 0
    summary["Flaring ARs"] = summary["Total ARs"] - summary["Non-flaring ARs"]
    summary["% Flaring"] = (summary["Flaring ARs"] / summary["Total ARs"] * 100).round(1)

    # Reorder columns - flare classes should be organized as individual columns
    cols = ["Total ARs", "Flaring ARs", "Non-flaring ARs", "% Flaring"] + available_classes
    summary = summary[cols]

    # Add totals row
    totals = summary.sum().to_dict()
    totals["% Flaring"] = round(totals["Flaring ARs"] / totals["Total ARs"] * 100, 1)
    summary.loc["Total"] = pd.Series(totals)

    # Convert the index to Greek labels for the display
    greek_indices = [labels.convert_to_greek_label([idx])[0] if idx != "Total" else idx for idx in summary.index]
    summary.index = greek_indices

    return summary
