#!/usr/bin/env python3
"""Analyze flare catalog FITS availability, pre-filter corrupted files, and save a visual report."""

from __future__ import annotations

import os
import json
import logging
import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from arccnet.models import preprocessing_common as pp_common
from arccnet.models.flares import utils as flare_utils
from arccnet.models.flares.binary_classification import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("prefilter_fits_report")


@dataclass
class ValidationResult:
    path: str
    filename: str
    ok: bool
    error: str = ""


def _resolved_path(row: pd.Series) -> str | None:
    """Resolve preferred mapped FITS path from dataframe row."""
    resolved = row.get("resolved_path")
    if not pp_common.is_missing_path_value(resolved):
        return str(resolved)

    for column in ("path_image_cutout_hmi", "path_image_cutout_mdi"):
        value = row.get(column)
        if not pp_common.is_missing_path_value(value):
            return str(value)

    return None


def _validate_fits_file(path: str) -> ValidationResult:
    """Try decoding FITS data to catch compressed-stream corruption."""
    name = os.path.basename(path)
    try:
        _ = pp_common.load_fits_hdu_data(path, hdu_index=1)
        return ValidationResult(path=path, filename=name, ok=True)
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(path=path, filename=name, ok=False, error=f"{type(exc).__name__}: {exc}")


def _save_plot(report_dir: str, metrics: dict[str, int]) -> str:
    labels = ["rows_input", "rows_mapped_exists", "rows_corrupted", "rows_filtered"]
    values = [metrics[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.set_title("FITS Pre-filter Summary")
    ax.set_ylabel("Row count")
    ax.set_xlabel("Stage")
    ax.bar_label(bars)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    plot_path = os.path.join(report_dir, "summary_counts.png")
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default=config.DATA_FOLDER)
    parser.add_argument("--dataset-folder", default=config.CUTOUT_DATASET_FOLDER)
    parser.add_argument("--input-parquet", default=config.FLARES_PARQ)
    parser.add_argument(
        "--output-parquet",
        default="mag-pit-flare-dataset_1996-01-01_2023-01-01_dev_prefiltered.parq",
        help="Output parquet file name/path. If relative, saved under data-folder.",
    )
    parser.add_argument(
        "--report-dir",
        default="flares_prefilter_report",
        help="Directory for CSV/JSON/PNG report files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap for validated unique files (for quick diagnostics).",
    )
    args = parser.parse_args()

    input_path = os.path.join(args.data_folder, args.input_parquet)
    output_path = args.output_parquet
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.data_folder, output_path)

    report_dir = args.report_dir
    if not os.path.isabs(report_dir):
        report_dir = os.path.join(args.data_folder, report_dir)
    os.makedirs(report_dir, exist_ok=True)

    logger.info("Loading flare dataframe from %s", input_path)
    df = pd.read_parquet(input_path)
    rows_input = len(df)

    logger.info("Resolving existing FITS paths using project mapper...")
    mapped_df, missing_idx = flare_utils.check_fits_file_existence(df.copy(), args.data_folder, args.dataset_folder)
    mapped_df = mapped_df[mapped_df["file_exists"]].copy()
    rows_mapped_exists = len(mapped_df)

    mapped_df["resolved_path"] = mapped_df.apply(_resolved_path, axis=1)
    mapped_df = mapped_df[mapped_df["resolved_path"].notna()].copy()

    unique_files = sorted(mapped_df["resolved_path"].unique().tolist())
    if args.max_files is not None:
        unique_files = unique_files[: args.max_files]
        logger.warning("--max-files set: validating only first %d unique files", len(unique_files))

    logger.info("Validating %d unique FITS files", len(unique_files))

    bad_records: list[dict[str, str]] = []
    for path in tqdm(unique_files, desc="Validating FITS files", unit="file"):
        result = _validate_fits_file(path)
        if not result.ok:
            bad_records.append({"path": result.path, "filename": result.filename, "error": result.error})

    bad_df = pd.DataFrame(bad_records, columns=["path", "filename", "error"])
    bad_file_set = set(bad_df["path"].tolist())

    filtered_df = mapped_df[~mapped_df["resolved_path"].isin(bad_file_set)].copy()
    rows_filtered = len(filtered_df)
    rows_corrupted = rows_mapped_exists - rows_filtered

    filtered_df = filtered_df.drop(columns=["resolved_path"], errors="ignore")

    logger.info("Saving filtered parquet to %s", output_path)
    filtered_df.to_parquet(output_path, index=False)

    bad_csv_path = os.path.join(report_dir, "corrupted_fits_files.csv")
    bad_df.to_csv(bad_csv_path, index=False)

    metrics = {
        "rows_input": int(rows_input),
        "rows_missing_path": int(len(missing_idx)),
        "rows_mapped_exists": int(rows_mapped_exists),
        "rows_corrupted": int(rows_corrupted),
        "rows_filtered": int(rows_filtered),
        "unique_files_validated": int(len(unique_files)),
        "unique_bad_files": int(len(bad_file_set)),
    }

    metrics_path = os.path.join(report_dir, "summary_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_path = _save_plot(report_dir, metrics)

    logger.info("Done. Metrics: %s", metrics)
    logger.info("Report files:\n- %s\n- %s\n- %s", metrics_path, bad_csv_path, plot_path)


if __name__ == "__main__":
    main()
