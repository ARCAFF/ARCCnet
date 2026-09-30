"""Derive geometric quality indicators for the full-disk detection dataset."""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

REQUIRED_COLUMNS = (
    "filtered",
    "instrument",
    "bottom_left_cutout",
    "bottom_left_cutout.mask",
    "top_right_cutout",
    "top_right_cutout.mask",
    "processed_path_image_mag",
    "processed_path_image_mag.mask",
)


def resolve_magnetogram(path, dataset_root):
    """Map a catalogue build-time path to its location in a release directory."""
    marker = "/03_processed/"
    if marker not in path:
        raise ValueError(f"Magnetogram path lacks {marker}: {path}")
    relative = path.split(marker, 1)[1]
    return Path(dataset_root) / "03_processed" / relative


def disk_geometry(header):
    """Return linear pixel-to-arcsecond geometry and apparent solar radius."""
    wcs = WCS(header, naxis=2)
    scale = np.asarray(wcs.pixel_scale_matrix, dtype=float)
    units = [u.Unit(str(unit or "arcsec")).to(u.arcsec) for unit in wcs.wcs.cunit]
    scale *= np.asarray(units)[:, None]
    reference = np.asarray(wcs.wcs.crval, dtype=float) * units
    origin = np.asarray(wcs.wcs.crpix, dtype=float) - 1
    radius = float(header["RSUN_OBS"])
    if not np.all(np.isfinite(scale)) or not np.all(np.isfinite(reference)) or radius <= 0:
        raise ValueError("Invalid WCS or RSUN_OBS in magnetogram FITS header")
    return scale, reference, origin, radius


def box_indicators(bottom_left, top_right, geometry, small_fraction=0.01, limb_fraction=0.9):
    """Measure box geometry relative to the apparent solar disk.

    ``small_box`` flags either dimension below ``small_fraction`` of the solar
    diameter; ``near_limb`` flags centres beyond ``limb_fraction`` of the radius;
    ``crosses_limb`` flags any corner outside the disk. These are geometric
    cautions, not physical validity labels.
    """
    x0, y0 = np.asarray(bottom_left, dtype=float)
    x1, y1 = np.asarray(top_right, dtype=float)
    if not np.all(np.isfinite([x0, y0, x1, y1])) or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid bounding-box coordinates")

    scale, reference, origin, radius = geometry
    corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
    offsets = (corners - origin) @ scale.T + reference
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    centre_offset = (centre - origin) @ scale.T + reference
    centre_rho = float(np.linalg.norm(centre_offset) / radius)
    width_fraction = float(np.linalg.norm(scale[:, 0]) * (x1 - x0) / (2 * radius))
    height_fraction = float(np.linalg.norm(scale[:, 1]) * (y1 - y0) / (2 * radius))
    return {
        "width_over_diameter": width_fraction,
        "height_over_diameter": height_fraction,
        "centre_rho": centre_rho,
        "centre_mu": float(np.sqrt(max(0.0, 1.0 - centre_rho**2))),
        "small_width": width_fraction < small_fraction,
        "small_height": height_fraction < small_fraction,
        "small_box": min(width_fraction, height_fraction) < small_fraction,
        "near_limb": centre_rho > limb_fraction,
        "crosses_limb": bool(np.any(np.linalg.norm(offsets, axis=1) > radius)),
    }


def derive_quality(catalogue, dataset_root, small_fraction=0.01, limb_fraction=0.9):
    """Read FITS headers and return one indicator row per clean catalogue row."""
    frame = pd.read_parquet(catalogue)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Catalogue lacks columns: {sorted(missing)}")
    frame = frame.reset_index(drop=True)
    clean = frame.loc[~frame["filtered"]]
    if clean.empty:
        raise ValueError("Catalogue contains no clean detections")
    mask_columns = ("bottom_left_cutout.mask", "top_right_cutout.mask", "processed_path_image_mag.mask")
    if any(np.asarray(mask, dtype=bool).any() for column in mask_columns for mask in clean[column]):
        raise ValueError("A clean detection has a masked box or magnetogram path")

    records = []
    for path, group in clean.groupby("processed_path_image_mag", sort=False):
        actual = resolve_magnetogram(path, dataset_root)
        if not actual.is_file():
            raise FileNotFoundError(f"Missing processed magnetogram: {actual}")
        geometry = disk_geometry(fits.getheader(actual, ext=1))
        for row_number, row in group.iterrows():
            try:
                indicators = box_indicators(
                    row["bottom_left_cutout"],
                    row["top_right_cutout"],
                    geometry,
                    small_fraction,
                    limb_fraction,
                )
            except ValueError as exc:
                raise ValueError(f"Invalid clean detection at catalogue row {row_number}") from exc
            records.append(
                {
                    "catalogue_row": int(row_number),
                    "instrument": row["instrument"],
                    "processed_path_image_mag": path,
                    **indicators,
                }
            )
    result = pd.DataFrame.from_records(records).sort_values("catalogue_row").reset_index(drop=True)
    result["quality_caution"] = result[["small_box", "near_limb", "crosses_limb"]].any(axis=1)
    if len(result) != len(clean):
        raise RuntimeError("Not every clean detection received quality indicators")
    return result


def summary(indicators):
    """Summarize the indicators without assuming that their groups are disjoint."""
    return {
        "clean_detections": len(indicators),
        "small_width": int(indicators["small_width"].sum()),
        "small_height": int(indicators["small_height"].sum()),
        "small_box": int(indicators["small_box"].sum()),
        "near_limb": int(indicators["near_limb"].sum()),
        "crosses_limb": int(indicators["crosses_limb"].sum()),
        "quality_caution_union": int(indicators["quality_caution"].sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=Path, help="Released full-disk Parquet catalogue")
    parser.add_argument("dataset_root", type=Path, help="Root of the released dataset")
    parser.add_argument("--output", type=Path, help="Optional CSV sidecar outside the release directory")
    parser.add_argument(
        "--small-fraction",
        type=float,
        default=0.01,
        help="Minimum box dimension as a fraction of apparent solar diameter",
    )
    parser.add_argument(
        "--limb-fraction",
        type=float,
        default=0.9,
        help="Centre-distance threshold as a fraction of apparent solar radius",
    )
    args = parser.parse_args(argv)
    if not 0 < args.small_fraction < 1 or not 0 < args.limb_fraction < 1:
        parser.error("Fractions must be between zero and one")
    if args.output:
        if args.output.resolve().is_relative_to(args.dataset_root.resolve()):
            parser.error("Output sidecar must be outside the released dataset directory")
        if args.output.resolve() == args.catalogue.resolve():
            parser.error("Output sidecar cannot replace the input catalogue")
        if args.output.exists():
            parser.error(f"Output file already exists: {args.output}")

    indicators = derive_quality(args.catalogue, args.dataset_root, args.small_fraction, args.limb_fraction)
    if args.output:
        indicators.to_csv(args.output, index=False, mode="x")
    print(json.dumps(summary(indicators), indent=2))


if __name__ == "__main__":
    main()
