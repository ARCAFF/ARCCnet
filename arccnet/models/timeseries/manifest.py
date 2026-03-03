import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REQUIRED_TIMESTEPS = 6


def parse_sample_dirname(dirname):
    """
    Parse: YYYY-MM-DD_NOAA_MagClass_McIntosh_Xb#_Mb#_Cb#_Xa#_Ma#_Ca#
    Example: 2011-01-03_11142_Beta_Dso_Xb0_Mb0_Cb0_Xa0_Ma0_Ca1
    """
    parts = dirname.split("_")
    if len(parts) < 10:
        return None

    try:
        return {
            "date": parts[0],
            "noaa_ar": int(parts[1]),
            "hale_class": parts[2],
            "mcintosh": parts[3],
            "xb": int(parts[4].replace("Xb", "")),
            "mb": int(parts[5].replace("Mb", "")),
            "cb": int(parts[6].replace("Cb", "")),
            "xa": int(parts[7].replace("Xa", "")),
            "ma": int(parts[8].replace("Ma", "")),
            "ca": int(parts[9].replace("Ca", "")),
        }
    except Exception as e:
        print(f"Failed to parse {dirname}: {e}")
        return None


def build_sample_record(sample_dir):
    """
    Build record for one sample directory.
    Returns dict with paths organized as [T=6][C=10] grid.
    """
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name

    metadata = parse_sample_dirname(sample_id)
    if metadata is None:
        return None

    csv_path = sample_dir / f"{sample_id}.csv"
    if not csv_path.exists():
        print(f"Warning: CSV not found for {sample_id}")
        return None

    df_csv = pd.read_csv(csv_path)

    aia_wavelengths = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]
    hmi_wavelength = 6173
    channel_order = aia_wavelengths + [hmi_wavelength]

    paths_grid = []  # Will be [T][C] list of file paths
    timestamps = []

    wavelength_groups = {}
    for _, row in df_csv.iterrows():
        wl = int(row["AIA wavelength"])
        aia_file = row["AIA files"]
        hmi_file = row["HMI files"]

        if wl == hmi_wavelength:
            file_path = hmi_file
        else:
            file_path = aia_file

        if wl not in wavelength_groups:
            wavelength_groups[wl] = []
        wavelength_groups[wl].append(sample_dir / file_path)

    timestep_count = min(len(wavelength_groups.get(wl, [])) for wl in channel_order)
    if timestep_count == 0:
        print(f"Warning: No valid timesteps for {sample_id}")
        return None
    if timestep_count < REQUIRED_TIMESTEPS:
        print(
            f"Warning: Incomplete sample {sample_id} has {timestep_count} timesteps (<{REQUIRED_TIMESTEPS}), skipping"
        )
        return None
    timestep_count = REQUIRED_TIMESTEPS

    for t in range(timestep_count):
        timestep_paths = []
        for wl in channel_order:
            if wl in wavelength_groups and t < len(wavelength_groups[wl]):
                timestep_paths.append(str(wavelength_groups[wl][t]))
            else:
                timestep_paths.append(None)
        paths_grid.append(timestep_paths)

        if channel_order[0] in wavelength_groups and t < len(wavelength_groups[channel_order[0]]):
            first_file = wavelength_groups[channel_order[0]][t]
            match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}\d{2}\d{2}Z)", first_file.name)
            if match:
                timestamps.append(match.group(1))
            else:
                timestamps.append(f"t{t}")

    # Legacy multi-label binary targets
    cplus = int(metadata["ca"] > 0 or metadata["ma"] > 0 or metadata["xa"] > 0)
    mplus = int(metadata["ma"] > 0 or metadata["xa"] > 0)
    xplus = int(metadata["xa"] > 0)

    # Multiclass label: Highest flare class in 24h window.
    # Label order: 0=No-flare, 1=C-only, 2=M+ (M or X)
    if metadata["ma"] > 0 or metadata["xa"] > 0:
        flare_class = 2  # M+ class (M or X)
    elif metadata["ca"] > 0:
        flare_class = 1  # C-class only
    else:
        flare_class = 0  # No post-window flare

    # Regression targets: log10 of flare counts (with smoothing)
    # Adding 1 to avoid log(0), then taking log10
    log_ca = float(np.log10(metadata["ca"] + 1))
    log_ma = float(np.log10(metadata["ma"] + 1))
    log_xa = float(np.log10(metadata["xa"] + 1))

    return {
        "sample_id": sample_id,
        "sample_path": str(sample_dir),
        "noaa_ar": metadata["noaa_ar"],
        "date": metadata["date"],
        "hale_class": metadata["hale_class"],
        "mcintosh": metadata["mcintosh"],
        "num_timesteps": timestep_count,
        "timestamps": timestamps,
        "paths": paths_grid,
        "xb": metadata["xb"],
        "mb": metadata["mb"],
        "cb": metadata["cb"],
        "xa": metadata["xa"],
        "ma": metadata["ma"],
        "ca": metadata["ca"],
        # Legacy binary targets
        "c_plus": cplus,
        "m_plus": mplus,
        "x_plus": xplus,
        # New targets
        "flare_class": flare_class,  # Multiclass: 0=NoFlare, 1=C, 2=M+
        "log_ca": log_ca,  # Regression: log10(C_count + 1)
        "log_ma": log_ma,  # Regression: log10(M_count + 1)
        "log_xa": log_xa,  # Regression: log10(X_count + 1)
    }


def build_dataset(root_dir, output_path=None, max_samples=None):
    """
    Scan root_dir for sample folders and build dataset manifest.

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing sample folders
    output_path : str or Path, optional
        Where to save manifest parquet file
    max_samples : int, optional
        Limit number of samples (for testing)

    Returns
    -------
    pd.DataFrame
        Dataset manifest dataframe
    """
    root_dir = Path(root_dir)
    sample_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])

    if max_samples:
        sample_dirs = sample_dirs[:max_samples]

    print(f"Building dataset from {len(sample_dirs)} samples...")

    records = []
    for sample_dir in tqdm(sample_dirs):
        record = build_sample_record(sample_dir)
        if record:
            records.append(record)

    df = pd.DataFrame(records)

    print(f"\nDataset built: {len(df)} valid samples")
    print(f"  NOAA ARs: {df['noaa_ar'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print("  Flare class distribution:")
    print(f"    No-flare:     {(df['flare_class'] == 0).sum()} ({(df['flare_class'] == 0).mean() * 100:.1f}%)")
    print(f"    C-class only: {(df['flare_class'] == 1).sum()} ({(df['flare_class'] == 1).mean() * 100:.1f}%)")
    print(f"    M+-class:     {(df['flare_class'] == 2).sum()} ({(df['flare_class'] == 2).mean() * 100:.1f}%)")
    print("  Regression targets (mean):")
    print(f"    log(Ca+1): {df['log_ca'].mean():.3f}")
    print(f"    log(Ma+1): {df['log_ma'].mean():.3f}")
    print(f"    log(Xa+1): {df['log_xa'].mean():.3f}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\nDataset manifest saved to: {output_path}")

    return df


def build_manifest(root_dir, output_path=None, max_samples=None):
    """Backward-compatible alias for build_dataset."""
    return build_dataset(root_dir=root_dir, output_path=output_path, max_samples=max_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, help="Dataset root directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    build_dataset(args.root_dir, args.output, args.max_samples)
