import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from astropy.io import fits

from .config import (
    NORM_STATS_MAX_PIXELS_PER_IMAGE,
    NORM_STATS_MAX_SAMPLES,
    NORM_STATS_MAX_TIMESTEPS,
    NUM_CHANNELS,
    NUM_TIMESTEPS,
    SEED,
    TASK_TYPE,
    TIMESTEP_SELECTION,
)
from .path_utils import parse_paths_grid


class SDOTimeseriesDataset(Dataset):
    """
    PyTorch Dataset for SDO timeseries data.

    Parameters
    ----------
    manifest : pd.DataFrame
        Dataset manifest with paths and labels
    split : str
        'train', 'val', or 'test'
    task_type : str
        'multiclass' or 'regression' (default: from config.TASK_TYPE)
    resize : tuple, optional
        (H, W) to resize images
    augment : bool
        Whether to apply augmentations
    norm_stats : dict, optional
        Normalization statistics per channel
    """

    def __init__(
        self,
        manifest,
        split="train",
        task_type=None,
        resize=(256, 512),
        augment=False,
        norm_stats=None,
        num_timesteps=NUM_TIMESTEPS,
        timestep_selection=TIMESTEP_SELECTION,
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotation_degrees=10,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.split = split
        self.task_type = task_type or TASK_TYPE
        self.resize = resize
        self.augment = augment and (split == "train")
        self.expected_timesteps = max(1, int(num_timesteps))
        self.timestep_selection = str(timestep_selection).strip().lower()
        if self.timestep_selection not in {"first", "last"}:
            raise ValueError(
                f"Unsupported timestep_selection={self.timestep_selection!r}. Expected one of: 'first', 'last'."
            )
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_degrees = rotation_degrees
        self.processed_roots = self._discover_processed_roots()
        self._missing_file_log_limit = 25
        self._missing_file_log_count = 0

        if norm_stats is None:
            print(f"Computing normalization stats for {split} split...")
            self.norm_stats = self._compute_norm_stats()
        else:
            self.norm_stats = norm_stats

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        paths = self._parse_paths(row["paths"])
        selected_timesteps = self._select_timesteps(paths, max_timesteps=self.expected_timesteps)
        if self.resize:
            image_shape = (self.resize[0], self.resize[1])
        else:
            image_shape = (400, 800)

        timesteps = []
        timestep_mask = []
        for t_paths in selected_timesteps:
            if isinstance(t_paths, np.ndarray):
                t_paths = t_paths.tolist()
            channels = []
            has_valid_channel = False
            for c_path in t_paths[:NUM_CHANNELS]:
                if c_path is None or c_path == "None":
                    channels.append(np.zeros(image_shape, dtype=np.float32))
                else:
                    img = self._load_fits(c_path)
                    channels.append(img)
                    has_valid_channel = True
            while len(channels) < NUM_CHANNELS:
                channels.append(np.zeros(image_shape, dtype=np.float32))
            timesteps.append(np.stack(channels, axis=0))
            timestep_mask.append(bool(has_valid_channel))

        while len(timesteps) < self.expected_timesteps:
            blank = np.zeros((NUM_CHANNELS, image_shape[0], image_shape[1]), dtype=np.float32)
            timesteps.append(blank)
            timestep_mask.append(False)

        x = np.stack(timesteps, axis=0)  # (T, C, H, W)
        x = torch.from_numpy(x).float()
        mask = torch.tensor(timestep_mask, dtype=torch.bool)

        x = self._normalize(x)

        if self.augment:
            x = self._augment(x)

        # Get targets based on task type
        if self.task_type == "multiclass":
            # Single class label: 0=No-flare, 1=C, 2=M+ (M or X)
            y = torch.tensor(row["flare_class"], dtype=torch.long)
        elif self.task_type == "regression":
            # Regression targets: [log(Ca+1), log(Ma+1), log(Xa+1)]
            y = torch.tensor([row["log_ca"], row["log_ma"], row["log_xa"]], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        meta = {
            "sample_id": row["sample_id"],
            "noaa_ar": row["noaa_ar"],
            "date": row["date"],
            "flare_class": row["flare_class"],
            "xa": row["xa"],
            "ma": row["ma"],
            "ca": row["ca"],
        }

        return {"x": x, "y": y, "mask": mask, "meta": meta}

    def _parse_paths(self, raw_paths):
        """Safely parse serialized path grids from manifest."""
        return parse_paths_grid(raw_paths)

    def _select_timesteps(self, paths, max_timesteps=None):
        """
        Select timesteps from the parsed path grid according to runtime strategy.

        By default this preserves the current behavior (first N timesteps). PIT mode
        can switch to the latest window via ``timestep_selection="last"``.
        """
        if isinstance(paths, np.ndarray):
            paths = paths.tolist()
        paths = list(paths)

        if max_timesteps is None:
            max_timesteps = self.expected_timesteps
        max_timesteps = max(1, int(max_timesteps))

        if self.timestep_selection == "last":
            return paths[-max_timesteps:]
        return paths[:max_timesteps]

    def _discover_processed_roots(self):
        """
        Find candidate 03_processed roots to repair broken symlink targets.

        This handles datasets where cutout symlinks point to ../../../../03_processed/*
        but the actual processed tree lives under a nested folder, e.g.
        /ARCAFF/data/timeseries/arcnet-timeseries-*/03_processed.
        """
        roots = []
        if self.manifest.empty or "sample_path" not in self.manifest.columns:
            return roots

        sample_path = Path(str(self.manifest.iloc[0]["sample_path"]))
        timeseries_root = None
        for parent in sample_path.parents:
            if parent.name == "timeseries":
                timeseries_root = parent
                break

        if timeseries_root is None:
            timeseries_root = sample_path

        direct = timeseries_root / "03_processed"
        if direct.exists():
            roots.append(direct)

        for candidate in sorted(timeseries_root.glob("*/03_processed")):
            if candidate.exists():
                roots.append(candidate)

        # Optional explicit override(s), colon-separated.
        env_roots = os.getenv("ARCAFF_TIMESERIES_PROCESSED_ROOT", "")
        for token in env_roots.split(":"):
            token = token.strip()
            if not token:
                continue
            candidate = Path(token)
            if candidate.exists():
                roots.append(candidate)

        # Broader fallback search under /ARCAFF/data for relocated processed archives.
        data_root = Path("/ARCAFF/data")
        if data_root.exists():
            for pattern in ("*/03_processed", "*/*/03_processed"):
                for candidate in sorted(data_root.glob(pattern)):
                    if candidate.exists():
                        roots.append(candidate)

        # De-duplicate while preserving order.
        dedup = []
        seen = set()
        for root in roots:
            root_str = str(root)
            if root_str not in seen:
                dedup.append(root)
                seen.add(root_str)
        return dedup

    def _resolve_existing_path(self, path):
        """
        Resolve potentially broken symlink paths into existing files.
        """
        p = Path(str(path))
        if p.exists():
            return p

        # For broken symlinks, inspect target path and try alternate 03_processed roots.
        if p.is_symlink():
            try:
                link_target = p.readlink()
                candidate = (p.parent / link_target).resolve(strict=False)
                if candidate.exists():
                    return candidate
                repaired = self._repair_processed_root(candidate)
                if repaired is not None:
                    return repaired
            except Exception:
                pass

        # Non-symlink path fallback.
        repaired = self._repair_processed_root(p)
        if repaired is not None:
            return repaired

        return p

    def _repair_processed_root(self, candidate):
        """Repair paths containing /03_processed/ by redirecting to discovered roots."""
        candidate_str = str(candidate)
        token = f"{Path('/').as_posix()}03_processed{Path('/').as_posix()}"
        if token not in candidate_str:
            return None

        suffix = candidate_str.split(token, 1)[1]
        for processed_root in self.processed_roots:
            repaired = processed_root / suffix
            if repaired.exists():
                return repaired
        return None

    def _load_fits(self, path):
        """Load FITS file and return 2D array."""
        resolved_path = self._resolve_existing_path(path)
        try:
            with fits.open(resolved_path) as hdul:
                hdu_index = 1 if len(hdul) > 1 and hdul[1].data is not None else 0
                data = hdul[hdu_index].data.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                if self.resize:
                    data_t = torch.from_numpy(data).unsqueeze(0)
                    data_t = TF.resize(data_t, self.resize, antialias=True)
                    data = data_t.squeeze(0).numpy()

                return data
        except Exception as e:
            if self._missing_file_log_count < self._missing_file_log_limit:
                print(f"Error loading {path} (resolved: {resolved_path}): {e}")
                self._missing_file_log_count += 1
                if self._missing_file_log_count == self._missing_file_log_limit:
                    print("Further FITS load errors suppressed for this dataset instance.")
            if self.resize:
                return np.zeros((self.resize[0], self.resize[1]), dtype=np.float32)
            return np.zeros((400, 800), dtype=np.float32)

    def _compute_norm_stats(self, max_samples=None, max_timesteps=None, max_pixels_per_image=None):
        """Compute per-channel mean/std using a representative subset of samples and timesteps."""
        if max_samples is None:
            max_samples = max(1, int(NORM_STATS_MAX_SAMPLES))
        if max_timesteps is None:
            max_timesteps = max(1, int(NORM_STATS_MAX_TIMESTEPS))
        max_timesteps = max(1, min(int(max_timesteps), int(self.expected_timesteps)))
        if max_pixels_per_image is None:
            max_pixels_per_image = int(NORM_STATS_MAX_PIXELS_PER_IMAGE)
        if max_pixels_per_image <= 0:
            max_pixels_per_image = None

        num_channels = NUM_CHANNELS
        channel_means = []
        channel_stds = []
        clip_lows = []
        clip_highs = []

        rng = np.random.default_rng(SEED)
        sample_count = min(int(max_samples), len(self))
        sample_indices = rng.choice(len(self), sample_count, replace=False)

        for c in range(num_channels):
            values = []
            for idx in sample_indices:
                row = self.manifest.iloc[idx]
                paths = self._parse_paths(row["paths"])
                selected_paths = self._select_timesteps(paths, max_timesteps=max_timesteps)

                for t_paths in selected_paths:
                    if c < len(t_paths) and t_paths[c] and t_paths[c] != "None":
                        img = self._load_fits(t_paths[c])
                        pixels = img.reshape(-1)
                        if max_pixels_per_image is not None and pixels.size > max_pixels_per_image:
                            sel = rng.choice(pixels.size, max_pixels_per_image, replace=False)
                            pixels = pixels[sel]
                        values.append(pixels.astype(np.float32, copy=False))

            if values:
                all_values = np.concatenate(values)
                p1, p99 = np.percentile(all_values, [1, 99])
                clipped = np.clip(all_values, p1, p99)
                channel_means.append(float(np.mean(clipped)))
                channel_stds.append(float(np.std(clipped)) + 1e-6)
                if c == NUM_CHANNELS - 1:
                    # Preserve signed magnetic structure from HMI with symmetric clip.
                    abs_clip = float(max(abs(p1), abs(p99)))
                    p1, p99 = -abs_clip, abs_clip
                clip_lows.append(float(p1))
                clip_highs.append(float(p99))
            else:
                channel_means.append(0.0)
                channel_stds.append(1.0)
                clip_lows.append(-1.0)
                clip_highs.append(1.0)

        stats = {
            "mean": channel_means,
            "std": channel_stds,
            "clip_low": clip_lows,
            "clip_high": clip_highs,
        }
        print(
            "Normalization stats computed: "
            f"{len(channel_means)} channels (samples={sample_count}, timesteps={max_timesteps}, "
            f"max_pixels_per_image={max_pixels_per_image if max_pixels_per_image is not None else 'all'})"
        )
        return stats

    def _normalize(self, x):
        """Normalize each channel independently."""
        _, C, _, _ = x.shape
        clip_lows = self.norm_stats.get("clip_low", [None] * C)
        clip_highs = self.norm_stats.get("clip_high", [None] * C)
        for c in range(C):
            mean = self.norm_stats["mean"][c]
            std = self.norm_stats["std"][c]
            clip_low = clip_lows[c]
            clip_high = clip_highs[c]

            if clip_low is not None and clip_high is not None:
                x[:, c] = torch.clamp(x[:, c], clip_low, clip_high)
            x[:, c] = (x[:, c] - mean) / std

        return x

    def _augment(self, x):
        """Apply spatial augmentations to all timesteps consistently."""
        num_timesteps = x.shape[0]

        if np.random.rand() < self.hflip_prob:
            x = torch.flip(x, dims=[3])

        if np.random.rand() < self.vflip_prob:
            x = torch.flip(x, dims=[2])

        if self.rotation_degrees > 0:
            angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
            x_aug = []
            for t in range(num_timesteps):
                x_t = TF.rotate(x[t], angle)
                x_aug.append(x_t)
            x = torch.stack(x_aug, dim=0)

        return x

    def get_norm_stats(self):
        """Return normalization statistics for use in other splits."""
        return self.norm_stats
