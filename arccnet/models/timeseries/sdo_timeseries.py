import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from astropy.io import fits

from .path_utils import parse_paths_grid


class SDOTimeseriesDataset(Dataset):
    """
    PyTorch Dataset for SDO timeseries data.

    Parameters
    ----------
    manifest : pd.DataFrame
        Manifest dataframe with paths and labels
    split : str
        'train', 'val', or 'test'
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
        resize=(256, 512),
        augment=False,
        norm_stats=None,
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotation_degrees=10,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.split = split
        self.resize = resize
        self.augment = augment and (split == "train")
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_degrees = rotation_degrees

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

        timesteps = []
        for t_paths in paths:
            channels = []
            for c_path in t_paths:
                if c_path is None or c_path == "None":
                    if self.resize:
                        channels.append(np.zeros((self.resize[0], self.resize[1]), dtype=np.float32))
                    else:
                        channels.append(np.zeros((400, 800), dtype=np.float32))
                else:
                    img = self._load_fits(c_path)
                    channels.append(img)
            timesteps.append(np.stack(channels, axis=0))

        x = np.stack(timesteps, axis=0)  # (T, C, H, W)
        x = torch.from_numpy(x).float()

        x = self._normalize(x)

        if self.augment:
            x = self._augment(x)

        y = torch.tensor([row["c_plus"], row["m_plus"], row["x_plus"]], dtype=torch.float32)

        meta = {
            "sample_id": row["sample_id"],
            "noaa_ar": row["noaa_ar"],
            "date": row["date"],
        }

        return {"x": x, "y": y, "meta": meta}

    def _load_fits(self, path):
        """Load FITS file and return 2D array."""
        try:
            with fits.open(path) as hdul:
                data = hdul[1].data.astype(np.float32)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                if self.resize:
                    data_t = torch.from_numpy(data).unsqueeze(0)
                    data_t = TF.resize(data_t, self.resize, antialias=True)
                    data = data_t.squeeze(0).numpy()

                return data
        except Exception as e:
            print(f"Error loading {path}: {e}")
            if self.resize:
                return np.zeros((self.resize[0], self.resize[1]), dtype=np.float32)
            return np.zeros((400, 800), dtype=np.float32)

    @staticmethod
    def _parse_paths(raw_paths):
        """Safely parse serialized path grids from manifest rows."""
        return parse_paths_grid(raw_paths)

    def _compute_norm_stats(self, max_samples=50):
        """Compute per-channel mean and std from subset of data."""
        num_channels = 10
        channel_means = []
        channel_stds = []

        sample_indices = np.random.choice(len(self), min(max_samples, len(self)), replace=False)

        for c in range(num_channels):
            values = []
            for idx in sample_indices:
                row = self.manifest.iloc[idx]
                paths = self._parse_paths(row["paths"])

                for t_paths in paths[:2]:  # Sample first 2 timesteps
                    if t_paths[c] and t_paths[c] != "None":
                        img = self._load_fits(t_paths[c])
                        values.append(img.flatten())

            if values:
                all_values = np.concatenate(values)
                p1, p99 = np.percentile(all_values, [1, 99])
                clipped = np.clip(all_values, p1, p99)
                channel_means.append(float(np.mean(clipped)))
                channel_stds.append(float(np.std(clipped)) + 1e-6)
            else:
                channel_means.append(0.0)
                channel_stds.append(1.0)

        stats = {
            "mean": channel_means,
            "std": channel_stds,
        }
        print(f"Normalization stats computed: {len(channel_means)} channels")
        return stats

    def _normalize(self, x):
        """Normalize each channel independently."""
        T, C, H, W = x.shape
        for c in range(C):
            mean = self.norm_stats["mean"][c]
            std = self.norm_stats["std"][c]

            p1, p99 = torch.quantile(x[:, c].flatten(), torch.tensor([0.01, 0.99]))
            x[:, c] = torch.clamp(x[:, c], p1, p99)
            x[:, c] = (x[:, c] - mean) / std

        return x

    def _augment(self, x):
        """Apply spatial augmentations to all timesteps consistently."""
        T, C, H, W = x.shape

        if np.random.rand() < self.hflip_prob:
            x = torch.flip(x, dims=[3])

        if np.random.rand() < self.vflip_prob:
            x = torch.flip(x, dims=[2])

        if self.rotation_degrees > 0:
            angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
            x_aug = []
            for t in range(T):
                x_t = TF.rotate(x[t], angle)
                x_aug.append(x_t)
            x = torch.stack(x_aug, dim=0)

        return x

    def get_norm_stats(self):
        """Return normalization statistics for use in other splits."""
        return self.norm_stats
