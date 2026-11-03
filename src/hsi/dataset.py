import numpy as np
import torch
from scipy.ndimage import binary_erosion
from torch.utils.data import Dataset


class RandomSpatialPatchDataset(Dataset):
    """Samples random reflection-padded patches from valid pixels."""

    def __init__(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        patch_size: int = 9,
        samples_per_epoch: int = 40_000,
        seed: int | None = 42,
    ) -> None:
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

        self.padded = np.pad(
            data,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode="reflect",
        )

        coords = np.argwhere(valid_mask)
        if len(coords) == 0:
            raise ValueError("Provided valid_mask does not contain any valid pixels.")
        self.valid_coords = coords

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> torch.Tensor:
        rand_idx = self.rng.integers(0, len(self.valid_coords))
        r, c = self.valid_coords[rand_idx]

        patch = self.padded[r : r + self.patch_size, c : c + self.patch_size, :]
        return torch.from_numpy(patch).permute(2, 0, 1).contiguous().float()


class SpatialPatchDataset(Dataset):
    """Padded spatial-spectral patch dataset anchored at labeled coordinates."""

    def __init__(self, data: np.ndarray, gt: np.ndarray, patch_size: int = 9) -> None:
        self.patch_size = patch_size
        pad = patch_size // 2

        padded = np.pad(
            data,
            pad_width=((pad, pad), (pad, pad), (0, 0)),
            mode="reflect",
        )
        self.padded_data = torch.from_numpy(
            np.ascontiguousarray(padded.transpose(2, 0, 1))
        ).float()

        y_coords, x_coords = np.where(gt > 0)
        self.y_coords = y_coords.astype(np.int64)
        self.x_coords = x_coords.astype(np.int64)
        self.labels = torch.from_numpy(gt[y_coords, x_coords] - 1).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        y = int(self.y_coords[idx])
        x = int(self.x_coords[idx])
        patch = self.padded_data[:, y : y + self.patch_size, x : x + self.patch_size]
        return patch, self.labels[idx]


class SSLNeighborhoodDataset(Dataset):
    """Extracts patches where the full spatial neighborhood consists of valid pixels."""

    def __init__(
        self,
        data: np.ndarray,
        patch_size: int = 9,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        self.data = torch.from_numpy(data).float().permute(2, 0, 1).contiguous()
        self.offset = patch_size // 2
        _, h, w = self.data.shape

        interior = np.zeros((h, w), dtype=bool)
        interior[self.offset : h - self.offset, self.offset : w - self.offset] = True

        if valid_mask is not None:
            footprint = np.ones((patch_size, patch_size), dtype=bool)
            eroded = binary_erosion(valid_mask, structure=footprint, border_value=False)
            interior &= eroded if np.any(interior & eroded) else valid_mask

        y_idx, x_idx = np.where(interior)
        self.y_coords = y_idx.astype(np.int64)
        self.x_coords = x_idx.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y_coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        y = int(self.y_coords[idx])
        x = int(self.x_coords[idx])
        patch = self.data[
            :,
            y - self.offset : y + self.offset + 1,
            x - self.offset : x + self.offset + 1,
        ]
        return patch, 0


class PixelInferenceDataset(Dataset):
    """Generates reflection-padded patches centered on every pixel coordinate."""

    def __init__(self, data: np.ndarray, patch_size: int = 9) -> None:
        self.patch_size = patch_size
        self.offset = patch_size // 2
        self.h, self.w, _ = data.shape

        padded = np.pad(
            data,
            ((self.offset, self.offset), (self.offset, self.offset), (0, 0)),
            mode="reflect",
        )
        self.padded_data = torch.from_numpy(
            np.ascontiguousarray(padded.transpose(2, 0, 1))
        ).float()

    def __len__(self) -> int:
        return self.h * self.w

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[int, int]]:
        y, x = divmod(idx, self.w)
        patch = self.padded_data[:, y : y + self.patch_size, x : x + self.patch_size]
        return patch, (y, x)


PaviaPatchDataset = SpatialPatchDataset
PaviaSSLDataset = SSLNeighborhoodDataset
PaviaInferenceDataset = PixelInferenceDataset
