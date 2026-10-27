import numpy as np
import torch
from torch.utils.data import Dataset


class PaviaPatchDataset(Dataset):
    """Extracts reflection-padded spatial-spectral patches at labeled coordinates."""

    def __init__(self, data: np.ndarray, gt: np.ndarray, patch_size: int = 9) -> None:
        self.patch_size = patch_size
        self.pad = patch_size // 2

        padded = np.pad(
            data,
            pad_width=((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
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


class PaviaSSLDataset(Dataset):
    """Extracts unlabeled patches across valid spatial coordinates for pretraining."""

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
            interior &= valid_mask

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


class PaviaInferenceDataset(Dataset):
    """Generates reflection-padded patches for every pixel in the scene."""

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
