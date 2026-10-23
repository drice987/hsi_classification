import numpy as np
import torch
from torch.utils.data import Dataset


class PaviaPatchDataset(Dataset):
    """Extracts reflection-padded spatial-spectral patches at labeled coordinates."""

    def __init__(self, data: np.ndarray, gt: np.ndarray, patch_size: int = 9) -> None:
        self.patch_size = patch_size
        self.pad = patch_size // 2

        self.padded_data = np.pad(
            data,
            pad_width=((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode="reflect",
        )

        y_coords, x_coords = np.where(gt > 0)
        self.coordinates = list(zip(y_coords, x_coords))
        self.labels = gt[y_coords, x_coords] - 1

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        y, x = self.coordinates[idx]
        y_pad, x_pad = y + self.pad, x + self.pad

        patch = self.padded_data[
            y_pad - self.pad : y_pad + self.pad + 1,
            x_pad - self.pad : x_pad + self.pad + 1,
            :,
        ]
        patch = np.transpose(patch, (2, 0, 1))

        return (
            torch.tensor(patch, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


class PaviaSSLDataset(Dataset):
    """Extracts unlabeled patches across a spatial slice for self-supervised pretraining."""

    def __init__(self, data: np.ndarray, patch_size: int = 9) -> None:
        self.data = torch.from_numpy(data).float().permute(2, 0, 1)
        self.offset = patch_size // 2
        _, h, w = self.data.shape

        self.coords = [
            (y, x)
            for y in range(self.offset, h - self.offset)
            for x in range(self.offset, w - self.offset)
        ]

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        y, x = self.coords[idx]
        patch = self.data[
            :,
            y - self.offset : y + self.offset + 1,
            x - self.offset : x + self.offset + 1,
        ]
        return patch, 0


class PaviaInferenceDataset(Dataset):
    """Generates reflection-padded patches for every pixel in the scene."""

    def __init__(self, data: np.ndarray, patch_size: int = 9) -> None:
        self.offset = patch_size // 2
        self.padded_data = np.pad(
            data,
            ((self.offset, self.offset), (self.offset, self.offset), (0, 0)),
            mode="reflect",
        )

        h, w, _ = data.shape
        self.coords = [(y, x) for y in range(h) for x in range(w)]

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[int, int]]:
        y, x = self.coords[idx]
        py, px = y + self.offset, x + self.offset

        patch = self.padded_data[
            py - self.offset : py + self.offset + 1,
            px - self.offset : px + self.offset + 1,
            :,
        ]
        tensor_patch = torch.tensor(patch.transpose(2, 0, 1), dtype=torch.float32)
        return tensor_patch, (y, x)