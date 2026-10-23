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