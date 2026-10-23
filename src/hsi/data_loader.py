from pathlib import Path

import numpy as np
import scipy.io as sio

CLASS_NAMES: list[str] = [
    "Asphalt",
    "Meadows",
    "Gravel",
    "Trees",
    "Metal Sheets",
    "Bare Soil",
    "Bitumen",
    "Bricks",
    "Shadows",
]


def load_pavia_data(
    data_path: Path | str = "data/PaviaU.mat",
    gt_path: Path | str = "data/PaviaU_gt.mat",
) -> tuple[np.ndarray, np.ndarray]:
    """Loads raw Pavia University data cube and ground-truth labels."""
    data = sio.loadmat(data_path)["paviaU"]
    gt = sio.loadmat(gt_path)["paviaU_gt"]
    return data.astype(np.float32), gt.astype(np.int64)


def split_spatial_gt(
    gt: np.ndarray, train_frac: float = 0.75
) -> tuple[np.ndarray, np.ndarray, int]:
    """Splits ground-truth mask along a vertical column boundary."""
    split_col = int(gt.shape[1] * train_frac)

    gt_train = gt.copy()
    gt_train[:, split_col:] = 0

    gt_val = gt.copy()
    gt_val[:, :split_col] = 0

    return gt_train, gt_val, split_col


def normalize_data(data: np.ndarray, split_col: int | None = None) -> np.ndarray:
    """Standardizes spectral bands using statistics from the training region."""
    if split_col is None:
        split_col = int(data.shape[1] * 0.75)

    train_pixels = data[:, :split_col, :].reshape(-1, data.shape[-1])
    means = train_pixels.mean(axis=0)
    stds = train_pixels.std(axis=0)

    flattened = data.reshape(-1, data.shape[-1])
    normalized = (flattened - means) / (stds + 1e-7)
    return normalized.reshape(data.shape)