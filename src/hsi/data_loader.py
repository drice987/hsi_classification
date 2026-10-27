from pathlib import Path

import numpy as np
import scipy.io as sio

from .enmap import ENMAP_SURFACE_CLASSES, load_enmap_l2a

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


def load_scene(
    source_path: Path | str = "data/PaviaU.mat",
    gt_path: Path | str | None = None,
    bbox: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Loads either an EnMAP L2A directory or the PaviaU dataset."""
    path = Path(source_path)
    if path.is_dir():
        scene = load_enmap_l2a(path, gt_tif_path=gt_path, bbox=bbox)
        return (
            scene.cube,
            scene.labels,
            scene.wavelengths,
            ENMAP_SURFACE_CLASSES,
        )

    default_gt = Path("data/PaviaU_gt.mat") if gt_path is None else Path(gt_path)
    cube, gt = load_pavia_data(path, default_gt)
    wavelengths = np.linspace(430.0, 860.0, cube.shape[-1], dtype=np.float32)
    return cube, gt, wavelengths, CLASS_NAMES


def split_spatial_gt(
    gt: np.ndarray, train_frac: float = 0.75
) -> tuple[np.ndarray, np.ndarray, int]:
    """Splits ground-truth mask along a vertical column boundary."""
    split_col = int(gt.shape[1] * train_frac)

    valid_cols = np.where(gt > 0)[1]
    if valid_cols.size > 1:
        has_left = np.any(valid_cols < split_col)
        has_right = np.any(valid_cols >= split_col)
        if not (has_left and has_right):
            split_col = int(np.quantile(valid_cols, train_frac))
            split_col = max(
                int(valid_cols.min()) + 1, min(int(valid_cols.max()), split_col)
            )

    gt_train = gt.copy()
    gt_train[:, split_col:] = 0

    gt_val = gt.copy()
    gt_val[:, :split_col] = 0

    return gt_train, gt_val, split_col


def normalize_data(
    data: np.ndarray,
    split_col: int | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Standardizes spectral bands using statistics from valid training pixels."""
    if split_col is None:
        split_col = int(data.shape[1] * 0.75)

    if valid_mask is None:
        valid_mask = np.any(data != 0.0, axis=-1)

    train_region = data[:, :split_col, :]
    train_valid = valid_mask[:, :split_col]

    if np.any(train_valid):
        train_pixels = train_region[train_valid]
    elif np.any(valid_mask):
        train_pixels = data[valid_mask]
    else:
        train_pixels = train_region.reshape(-1, data.shape[-1])

    means = train_pixels.mean(axis=0)
    stds = train_pixels.std(axis=0)

    normalized = (data - means[np.newaxis, np.newaxis, :]) / (
        stds[np.newaxis, np.newaxis, :] + 1e-7
    )
    normalized[~valid_mask] = 0.0
    return normalized.astype(np.float32)
