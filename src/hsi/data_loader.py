from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from .enmap import ENMAP_SURFACE_CLASSES, GeoMetadata, load_enmap_l2a

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


@dataclass
class SceneData:
    cube: np.ndarray
    labels: np.ndarray
    wavelengths: np.ndarray
    class_names: list[str]
    valid_mask: np.ndarray
    geo: GeoMetadata | None = None

    def __iter__(self) -> Iterator[object]:
        yield self.cube
        yield self.labels
        yield self.wavelengths
        yield self.class_names


def load_pavia_data(
    data_path: Path | str = "data/PaviaU.mat",
    gt_path: Path | str = "data/PaviaU_gt.mat",
) -> tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(data_path)["paviaU"]
    gt = sio.loadmat(gt_path)["paviaU_gt"]
    return data.astype(np.float32), gt.astype(np.int64)


def load_scene(
    source_path: Path | str = "data/PaviaU.mat",
    gt_path: Path | str | None = None,
    bbox: tuple[int, int, int, int] | None = None,
) -> SceneData:
    path = Path(source_path)
    if path.is_dir():
        scene = load_enmap_l2a(path, gt_tif_path=gt_path, bbox=bbox)
        return SceneData(
            cube=scene.cube,
            labels=scene.labels,
            wavelengths=scene.wavelengths,
            class_names=ENMAP_SURFACE_CLASSES,
            valid_mask=scene.valid_mask,
            geo=scene.geo,
        )

    default_gt = Path("data/PaviaU_gt.mat") if gt_path is None else Path(gt_path)
    cube, gt = load_pavia_data(path, default_gt)
    if bbox is not None:
        r0, r1 = max(0, bbox[0]), min(cube.shape[0], bbox[1])
        c0, c1 = max(0, bbox[2]), min(cube.shape[1], bbox[3])
        cube = cube[r0:r1, c0:c1, :]
        gt = gt[r0:r1, c0:c1]

    wavelengths = np.linspace(430.0, 860.0, cube.shape[-1], dtype=np.float32)
    valid_mask = np.any(cube > 1e-4, axis=-1)
    return SceneData(
        cube=cube,
        labels=gt,
        wavelengths=wavelengths,
        class_names=CLASS_NAMES,
        valid_mask=valid_mask,
        geo=None,
    )


def split_spatial_gt(
    gt: np.ndarray,
    train_frac: float = 0.75,
    buffer_cols: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Splits ground-truth labels along a vertical column boundary."""
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

    margin = max(0, buffer_cols)
    train_cutoff = max(0, split_col - margin)
    val_start = min(gt.shape[1], split_col + margin)

    gt_train = gt.copy()
    gt_val = gt.copy()

    if (
        margin > 0
        and np.any(gt_train[:, :train_cutoff] > 0)
        and np.any(gt_val[:, val_start:] > 0)
    ):
        gt_train[:, train_cutoff:] = 0
        gt_val[:, :val_start] = 0
    else:
        gt_train[:, split_col:] = 0
        gt_val[:, :split_col] = 0

    return gt_train, gt_val, split_col


def normalize_data(
    data: np.ndarray,
    split_col: int | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if valid_mask is None:
        valid_mask = np.any(data != 0.0, axis=-1)

    if split_col is not None:
        train_region = data[:, :split_col, :]
        train_valid = valid_mask[:, :split_col]
    else:
        train_region = data
        train_valid = valid_mask

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
