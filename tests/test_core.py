import numpy as np
import pytest
import torch

from hsi.baseline import build_rf_pipeline, prepare_tabular_data
from hsi.data_loader import normalize_data, split_spatial_gt
from hsi.dataset import PaviaPatchDataset
from hsi.model import SpatialSpectralCNN


@pytest.fixture
def synthetic_pavia() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    cube = rng.uniform(100.0, 2500.0, size=(20, 24, 103)).astype(np.float32)
    gt = np.zeros((20, 24), dtype=np.int64)
    gt[2:18, 2:10] = 1
    gt[2:18, 12:22] = 2
    gt[5:10, 5:8] = 6
    gt[12:16, 14:18] = 1
    gt[4:8, 14:18] = 6
    return cube, gt


def test_split_spatial_gt_disjoint(
    synthetic_pavia: tuple[np.ndarray, np.ndarray],
) -> None:
    _, gt = synthetic_pavia
    gt_train, gt_val, split_col = split_spatial_gt(gt, train_frac=0.75)

    assert split_col == 18
    assert np.all(gt_train[:, split_col:] == 0)
    assert np.all(gt_val[:, :split_col] == 0)
    assert not np.any((gt_train > 0) & (gt_val > 0))


def test_normalize_data_uses_training_statistics(
    synthetic_pavia: tuple[np.ndarray, np.ndarray],
) -> None:
    cube, _ = synthetic_pavia
    split_col = 18
    cube_skewed = cube.copy()
    cube_skewed[:, split_col:, :] += 50000.0

    norm_cube = normalize_data(cube_skewed, split_col=split_col)
    train_region = norm_cube[:, :split_col, :].reshape(-1, 103)

    np.testing.assert_allclose(train_region.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(train_region.std(axis=0), 1.0, atol=1e-4)

    mask = np.zeros((20, 24), dtype=bool)
    mask[2:16, 2:16] = True
    cube_masked = cube.copy()
    cube_masked[~mask] = 0.0
    norm_masked = normalize_data(cube_masked, split_col=split_col, valid_mask=mask)
    assert np.all(norm_masked[~mask] == 0.0)


def test_prepare_tabular_data_and_rf(
    synthetic_pavia: tuple[np.ndarray, np.ndarray],
) -> None:
    cube, gt = synthetic_pavia
    x_clean, y_clean = prepare_tabular_data(cube, gt)

    expected_count = int(np.sum(gt > 0))
    assert x_clean.shape == (expected_count, 103)
    assert y_clean.shape == (expected_count,)
    assert y_clean.min() == 0

    rf = build_rf_pipeline(n_components=5)
    rf.fit(x_clean, y_clean)
    preds = rf.predict(x_clean)
    assert preds.shape == (expected_count,)


def test_pavia_patch_dataset(
    synthetic_pavia: tuple[np.ndarray, np.ndarray],
) -> None:
    cube, gt = synthetic_pavia
    ds = PaviaPatchDataset(cube, gt, patch_size=9)

    assert len(ds) == int(np.sum(gt > 0))
    patch, label = ds[0]
    assert patch.shape == (103, 9, 9)
    assert patch.dtype == torch.float32
    assert label.dtype == torch.long
    assert 0 <= label.item() <= 8


def test_spatial_spectral_cnn_forward() -> None:
    model = SpatialSpectralCNN(in_channels=103, num_classes=9)
    batch = torch.randn(4, 103, 9, 9)
    out = model(batch)

    assert out.shape == (4, 9)
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
