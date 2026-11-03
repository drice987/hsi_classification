import numpy as np
import torch
from torch import nn

from hsi.cluster import compute_latent_and_anomaly_maps
from hsi.dataset import (
    PaviaInferenceDataset,
    PaviaSSLDataset,
    RandomSpatialPatchDataset,
)
from hsi.finetune import apply_stratified_few_shot_split
from hsi.model import MaskedConvolutionalAutoencoder, SSLClassifier


def test_ssl_and_inference_datasets() -> None:
    cube = np.ones((20, 24, 103), dtype=np.float32)
    valid_mask = np.ones((20, 24), dtype=bool)
    valid_mask[:, :5] = False

    ssl_ds = PaviaSSLDataset(cube, patch_size=9, valid_mask=valid_mask)
    assert len(ssl_ds) < (20 - 8) * (24 - 8)
    patch, dummy_label = ssl_ds[0]
    assert patch.shape == (103, 9, 9)
    assert dummy_label == 0

    rand_ds = RandomSpatialPatchDataset(
        cube, valid_mask=valid_mask, patch_size=9, samples_per_epoch=50, seed=42
    )
    assert len(rand_ds) == 50
    rand_patch = rand_ds[0]
    assert rand_patch.shape == (103, 9, 9)
    assert rand_patch.dtype == torch.float32

    inf_ds = PaviaInferenceDataset(cube, patch_size=9)
    assert len(inf_ds) == 20 * 24
    inf_patch, coords = inf_ds[0]
    assert inf_patch.shape == (103, 9, 9)
    assert coords == (0, 0)


def test_apply_stratified_few_shot_split() -> None:
    gt = np.zeros((20, 20), dtype=np.int64)
    gt[:10, :10] = 1
    gt[10:, 10:] = 2

    cube = np.zeros((20, 20, 10), dtype=np.float32)
    cube[:5, :10, 5] = 0.10
    cube[5:10, :10, 5] = 0.45
    cube[10:15, 10:, 5] = 0.15
    cube[15:, 10:, 5] = 0.50

    gt_train, gt_val = apply_stratified_few_shot_split(
        gt, cube, nir_band_idx=5, shots_per_class=8, max_val_per_class=30, seed=42
    )

    assert int(np.sum(gt_train == 1)) == 8
    assert int(np.sum(gt_train == 2)) == 8
    assert int(np.sum(gt_val == 1)) == 30
    assert int(np.sum(gt_val == 2)) == 30
    assert not np.any((gt_train > 0) & (gt_val > 0))

    c1_train_coords = np.argwhere(gt_train == 1)
    c1_albedos = cube[c1_train_coords[:, 0], c1_train_coords[:, 1], 5]
    assert int(np.sum(c1_albedos < 0.25)) == 4
    assert int(np.sum(c1_albedos > 0.25)) == 4


def test_mae_forward_and_anomaly_clustering() -> None:
    torch.manual_seed(42)
    batch = torch.ones(4, 30, 9, 9)

    mae = MaskedConvolutionalAutoencoder(in_channels=30, mask_ratio=0.5)
    mae.train()
    recon_train = mae(batch)
    assert recon_train.shape == (4, 30, 9, 9)

    loss = nn.MSELoss()(recon_train, batch)
    assert loss.item() > 0.0

    mae.eval()
    recon_eval_1 = mae(batch)
    recon_eval_2 = mae(batch)
    assert torch.allclose(recon_eval_1, recon_eval_2)

    clf = SSLClassifier(
        pretrained_encoder=mae.encoder, num_classes=5, use_center_pixel=True
    )
    clf.eval()
    logits = clf(batch)
    assert logits.shape == (4, 5)

    rng = np.random.default_rng(42)
    cube = rng.normal(0.0, 1.0, size=(12, 14, 30)).astype(np.float32)
    valid_mask = np.ones((12, 14), dtype=bool)
    valid_mask[0, :] = False

    cluster_map, anomaly_map = compute_latent_and_anomaly_maps(
        cube,
        valid_mask,
        mae,
        torch.device("cpu"),
        n_clusters=3,
        patch_size=9,
        batch_size=32,
    )
    assert cluster_map.shape == (12, 14)
    assert anomaly_map.shape == (12, 14)
    assert np.all(cluster_map[0, :] == 0)
    assert np.all(cluster_map[1:, :] > 0)
    assert np.all(anomaly_map[0, :] == 0.0)
    assert np.all(anomaly_map[1:, :] > 0.0)
