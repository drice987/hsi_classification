import numpy as np
import torch
from torch import nn

from hsi.cluster import compute_latent_and_anomaly_maps
from hsi.dataset import PaviaInferenceDataset, PaviaSSLDataset
from hsi.finetune import apply_few_shot_mask
from hsi.model import MaskedConvolutionalAutoencoder, SSLClassifier
from hsi.pretrain_ssl import apply_spatial_mask, masked_mse_loss


def test_ssl_and_inference_datasets() -> None:
    cube = np.ones((20, 24, 103), dtype=np.float32)
    valid_mask = np.ones((20, 24), dtype=bool)
    valid_mask[:, :5] = False

    ssl_ds = PaviaSSLDataset(cube, patch_size=9, valid_mask=valid_mask)
    assert len(ssl_ds) < (20 - 8) * (24 - 8)
    patch, dummy_label = ssl_ds[0]
    assert patch.shape == (103, 9, 9)
    assert dummy_label == 0

    inf_ds = PaviaInferenceDataset(cube, patch_size=9)
    assert len(inf_ds) == 20 * 24
    inf_patch, coords = inf_ds[0]
    assert inf_patch.shape == (103, 9, 9)
    assert coords == (0, 0)


def test_apply_few_shot_mask() -> None:
    gt = np.zeros((20, 20), dtype=np.int64)
    gt[:10, :10] = 1
    gt[10:, 10:] = 2

    masked = apply_few_shot_mask(gt, shots_per_class=7, seed=42)
    assert int(np.sum(masked == 1)) == 7
    assert int(np.sum(masked == 2)) == 7


def test_mae_masked_loss_and_anomaly_clustering() -> None:
    batch = torch.ones(4, 30, 9, 9)
    masked_batch, drop_mask = apply_spatial_mask(batch, mask_ratio=0.25)

    assert drop_mask.shape == batch.shape
    assert torch.all(masked_batch[drop_mask] == 0.0)
    assert torch.all(masked_batch[~drop_mask] == 1.0)

    mae = MaskedConvolutionalAutoencoder(in_channels=30)
    recon = mae(masked_batch)
    assert recon.shape == (4, 30, 9, 9)

    loss = masked_mse_loss(recon, batch, drop_mask, nn.MSELoss())
    assert loss.item() > 0.0

    clf = SSLClassifier(pretrained_encoder=mae.encoder, num_classes=5)
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
