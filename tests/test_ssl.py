import numpy as np
import torch
import torch.nn as nn

from hsi.dataset import PaviaInferenceDataset, PaviaSSLDataset
from hsi.finetune import apply_few_shot_mask
from hsi.model import MaskedConvolutionalAutoencoder, SSLClassifier
from hsi.pretrain_ssl import apply_spatial_mask, masked_mse_loss


def test_ssl_and_inference_datasets() -> None:
    cube = np.ones((20, 24, 103), dtype=np.float32)

    ssl_ds = PaviaSSLDataset(cube, patch_size=9)
    assert len(ssl_ds) == (20 - 8) * (24 - 8)
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


def test_mae_and_masked_loss() -> None:
    batch = torch.ones(4, 103, 9, 9)
    masked_batch, drop_mask = apply_spatial_mask(batch, mask_ratio=0.25)

    assert drop_mask.shape == batch.shape
    assert torch.all(masked_batch[drop_mask] == 0.0)
    assert torch.all(masked_batch[~drop_mask] == 1.0)

    mae = MaskedConvolutionalAutoencoder(in_channels=103)
    recon = mae(masked_batch)
    assert recon.shape == (4, 103, 9, 9)

    loss = masked_mse_loss(recon, batch, drop_mask, nn.MSELoss())
    assert loss.item() > 0.0

    clf = SSLClassifier(pretrained_encoder=mae.encoder, num_classes=9)
    logits = clf(batch)
    assert logits.shape == (4, 9)
