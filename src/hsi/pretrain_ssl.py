import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .data_loader import load_scene, normalize_data
from .dataset import RandomSpatialPatchDataset
from .model import MaskedConvolutionalAutoencoder, get_device


def pretrain_ssl(
    source_path: Path | str = "data/enmap_sample",
    bbox: tuple[int, int, int, int] | None = None,
    epochs: int = 40,
    batch_size: int = 128,
    samples_per_epoch: int = 40_000,
    lr: float = 5e-4,
    mask_ratio: float = 0.5,
    patch_size: int = 9,
    weights_path: Path = Path("outputs/ssl_encoder_pretrained.pth"),
) -> Path:
    device = get_device()
    print(f"Using compute device: {device}")

    scene = load_scene(source_path, bbox=bbox)
    h, w, c = scene.cube.shape
    valid_pixels = int(np.count_nonzero(scene.valid_mask))
    print(
        f"Loaded scene dimensions: {w}x{h} ({c} bands) | Valid pixels: {valid_pixels:,}"
    )

    norm_data = normalize_data(scene.cube, valid_mask=scene.valid_mask)

    dataset = RandomSpatialPatchDataset(
        norm_data,
        scene.valid_mask,
        patch_size=patch_size,
        samples_per_epoch=samples_per_epoch,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = MaskedConvolutionalAutoencoder(in_channels=c, mask_ratio=mask_ratio).to(
        device
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print(
        f"Starting SSL pretraining: {epochs} epochs | {samples_per_epoch:,} patches/epoch | Batch size: {batch_size}"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        epoch_loss = total_loss / len(dataset)
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:02d}/{epochs} | Reconstruction MSE: {epoch_loss:.5f} | Elapsed: {elapsed:.1f}s"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), weights_path)

    print(f"Pretraining complete. Best model saved to: {weights_path}")
    return weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-Supervised Pretraining via Random Spatial Patch Sampling"
    )
    parser.add_argument("source", nargs="?", default="data/enmap_sample")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        default=None,
        help="Optional spatial crop (omit to train on full tile)",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-epoch", type=int, default=40_000)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/ssl_encoder_pretrained.pth")
    )
    args = parser.parse_args()

    bbox_tuple = tuple(args.bbox) if args.bbox else None
    pretrain_ssl(
        source_path=args.source,
        bbox=bbox_tuple,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        weights_path=args.out,
    )
