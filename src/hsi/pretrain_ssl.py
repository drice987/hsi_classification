from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data_loader import load_pavia_data, normalize_data
from .dataset import PaviaSSLDataset
from .model import MaskedConvolutionalAutoencoder, get_device


def apply_spatial_mask(
    batch: torch.Tensor, mask_ratio: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeros out a random fraction of spatial pixels and returns the boolean drop mask."""
    b, c, h, w = batch.shape
    drop_mask = (torch.rand(b, 1, h, w, device=batch.device) < mask_ratio).expand(
        b, c, h, w
    )
    masked_batch = batch.masked_fill(drop_mask, 0.0)
    return masked_batch, drop_mask


def masked_mse_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    drop_mask: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Evaluates mean squared error strictly over masked spatial pixels."""
    if drop_mask.any():
        return criterion(reconstructed[drop_mask], target[drop_mask])
    return criterion(reconstructed, target)


def pretrain(
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    mask_ratio: float = 0.25,
    weights_path: Path = Path("outputs/ssl_encoder_pretrained.pth"),
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()

    data, _ = load_pavia_data()
    split_col = int(data.shape[1] * 0.75)
    data = normalize_data(data, split_col=split_col)

    train_dataset = PaviaSSLDataset(data[:, :split_col, :], patch_size=9)
    val_dataset = PaviaSSLDataset(data[:, split_col:, :], patch_size=9)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MaskedConvolutionalAutoencoder(in_channels=data.shape[-1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            masked_images, drop_mask = apply_spatial_mask(images, mask_ratio=mask_ratio)

            optimizer.zero_grad()
            reconstructed = model(masked_images)
            loss = masked_mse_loss(reconstructed, images, drop_mask, criterion)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                masked_images, drop_mask = apply_spatial_mask(
                    images, mask_ratio=mask_ratio
                )
                reconstructed = model(masked_images)
                val_loss += masked_mse_loss(
                    reconstructed, images, drop_mask, criterion
                ).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train MSE: {avg_train_loss:.4f} | "
            f"Val MSE: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.encoder.state_dict(), weights_path)

    print(f"Pretraining complete. Best Val MSE: {best_val_loss:.4f} ({weights_path})")


if __name__ == "__main__":
    pretrain()