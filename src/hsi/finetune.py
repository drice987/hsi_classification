from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .data_loader import load_pavia_data, normalize_data, split_spatial_gt
from .dataset import PaviaPatchDataset
from .model import (
    MaskedConvolutionalAutoencoder,
    SSLClassifier,
    get_device,
    load_pretrained_mae,
)


def apply_few_shot_mask(
    gt_train: np.ndarray, shots_per_class: int = 15, seed: int = 42
) -> np.ndarray:
    """Subsamples training labels to a fixed number of pixels per class."""
    few_shot_gt = np.zeros_like(gt_train)
    rng = np.random.default_rng(seed)

    for class_idx in range(1, int(gt_train.max()) + 1):
        y_coords, x_coords = np.where(gt_train == class_idx)
        if len(y_coords) > 0:
            chosen = rng.permutation(len(y_coords))[:shots_per_class]
            few_shot_gt[y_coords[chosen], x_coords[chosen]] = class_idx

    return few_shot_gt


def finetune(
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    unfreeze_epoch: int = 41,
    pretrained_weights: Path = Path("outputs/ssl_encoder_pretrained.pth"),
    final_weights: Path = Path("outputs/finetuned_model.pth"),
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()

    data, gt = load_pavia_data()
    gt_train, gt_val, split_col = split_spatial_gt(gt, train_frac=0.75)
    data = normalize_data(data, split_col=split_col)

    gt_train = apply_few_shot_mask(gt_train, shots_per_class=15)

    train_dataset = PaviaPatchDataset(data, gt_train, patch_size=9)
    val_dataset = PaviaPatchDataset(data, gt_val, patch_size=9)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    mae = MaskedConvolutionalAutoencoder(in_channels=data.shape[-1])
    load_pretrained_mae(mae, pretrained_weights, device)
    model = SSLClassifier(pretrained_encoder=mae.encoder, num_classes=9).to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)

    final_weights.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch:
            for param in model.encoder.parameters():
                param.requires_grad = True

            optimizer = optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": 1e-5},
                    {"params": model.classifier.parameters(), "lr": 1e-4},
                ],
                weight_decay=1e-4,
            )

        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                val_loss += criterion(outputs, labels).item()
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100.0 * correct / max(total, 1)
        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), final_weights)

    print(f"Fine-tuning complete. Best Val Acc: {best_val_acc:.2f}% ({final_weights})")


if __name__ == "__main__":
    finetune()
