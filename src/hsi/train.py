from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .data_loader import load_pavia_data, normalize_data, split_spatial_gt
from .dataset import PaviaPatchDataset
from .model import SpatialSpectralCNN, get_device


def train(
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    weights_path: Path = Path("outputs/best_model.pth"),
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()

    data, gt = load_pavia_data()
    gt_train, gt_val, split_col = split_spatial_gt(gt, train_frac=0.75)
    data = normalize_data(data, split_col=split_col)

    train_dataset = PaviaPatchDataset(data, gt_train, patch_size=9)
    val_dataset = PaviaPatchDataset(data, gt_val, patch_size=9)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SpatialSpectralCNN(in_channels=data.shape[-1], num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
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
            torch.save(model.state_dict(), weights_path)

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Weights saved to: {weights_path}")


if __name__ == "__main__":
    train()
