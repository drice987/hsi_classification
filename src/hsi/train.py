import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader

from .data_loader import load_scene, normalize_data, split_spatial_gt
from .dataset import PaviaPatchDataset
from .model import SpatialSpectralCNN, get_device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    if not all_targets:
        return 0.0, 0.0, 0.0

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    avg_loss = total_loss / max(len(loader), 1)
    oa = 100.0 * float(np.mean(y_pred == y_true))
    macro_f1 = 100.0 * float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return avg_loss, oa, macro_f1


def train(
    source_path: Path | str = "data/PaviaU.mat",
    gt_path: Path | str | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    patch_size: int = 9,
    final_weights: Path = Path("outputs/supervised_model.pth"),
) -> float:
    final_weights = Path(final_weights)
    final_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()

    scene = load_scene(source_path, gt_path=gt_path, bbox=bbox)
    gt_train, gt_val, split_col = split_spatial_gt(
        scene.labels, train_frac=0.75, buffer_cols=patch_size // 2
    )
    data = normalize_data(scene.cube, split_col=split_col, valid_mask=scene.valid_mask)

    train_dataset = PaviaPatchDataset(data, gt_train, patch_size=patch_size)
    val_dataset = PaviaPatchDataset(data, gt_val, patch_size=patch_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SpatialSpectralCNN(
        in_channels=data.shape[-1], num_classes=len(scene.class_names)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        avg_val_loss, val_acc, val_f1 = evaluate_classifier(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val OA: {val_acc:.2f}% | "
            f"Val Macro F1: {val_f1:.2f}%"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), final_weights)

    print(
        f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}% "
        f"(Macro F1: {best_val_f1:.2f}%) saved to: {final_weights}"
    )
    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised Spatial-Spectral CNN Training"
    )
    parser.add_argument("source", nargs="?", default="data/PaviaU.mat")
    parser.add_argument("--gt-tif", type=Path, default=None)
    parser.add_argument("--bbox", nargs=4, type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/supervised_model.pth")
    )
    args = parser.parse_args()

    bbox_tuple = tuple(args.bbox) if args.bbox else None
    train(
        source_path=args.source,
        gt_path=args.gt_tif,
        bbox=bbox_tuple,
        epochs=args.epochs,
        final_weights=args.weights,
    )
