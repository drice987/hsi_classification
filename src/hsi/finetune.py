import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .data_loader import load_scene, normalize_data
from .dataset import PaviaPatchDataset
from .model import (
    MaskedConvolutionalAutoencoder,
    SSLClassifier,
    get_device,
    load_pretrained_mae,
)
from .train import evaluate_classifier, train_one_epoch


def apply_stratified_few_shot_split(
    labels: np.ndarray,
    cube: np.ndarray,
    nir_band_idx: int = 60,
    shots_per_class: int = 200,
    max_val_per_class: int = 2500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gt_train = np.zeros_like(labels)
    gt_val = np.zeros_like(labels)
    band_idx = min(max(0, nir_band_idx), cube.shape[-1] - 1)

    unique_classes = np.unique(labels)
    unique_classes = unique_classes[unique_classes > 0]

    for c in unique_classes:
        coords = np.argwhere(labels == c)
        if len(coords) == 0:
            continue

        if len(coords) <= shots_per_class:
            gt_train[coords[:, 0], coords[:, 1]] = c
            continue

        nir_vals = cube[coords[:, 0], coords[:, 1], band_idx]
        median_val = float(np.median(nir_vals))

        low_indices = np.argwhere(nir_vals <= median_val).flatten()
        high_indices = np.argwhere(nir_vals > median_val).flatten()

        n_low = shots_per_class // 2
        n_high = shots_per_class - n_low

        chosen_low = rng.choice(
            len(low_indices),
            size=min(n_low, len(low_indices)),
            replace=False,
        )
        chosen_high = rng.choice(
            len(high_indices),
            size=min(n_high, len(high_indices)),
            replace=False,
        )

        selected_train_idx = np.concatenate(
            [low_indices[chosen_low], high_indices[chosen_high]]
        )
        selected_coords = coords[selected_train_idx]
        gt_train[selected_coords[:, 0], selected_coords[:, 1]] = c

        all_indices = np.arange(len(coords))
        remaining_indices = np.setdiff1d(all_indices, selected_train_idx)

        val_size = min(max_val_per_class, len(remaining_indices))
        val_sub = rng.choice(remaining_indices, size=val_size, replace=False)
        val_coords = coords[val_sub]
        gt_val[val_coords[:, 0], val_coords[:, 1]] = c

    return gt_train, gt_val


def finetune(
    source_path: Path | str = "data/enmap_sample",
    gt_path: Path | str | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    shots_per_class: int = 200,
    patch_size: int = 9,
    unfreeze_epoch: int = 15,
    pretrained_weights: Path = Path("outputs/enmap_ssl_full.pth"),
    final_weights: Path = Path("outputs/finetuned_stratified.pth"),
) -> float:
    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()

    scene = load_scene(source_path, gt_path=gt_path, bbox=bbox)
    data = normalize_data(scene.cube, valid_mask=scene.valid_mask)
    nir_idx = int(np.argmin(np.abs(scene.wavelengths - 865.0)))

    gt_train, gt_val = apply_stratified_few_shot_split(
        scene.labels,
        scene.cube,
        nir_band_idx=nir_idx,
        shots_per_class=shots_per_class,
        seed=42,
    )

    train_dataset = PaviaPatchDataset(data, gt_train, patch_size=patch_size)
    val_dataset = PaviaPatchDataset(data, gt_val, patch_size=patch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    mae = MaskedConvolutionalAutoencoder(in_channels=data.shape[-1])
    load_pretrained_mae(mae, pretrained_weights, device)

    model = SSLClassifier(
        pretrained_encoder=mae.encoder,
        num_classes=len(scene.class_names),
        use_center_pixel=True,
    ).to(device)

    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)

    final_weights = Path(final_weights)
    final_weights.parent.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    best_val_acc = 0.0

    print(
        f"Starting Fine-Tuning: {shots_per_class} shots/class | "
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}"
    )

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch:
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                [
                    {"params": model.encoder.parameters(), "lr": 1e-4},
                    {"params": model.classifier.parameters(), "lr": 5e-4},
                ],
                weight_decay=1e-4,
            )

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
        f"Fine-tuning complete. Best Macro F1: {best_val_f1:.2f}% "
        f"(OA: {best_val_acc:.2f}%) saved to: {final_weights}"
    )
    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified Few-Shot Fine-Tuning")
    parser.add_argument("source", nargs="?", default="data/enmap_sample")
    parser.add_argument(
        "--gt-tif",
        type=Path,
        default=None,
        help="Path to external aligned ground-truth raster",
    )
    parser.add_argument("--bbox", nargs=4, type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("outputs/enmap_ssl_full.pth"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("outputs/finetuned_stratified.pth"),
    )
    args = parser.parse_args()

    bbox_tuple = tuple(args.bbox) if args.bbox else None
    finetune(
        source_path=args.source,
        gt_path=args.gt_tif,
        bbox=bbox_tuple,
        epochs=args.epochs,
        shots_per_class=args.shots,
        pretrained_weights=args.pretrained,
        final_weights=args.weights,
    )
