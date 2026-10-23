from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_loader import load_pavia_data, normalize_data
from .dataset import PaviaPatchDataset
from .model import SpatialSpectralCNN, get_device


def generate_prediction_map(
    model_path: Path | str = "outputs/best_model.pth",
    patch_size: int = 9,
    batch_size: int = 256,
    out_dir: Path | str = "outputs",
) -> None:
    device = get_device()

    data, gt = load_pavia_data()
    norm_data = normalize_data(data)
    dataset = PaviaPatchDataset(norm_data, gt, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SpatialSpectralCNN(in_channels=data.shape[-1], num_classes=9).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            all_preds.append(predicted.cpu().numpy())

    preds_arr = np.concatenate(all_preds)
    coords = np.array(dataset.coordinates)

    pred_map = np.zeros(gt.shape, dtype=np.uint8)
    pred_map[coords[:, 0], coords[:, 1]] = preds_arr + 1

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("nipy_spectral", 10)

    plt.figure(figsize=(6, 10))
    plt.imshow(pred_map, cmap=cmap, vmin=0, vmax=9)
    plt.title("CNN Spatial-Spectral Prediction Map")
    plt.colorbar(ticks=range(10), label="Class ID")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(out_path / "cnn_prediction_map.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    generate_prediction_map()