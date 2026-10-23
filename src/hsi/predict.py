from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_loader import load_pavia_data, normalize_data
from .dataset import PaviaInferenceDataset
from .model import MaskedConvolutionalAutoencoder, SSLClassifier, get_device


def generate_prediction_map(
    model_path: Path | str = "outputs/finetuned_model.pth",
    patch_size: int = 9,
    batch_size: int = 512,
    out_dir: Path | str = "outputs",
) -> None:
    device = get_device()

    data, _ = load_pavia_data()
    norm_data = normalize_data(data)

    dataset = PaviaInferenceDataset(norm_data, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mae = MaskedConvolutionalAutoencoder(in_channels=data.shape[-1])
    model = SSLClassifier(pretrained_encoder=mae.encoder, num_classes=9).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    h, w, _ = data.shape
    pred_map = np.zeros((h, w), dtype=np.uint8)

    with torch.no_grad():
        for images, coords in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_coords, x_coords = coords[0].numpy(), coords[1].numpy()
            pred_map[y_coords, x_coords] = preds + 1

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("nipy_spectral", 10)

    plt.figure(figsize=(8, 12))
    plt.imshow(pred_map, cmap=cmap, vmin=0, vmax=9)
    plt.title("Masked Autoencoder (SSL) Full-Scene Classification")
    plt.colorbar(ticks=range(10), label="Class ID")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(out_path / "ssl_full_prediction_map.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    generate_prediction_map()