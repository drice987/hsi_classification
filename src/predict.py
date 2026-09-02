import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from data_loader import load_pavia_data
from dataset import PaviaPatchDataset
from model import SpatialSpectralCNN
from train import normalize_data

def generate_prediction_map(model_path: str = "outputs/best_model.pth",patch_size: int = 9,batch_size: int = 256,out_dir: str = "outputs") -> None:
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    data, gt = load_pavia_data()
    norm_data = normalize_data(data)
    dataset = PaviaPatchDataset(norm_data, gt, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = SpatialSpectralCNN(in_channels=103, num_classes=9).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    h, w = gt.shape
    pred_map = np.zeros((h, w), dtype=np.uint8)

    for idx, (y, x) in enumerate(dataset.coordinates):
        pred_map[y, x] = all_preds[idx] + 1  

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap('nipy_spectral', 10)

    plt.figure(figsize=(6, 10))
    plt.imshow(pred_map, cmap=cmap, vmin=0, vmax=9)
    plt.title("CNN Spatial-Spectral Prediction Map")
    plt.colorbar(ticks=range(10), label="Class ID")
    plt.axis('off')
    plt.tight_layout()

    out_path = Path(out_dir) / "cnn_prediction_map.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_prediction_map()