from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from .data_loader import load_pavia_data, normalize_data
from .dataset import PaviaInferenceDataset
from .model import MaskedConvolutionalAutoencoder, SSLClassifier, get_device


def run_clustering(
    model_path: Path | str = "outputs/finetuned_model.pth",
    n_clusters: int = 15,
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

    encoder = model.encoder
    pool = model.pool
    encoder.eval()

    h, w, _ = data.shape
    all_features = []
    y_list, x_list = [], []

    with torch.no_grad():
        for images, coords in loader:
            images = images.to(device)
            features = torch.flatten(pool(encoder(images)), 1)
            all_features.append(features.cpu().numpy())
            y_list.append(coords[0].numpy())
            x_list.append(coords[1].numpy())

    feature_matrix = np.vstack(all_features)
    y_all = np.concatenate(y_list)
    x_all = np.concatenate(x_list)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, batch_size=4096, n_init="auto"
    )
    cluster_labels = kmeans.fit_predict(feature_matrix)

    cluster_map = np.zeros((h, w), dtype=np.uint8)
    cluster_map[y_all, x_all] = cluster_labels

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab20", n_clusters)

    plt.figure(figsize=(8, 12))
    plt.imshow(cluster_map, cmap=cmap)
    plt.title(f"Latent Feature K-Means Clustering (k={n_clusters})")
    plt.colorbar(ticks=range(n_clusters), label="Cluster ID")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(out_path / f"unsupervised_clusters_{n_clusters}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_clustering()