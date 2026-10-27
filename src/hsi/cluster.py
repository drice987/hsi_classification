import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader

from .data_loader import load_scene, normalize_data, split_spatial_gt
from .dataset import PaviaInferenceDataset
from .model import MaskedConvolutionalAutoencoder, get_device, load_pretrained_mae


def compute_latent_and_anomaly_maps(
    data: np.ndarray,
    valid_mask: np.ndarray,
    mae: MaskedConvolutionalAutoencoder,
    device: torch.device,
    n_clusters: int = 12,
    patch_size: int = 9,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts center-pixel latent K-Means clusters and center-masked reconstruction RMSE."""
    dataset = PaviaInferenceDataset(data, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mae = mae.to(device)
    mae.eval()

    h, w, _ = data.shape
    c_idx = patch_size // 2
    all_features = []
    all_rmse = []
    y_list, x_list = [], []

    with torch.no_grad():
        for images, coords in loader:
            images = images.to(device)
            latent = mae.encoder(images)
            center_features = latent[:, :, c_idx, c_idx]

            masked_input = images.clone()
            r0 = max(0, c_idx - 1)
            r1 = min(patch_size, c_idx + 2)
            masked_input[:, :, r0:r1, r0:r1] = 0.0

            recon = mae(masked_input)
            center_diff = recon[:, :, c_idx, c_idx] - images[:, :, c_idx, c_idx]
            center_rmse = torch.sqrt(torch.mean(center_diff**2, dim=1))

            all_features.append(center_features.cpu().numpy())
            all_rmse.append(center_rmse.cpu().numpy())
            y_list.append(coords[0].numpy())
            x_list.append(coords[1].numpy())

    feature_matrix = np.vstack(all_features)
    rmse_vec = np.concatenate(all_rmse)
    y_all = np.concatenate(y_list)
    x_all = np.concatenate(x_list)

    flat_valid = valid_mask[y_all, x_all]
    cluster_map = np.zeros((h, w), dtype=np.uint8)

    if np.any(flat_valid):
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=4096, n_init="auto"
        )
        valid_labels = kmeans.fit_predict(feature_matrix[flat_valid]) + 1
        cluster_map[y_all[flat_valid], x_all[flat_valid]] = valid_labels

    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[y_all[flat_valid], x_all[flat_valid]] = rmse_vec[flat_valid]

    return cluster_map, anomaly_map


def run_clustering(
    source_path: Path | str = "data/PaviaU.mat",
    bbox: tuple[int, int, int, int] | None = None,
    model_path: Path | str = "outputs/ssl_encoder_pretrained.pth",
    n_clusters: int = 12,
    patch_size: int = 9,
    batch_size: int = 512,
    out_dir: Path | str = "outputs",
) -> tuple[np.ndarray, np.ndarray]:
    device = get_device()

    data, gt, _, _ = load_scene(source_path, bbox=bbox)
    valid_mask = np.any(data > 1e-4, axis=-1)
    _, _, split_col = split_spatial_gt(gt, train_frac=0.75)
    norm_data = normalize_data(data, split_col=split_col, valid_mask=valid_mask)

    mae = MaskedConvolutionalAutoencoder(in_channels=data.shape[-1])
    weights_file = Path(model_path)
    if weights_file.exists():
        load_pretrained_mae(mae, weights_file, device)

    cluster_map, anomaly_map = compute_latent_and_anomaly_maps(
        norm_data,
        valid_mask,
        mae,
        device,
        n_clusters=n_clusters,
        patch_size=patch_size,
        batch_size=batch_size,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    cmap = plt.get_cmap("tab20", n_clusters + 1)
    im0 = axes[0].imshow(cluster_map, cmap=cmap, vmin=0, vmax=n_clusters)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Cluster ID (0=Mask)")
    axes[0].set_title(f"Latent Spatial-Spectral Clustering (k={n_clusters})")
    axes[0].axis("off")

    vmax_anom = (
        float(np.percentile(anomaly_map[valid_mask], 99)) if np.any(valid_mask) else 1.0
    )
    im1 = axes[1].imshow(
        anomaly_map, cmap="inferno", vmin=0.0, vmax=max(vmax_anom, 1e-4)
    )
    plt.colorbar(
        im1, ax=axes[1], fraction=0.046, pad=0.04, label="Center-Pixel Recon RMSE"
    )
    axes[1].set_title("Self-Supervised Spatial-Spectral Anomaly Map")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path / f"ssl_clusters_anomalies_{n_clusters}.png", dpi=300)
    plt.close(fig)

    return cluster_map, anomaly_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SSL Latent Clusters and Spatial-Spectral Anomaly Map"
    )
    parser.add_argument("source", nargs="?", default="data/PaviaU.mat")
    parser.add_argument("--bbox", nargs=4, type=int, default=None)
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/ssl_encoder_pretrained.pth")
    )
    parser.add_argument("--clusters", type=int, default=12)
    args = parser.parse_args()

    bbox_tuple = tuple(args.bbox) if args.bbox else None
    run_clustering(
        source_path=args.source,
        bbox=bbox_tuple,
        model_path=args.weights,
        n_clusters=args.clusters,
    )
