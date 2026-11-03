import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from .data_loader import load_scene, normalize_data
from .model import (
    MaskedConvolutionalAutoencoder,
    get_device,
    load_pretrained_mae,
)


def compute_latent_and_anomaly_maps(
    data: np.ndarray,
    valid_mask: np.ndarray,
    mae: MaskedConvolutionalAutoencoder,
    device: torch.device,
    patch_size: int = 9,
    n_clusters: int = 10,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    mae.eval()
    h, w, _ = data.shape
    pad = patch_size // 2
    padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    coords = np.argwhere(valid_mask)

    latents: list[np.ndarray] = []
    anomaly_map = np.zeros((h, w), dtype=np.float32)

    center_mask = torch.zeros(
        (1, 1, patch_size, patch_size), dtype=torch.bool, device=device
    )
    r0, r1 = max(0, pad - 1), min(patch_size, pad + 2)
    center_mask[:, :, r0:r1, r0:r1] = True

    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i : i + batch_size]
            patches = np.stack(
                [
                    padded[r : r + patch_size, c : c + patch_size]
                    for r, c in batch_coords
                ]
            )

            tensor_patches = (
                torch.from_numpy(patches).permute(0, 3, 1, 2).float().to(device)
            )
            center_true = tensor_patches[:, :, pad, pad]

            feat = mae.encoder(tensor_patches)
            c_h = feat.shape[-2] // 2
            c_w = feat.shape[-1] // 2
            latents.append(feat[:, :, c_h, c_w].cpu().numpy())

            recon = mae(tensor_patches, spatial_mask=center_mask)
            center_recon = recon[:, :, pad, pad]
            batch_rmse = (
                torch.sqrt(torch.mean((center_true - center_recon) ** 2, dim=1))
                .cpu()
                .numpy()
            )
            anomaly_map[batch_coords[:, 0], batch_coords[:, 1]] = batch_rmse

    all_latents = np.concatenate(latents, axis=0)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, batch_size=2048, n_init="auto"
    )
    cluster_labels = kmeans.fit_predict(all_latents)

    cluster_map = np.zeros((h, w), dtype=np.int32)
    cluster_map[coords[:, 0], coords[:, 1]] = cluster_labels + 1

    return cluster_map, anomaly_map


def run_clustering(
    source_path: Path | str = "data/PaviaU.mat",
    weights_path: Path | str = Path("outputs/enmap_ssl_full.pth"),
    bbox: tuple[int, int, int, int] | None = None,
    n_clusters: int = 10,
    patch_size: int = 9,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if device is None:
        device = get_device()

    scene = load_scene(source_path, bbox=bbox)
    norm_data = normalize_data(scene.cube, valid_mask=scene.valid_mask)

    mae = MaskedConvolutionalAutoencoder(in_channels=norm_data.shape[-1]).to(device)
    load_pretrained_mae(mae, weights_path, device)

    cluster_map, anomaly_map = compute_latent_and_anomaly_maps(
        norm_data,
        scene.valid_mask,
        mae,
        device,
        patch_size=patch_size,
        n_clusters=n_clusters,
    )

    return cluster_map, anomaly_map, scene.valid_mask


if __name__ == "__main__":
    from .visualization import plot_clustering_and_anomalies

    parser = argparse.ArgumentParser(
        description="Latent Spatial-Spectral Clustering and Anomaly Detection"
    )
    parser.add_argument("source", nargs="?", default="data/enmap_sample")
    parser.add_argument("--bbox", nargs=4, type=int, default=None)
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/enmap_ssl_full.pth")
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=10,
        help="Number of latent K-Means clusters (default: 10)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/ssl_clusters_anomalies_10.png"),
    )
    args = parser.parse_args()

    bbox_tuple = tuple(args.bbox) if args.bbox else None
    c_map, a_map, mask = run_clustering(
        args.source,
        weights_path=args.weights,
        bbox=bbox_tuple,
        n_clusters=args.clusters,
    )
    out_file = plot_clustering_and_anomalies(
        c_map, a_map, mask, args.out, n_clusters=args.clusters
    )
    print(f"Cluster and anomaly visualization saved to: {out_file}")
