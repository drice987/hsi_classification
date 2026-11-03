import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import load_scene
from .spectral_analysis import SpectroscopyResult, run_spectral_analysis


def _save_figure(fig: plt.Figure, out_path: Path | str) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_false_color(
    cube: np.ndarray,
    out_path: Path | str,
    bands: tuple[int, int, int] = (60, 30, 2),
) -> Path:
    n_bands = cube.shape[-1]
    safe_bands = [min(max(0, b), n_bands - 1) for b in bands]
    rgb = cube[:, :, safe_bands].astype(np.float32)
    p2, p98 = np.percentile(rgb, (2.0, 98.0))
    rgb_stretched = np.clip((rgb - p2) / (p98 - p2 + 1e-7), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(rgb_stretched)
    ax.set_title(f"Composite (Bands {safe_bands[0]}, {safe_bands[1]}, {safe_bands[2]})")
    ax.axis("off")
    return _save_figure(fig, out_path)


def plot_classification_map(
    label_map: np.ndarray,
    out_path: Path | str,
    class_names: list[str] | None = None,
    title: str = "Surface Classification",
) -> Path:
    num_classes = len(class_names) if class_names is not None else int(label_map.max())
    cmap = plt.get_cmap("tab20" if num_classes > 9 else "tab10", num_classes + 1)

    fig, ax = plt.subplots(figsize=(8, 11))
    im = ax.imshow(label_map, cmap=cmap, vmin=0, vmax=num_classes)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if class_names is not None:
        ticks = np.arange(len(class_names) + 1)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(["Unlabeled / Mask"] + class_names)
    else:
        cbar.set_label("Class Index (0 = Masked)")

    ax.set_title(title)
    ax.axis("off")
    return _save_figure(fig, out_path)


def plot_spectral_signatures(
    cube: np.ndarray,
    labels: np.ndarray,
    wavelengths: np.ndarray,
    out_path: Path | str,
    class_names: list[str],
) -> Path:
    cmap = plt.get_cmap("tab10", len(class_names) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    gap_idx = np.nonzero(np.diff(wavelengths) > 35.0)[0] + 1
    wl_plot = (
        np.insert(wavelengths.astype(np.float64), gap_idx, np.nan)
        if gap_idx.size > 0
        else wavelengths
    )

    for class_id, name in enumerate(class_names, start=1):
        mask = labels == class_id
        if np.any(mask):
            mean_spec = cube[mask].mean(axis=0).astype(np.float64)
            if gap_idx.size > 0:
                mean_spec = np.insert(mean_spec, gap_idx, np.nan)
            ax.plot(wl_plot, mean_spec, label=name, color=cmap(class_id), lw=1.8)

    ax.set_title("Mean Class Spectral Signatures")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    return _save_figure(fig, out_path)


def plot_clustering_and_anomalies(
    cluster_map: np.ndarray,
    anomaly_map: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path | str,
    n_clusters: int = 10,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    cmap = plt.get_cmap("tab20", n_clusters + 1)
    im0 = axes[0].imshow(cluster_map, cmap=cmap, vmin=0, vmax=n_clusters)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Cluster ID (0=Mask)")
    axes[0].set_title(f"Latent Clusters (k={n_clusters})")
    axes[0].axis("off")

    vmax = (
        float(np.percentile(anomaly_map[valid_mask], 99.0))
        if np.any(valid_mask)
        else 1.0
    )
    im1 = axes[1].imshow(anomaly_map, cmap="inferno", vmin=0.0, vmax=max(vmax, 1e-4))
    fig.colorbar(
        im1, ax=axes[1], fraction=0.046, pad=0.04, label="Center Reconstruction RMSE"
    )
    axes[1].set_title("Self-Supervised Spatial-Spectral Anomaly Map")
    axes[1].axis("off")

    return _save_figure(fig, out_path)


def plot_spectroscopy_summary(
    results: SpectroscopyResult,
    out_path: Path | str,
) -> Path:
    wavelengths = results.wavelengths
    endmembers = results.endmembers
    window_results = results.window_results
    rmse_map = results.rmse_map
    convolved_centers = results.convolved_centers
    convolved_endmembers = results.convolved_endmembers
    sensor_name = results.sensor_name

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    gap_idx = np.nonzero(np.diff(wavelengths) > 35.0)[0] + 1
    if gap_idx.size > 0:
        wl_plot = np.insert(wavelengths.astype(np.float64), gap_idx, np.nan)
        em_plot = np.insert(endmembers.astype(np.float64), gap_idx, np.nan, axis=-1)
    else:
        wl_plot = wavelengths
        em_plot = endmembers

    ax0 = axes[0, 0]
    palette = ["#1b7837", "#d95f02", "#8c510a", "#2b5c8f", "#7570b3"]
    for i, spec in enumerate(em_plot):
        color = palette[i % len(palette)]
        ax0.plot(wl_plot, spec, lw=1.6, color=color, label=f"Endmember {i + 1}")
        if convolved_centers is not None and convolved_endmembers is not None:
            ax0.scatter(
                convolved_centers,
                convolved_endmembers[i],
                color=color,
                s=24,
                zorder=4,
            )

    title = "ATGP Extracted Endmembers"
    if sensor_name:
        title += f" (Dots: {sensor_name} Bandpass)"
    ax0.set_title(title, fontsize=11)
    ax0.set_xlabel("Wavelength (nm)")
    ax0.set_ylabel("Surface Reflectance")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[0, 1]
    if "chlorophyll_red_edge" in window_results:
        depth = window_results["chlorophyll_red_edge"]["depth"]
        im1 = ax1.imshow(depth, cmap="YlGn", vmin=0.0, vmax=0.85)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Band Depth")
        ax1.set_title("Chlorophyll Red-Edge Absorption Depth (670-760 nm)", fontsize=11)
    ax1.axis("off")

    ax2 = axes[1, 0]
    if "clay_al_oh" in window_results:
        clay = window_results["clay_al_oh"]["depth"]
        vmax = max(float(np.percentile(clay, 99.0)), 0.05)
        im2 = ax2.imshow(clay, cmap="OrRd", vmin=0.0, vmax=vmax)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Al-OH Depth")
        ax2.set_title("Clay Mineral Absorption (2160-2240 nm)", fontsize=11)
    elif "chlorophyll_red_edge" in window_results:
        reip = window_results["chlorophyll_red_edge"]["reip_nm"]
        depth = window_results["chlorophyll_red_edge"]["depth"]
        masked_reip = np.where(depth > 0.15, reip, np.nan)
        im2 = ax2.imshow(masked_reip, cmap="viridis", vmin=695.0, vmax=735.0)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="REIP (nm)")
        ax2.set_title("Red-Edge Inflection Point (Vegetation)", fontsize=11)
    ax2.axis("off")

    ax3 = axes[1, 1]
    vmax_rmse = max(float(np.percentile(rmse_map, 99.0)), 1e-3)
    im3 = ax3.imshow(rmse_map, cmap="magma", vmin=0.0, vmax=vmax_rmse)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Spectral RMSE")
    ax3.set_title("Linear Unmixing Reconstruction Residual", fontsize=11)
    ax3.axis("off")

    return _save_figure(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render figures for hyperspectral scenes, spectroscopy, clusters, and predictions."
    )
    parser.add_argument("source", nargs="?", default="data/PaviaU.mat")
    parser.add_argument(
        "--mode",
        choices=["all", "overview", "spectroscopy", "clusters", "prediction"],
        default="all",
    )
    parser.add_argument("--bbox", nargs=4, type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--convolve", choices=["s2", "s3", "metimage", "fci"], default=None
    )
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument(
        "--ssl-weights", type=Path, default=Path("outputs/ssl_encoder_pretrained.pth")
    )
    parser.add_argument(
        "--clf-weights", type=Path, default=Path("outputs/finetuned_model.pth")
    )
    parser.add_argument("--gt-tif", type=Path, default=None)
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else None
    scene = load_scene(args.source, gt_path=args.gt_tif, bbox=bbox)
    out_dir = args.out_dir
    prefix = "enmap" if Path(args.source).is_dir() else "pavia"

    if args.mode in ("all", "overview"):
        p_fc = plot_false_color(scene.cube, out_dir / f"{prefix}_false_color.png")
        p_gt = plot_classification_map(
            scene.labels,
            out_dir / f"{prefix}_ground_truth.png",
            class_names=scene.class_names,
            title="Ground Truth Labels",
        )
        p_sig = plot_spectral_signatures(
            scene.cube,
            scene.labels,
            scene.wavelengths,
            out_dir / f"{prefix}_spectral_signatures.png",
            class_names=scene.class_names,
        )
        print(f"Saved overview figures: {p_fc}, {p_gt}, {p_sig}")

    if args.mode in ("all", "spectroscopy"):
        spec_res = run_spectral_analysis(
            scene.cube,
            scene.wavelengths,
            convolve_sensor=args.convolve,
            valid_mask=scene.valid_mask,
        )
        p_spec = plot_spectroscopy_summary(
            spec_res, out_dir / f"{prefix}_spectroscopy_summary.png"
        )
        print(f"Saved spectroscopy summary: {p_spec}")

    if args.mode in ("all", "clusters"):
        if args.ssl_weights.exists():
            from .cluster import run_clustering

            c_map, a_map, v_mask = run_clustering(
                args.source,
                bbox=bbox,
                weights_path=args.ssl_weights,
                n_clusters=args.clusters,
            )
            p_cl = plot_clustering_and_anomalies(
                c_map,
                a_map,
                v_mask,
                out_dir / f"ssl_clusters_anomalies_{args.clusters}.png",
                n_clusters=args.clusters,
            )
            print(f"Saved cluster/anomaly figure: {p_cl}")
        elif args.mode == "clusters":
            raise FileNotFoundError(f"Missing SSL weights: {args.ssl_weights}")

    if args.mode in ("all", "prediction"):
        if args.clf_weights.exists():
            from .predict import run_prediction

            pred_map, _ = run_prediction(
                args.source, bbox=bbox, weights_path=args.clf_weights
            )
            p_pr = plot_classification_map(
                pred_map,
                out_dir / "ssl_full_prediction_map.png",
                class_names=scene.class_names,
                title="Masked Autoencoder (SSL) Surface Classification",
            )
            print(f"Saved prediction map: {p_pr}")
        elif args.mode == "prediction":
            raise FileNotFoundError(f"Missing classifier weights: {args.clf_weights}")


if __name__ == "__main__":
    main()
