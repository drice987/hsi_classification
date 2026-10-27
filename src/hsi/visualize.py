from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import CLASS_NAMES, load_pavia_data


def save_false_color_image(data: np.ndarray, out_dir: Path) -> None:
    rgb = data[:, :, [60, 30, 2]]
    rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    plt.figure(figsize=(6, 10))
    plt.imshow(rgb_normalized)
    plt.title("Pavia False Color Composite")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "false_color.png", dpi=300)
    plt.close()


def save_ground_truth_map(gt: np.ndarray, out_dir: Path) -> None:
    cmap = plt.get_cmap("nipy_spectral", 10)

    plt.figure(figsize=(6, 10))
    plt.imshow(gt, cmap=cmap, vmin=0, vmax=9)
    plt.title("Ground Truth Labels")
    plt.colorbar(ticks=range(10), label="Class ID")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "ground_truth.png", dpi=300)
    plt.close()


def save_spectral_signatures(
    data: np.ndarray,
    gt: np.ndarray,
    out_dir: Path,
    spectral_range: tuple[float, float] = (430.0, 860.0),
) -> None:
    wavelengths = np.linspace(spectral_range[0], spectral_range[1], data.shape[-1])
    cmap = plt.get_cmap("nipy_spectral", 10)

    plt.figure(figsize=(10, 6))

    for class_id, name in enumerate(CLASS_NAMES, start=1):
        y_coords, x_coords = np.where(gt == class_id)
        if y_coords.size > 0:
            mean_signature = data[y_coords, x_coords, :].mean(axis=0)
            plt.plot(
                wavelengths,
                mean_signature,
                label=name,
                color=cmap(class_id),
                linewidth=2,
            )

    plt.title("Mean Spectral Signatures of Pavia Materials", fontsize=14)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Reflectance Intensity", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "spectral_signatures.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    out_path = Path("outputs")
    out_path.mkdir(parents=True, exist_ok=True)

    cube, ground_truth = load_pavia_data()
    save_false_color_image(cube, out_path)
    save_ground_truth_map(ground_truth, out_path)
    save_spectral_signatures(cube, ground_truth, out_path)
