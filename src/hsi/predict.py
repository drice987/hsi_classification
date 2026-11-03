import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data_loader import SceneData, load_scene, normalize_data
from .dataset import PixelInferenceDataset
from .model import MaskedConvolutionalAutoencoder, SSLClassifier, get_device


def predict_scene(
    data: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    valid_mask: np.ndarray | None = None,
    patch_size: int = 9,
    batch_size: int = 512,
) -> np.ndarray:
    dataset = PixelInferenceDataset(data, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    h, w, _ = data.shape
    pred_map = np.zeros((h, w), dtype=np.uint8)

    with torch.no_grad():
        for patches, coords in loader:
            patches = patches.to(device)
            outputs = model(patches)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_coords, x_coords = coords[0].numpy(), coords[1].numpy()
            pred_map[y_coords, x_coords] = preds + 1

    if valid_mask is not None:
        pred_map[~valid_mask] = 0

    return pred_map


def run_prediction(
    source_path: Path | str = "data/PaviaU.mat",
    bbox: tuple[int, int, int, int] | None = None,
    weights_path: Path | str = "outputs/finetuned_model.pth",
    patch_size: int = 9,
) -> tuple[np.ndarray, SceneData]:
    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"Model weights not found: '{weights_file}'")

    device = get_device()
    scene = load_scene(source_path, bbox=bbox)
    norm_data = normalize_data(scene.cube, valid_mask=scene.valid_mask)

    mae = MaskedConvolutionalAutoencoder(in_channels=scene.cube.shape[-1])
    model = SSLClassifier(
        pretrained_encoder=mae.encoder,
        num_classes=len(scene.class_names),
        use_center_pixel=True,
    )
    model.load_state_dict(
        torch.load(weights_file, map_location=device, weights_only=True)
    )

    pred_map = predict_scene(
        norm_data, model, device, valid_mask=scene.valid_mask, patch_size=patch_size
    )
    return pred_map, scene


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sliding-window patch classification on a scene."
    )
    parser.add_argument("source", nargs="?", default="data/PaviaU.mat")
    parser.add_argument("--bbox", nargs=4, type=int, default=None)
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/finetuned_model.pth")
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else None
    pred_map, scene = run_prediction(
        source_path=args.source, bbox=bbox, weights_path=args.weights
    )
    print(f"Inference complete | Classified pixels: {int(np.sum(pred_map > 0))}")

    if args.out is not None:
        from .visualization import plot_classification_map

        plot_classification_map(
            pred_map,
            args.out,
            class_names=scene.class_names,
            title="Masked Autoencoder (SSL) Surface Classification",
        )


if __name__ == "__main__":
    main()
