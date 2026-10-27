import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import CLASS_NAMES, load_scene, split_spatial_gt


def prepare_tabular_data(
    data: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flattens the 3D hyperspectral cube and removes unlabeled background pixels."""
    bands = data.shape[-1]
    x_flat = data.reshape(-1, bands)
    y_flat = gt.reshape(-1)

    valid_pixels = y_flat > 0
    return x_flat[valid_pixels], y_flat[valid_pixels] - 1


def build_rf_pipeline(n_components: int = 30) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            (
                "rf",
                RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            ),
        ]
    )


def evaluate_baseline(
    data: np.ndarray,
    gt: np.ndarray,
    n_components: int = 30,
    class_names: list[str] | None = None,
) -> None:
    """Evaluates PCA + Random Forest under random and spatially disjoint splits."""
    if class_names is None:
        class_names = CLASS_NAMES

    label_indices = list(range(len(class_names)))
    x_all, y_all = prepare_tabular_data(data, gt)
    n_pca = min(n_components, data.shape[-1], len(x_all))

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )

    pipeline = build_rf_pipeline(n_components=n_pca)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    print("\n--- Baseline RF (Random Stratified Split) ---")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=label_indices,
            target_names=class_names,
            zero_division=0,
        )
    )

    gt_train, gt_val, _ = split_spatial_gt(gt, train_frac=0.75)
    x_sp_train, y_sp_train = prepare_tabular_data(data, gt_train)
    x_sp_val, y_sp_val = prepare_tabular_data(data, gt_val)

    sp_pipeline = build_rf_pipeline(n_components=min(n_pca, len(x_sp_train)))
    sp_pipeline.fit(x_sp_train, y_sp_train)
    y_sp_pred = sp_pipeline.predict(x_sp_val)

    print("\n--- Baseline RF (75/25 Spatial Column Split) ---")
    print(
        classification_report(
            y_sp_val,
            y_sp_pred,
            labels=label_indices,
            target_names=class_names,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PCA + RF Baseline")
    parser.add_argument("source", nargs="?", default="data/PaviaU.mat")
    parser.add_argument("--bbox", nargs=4, type=int, default=None)
    args = parser.parse_args()

    bbox_tuple = tuple(args.bbox) if args.bbox else None
    cube, ground_truth, _, names = load_scene(Path(args.source), bbox=bbox_tuple)
    evaluate_baseline(cube, ground_truth, class_names=names)
