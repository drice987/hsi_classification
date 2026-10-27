import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baseline import prepare_tabular_data
from .data_loader import load_scene, split_spatial_gt

ABSORPTION_WINDOWS: dict[str, tuple[float, float]] = {
    "chlorophyll_red_edge": (670.0, 760.0),
    "water_nir": (920.0, 1020.0),
    "cellulose_swir": (2030.0, 2140.0),
    "clay_al_oh": (2160.0, 2240.0),
    "carbonate_swir": (2280.0, 2360.0),
}

SENTINEL2_MSI_BANDS: dict[int, tuple[float, float]] = {
    1: (442.7, 20.0),
    2: (492.4, 65.0),
    3: (559.8, 35.0),
    4: (664.6, 30.0),
    5: (704.1, 14.0),
    6: (740.5, 14.0),
    7: (782.8, 19.0),
    8: (832.8, 105.0),
    9: (864.7, 21.0),
    10: (945.1, 20.0),
    11: (1613.7, 90.0),
    12: (2202.4, 174.0),
}

OLCI_VNIR_BANDS: dict[int, tuple[float, float]] = {
    3: (442.5, 10.0),
    4: (490.0, 10.0),
    5: (510.0, 10.0),
    6: (560.0, 10.0),
    7: (620.0, 10.0),
    8: (665.0, 10.0),
    9: (673.75, 7.5),
    10: (681.25, 7.5),
    11: (708.75, 10.0),
    12: (753.75, 7.5),
    13: (761.25, 2.5),
    14: (764.375, 3.75),
    15: (767.5, 2.5),
    16: (778.75, 15.0),
    17: (865.0, 20.0),
}

SENSOR_SRF_SPECS: dict[str, dict[int, tuple[float, float]]] = {
    "s2": SENTINEL2_MSI_BANDS,
    "s3": OLCI_VNIR_BANDS,
}


def get_wavelength_axis(
    n_bands: int = 103, bounds_nm: tuple[float, float] = (430.0, 860.0)
) -> np.ndarray:
    return np.linspace(bounds_nm[0], bounds_nm[1], n_bands, dtype=np.float32)


def build_srf_weights(
    wavelengths: np.ndarray,
    band_specs: dict[int, tuple[float, float]] = OLCI_VNIR_BANDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds a row-normalized Gaussian Spectral Response Function (SRF) matrix."""
    wl_min, wl_max = float(wavelengths.min()), float(wavelengths.max())
    weights = []
    centers = []
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    for _, (center_nm, fwhm_nm) in sorted(band_specs.items()):
        if center_nm < wl_min - fwhm_nm or center_nm > wl_max + fwhm_nm:
            continue
        sigma = fwhm_nm * fwhm_to_sigma
        profile = np.exp(-0.5 * ((wavelengths - center_nm) / sigma) ** 2)
        profile_sum = float(np.sum(profile))
        if profile_sum <= 1e-8:
            continue
        weights.append(profile / profile_sum)
        centers.append(center_nm)

    if not weights:
        raise ValueError("No target sensor bands overlap the input wavelength range.")

    return np.vstack(weights).astype(np.float32), np.array(centers, dtype=np.float32)


def resample_cube(cube: np.ndarray, srf_weights: np.ndarray) -> np.ndarray:
    """Projects a 3D hyperspectral cube onto target multispectral SRF channels."""
    h, w, c = cube.shape
    projected = cube.reshape(-1, c) @ srf_weights.T
    return projected.reshape(h, w, srf_weights.shape[0])


def _parabolic_subband_extremum(
    values: np.ndarray, wl_axis: np.ndarray, discrete_idx: np.ndarray
) -> np.ndarray:
    """Refines discrete band extrema using 3-point parabolic vertex interpolation."""
    n_samples, n_bands = values.shape
    clamped = np.clip(discrete_idx, 1, n_bands - 2)
    rows = np.arange(n_samples)

    y_prev = values[rows, clamped - 1]
    y_curr = values[rows, clamped]
    y_next = values[rows, clamped + 1]

    denom = 2.0 * (y_prev - 2.0 * y_curr + y_next)
    delta = np.divide(
        y_prev - y_next,
        denom,
        out=np.zeros_like(y_prev),
        where=np.abs(denom) > 1e-9,
    )
    delta = np.clip(delta, -0.5, 0.5)

    is_boundary = (discrete_idx == 0) | (discrete_idx == n_bands - 1)
    delta = np.where(is_boundary, 0.0, delta)

    step_nm = 0.5 * (wl_axis[clamped + 1] - wl_axis[clamped - 1])
    return (wl_axis[discrete_idx] + delta * step_nm).astype(np.float32)


def _compute_upper_hull_continuum(
    spec_win: np.ndarray, wl_win: np.ndarray
) -> np.ndarray:
    """Computes an upper envelope continuum anchored at shoulder maxima."""
    n_pixels, n_bands = spec_win.shape
    rows = np.arange(n_pixels)

    margin = max(1, n_bands // 4)
    left_idx = np.argmax(spec_win[:, :margin], axis=1)
    right_idx = (n_bands - margin) + np.argmax(spec_win[:, n_bands - margin :], axis=1)

    wl_left = wl_win[left_idx][:, np.newaxis]
    wl_right = wl_win[right_idx][:, np.newaxis]
    r_left = spec_win[rows, left_idx][:, np.newaxis]
    r_right = spec_win[rows, right_idx][:, np.newaxis]

    span = np.maximum(wl_right - wl_left, 1e-5)
    frac = np.clip((wl_win[np.newaxis, :] - wl_left) / span, 0.0, 1.0)
    continuum = r_left + frac * (r_right - r_left)

    return np.maximum(continuum, spec_win)


def compute_absorption_feature(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    window_nm: tuple[float, float],
    min_continuum_refl: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes upper-hull continuum removal and sub-band parabolic absorption minimum."""
    mask = (wavelengths >= window_nm[0]) & (wavelengths <= window_nm[1])
    if int(np.sum(mask)) < 3:
        raise ValueError(
            f"Wavelength axis ({wavelengths[0]:.1f}-{wavelengths[-1]:.1f} nm) "
            f"has fewer than 3 bands in window {window_nm}."
        )

    wl_win = wavelengths[mask]
    spec_win = spectra[:, mask]

    continuum = _compute_upper_hull_continuum(spec_win, wl_win)
    valid_cont = continuum > 1e-4
    continuum_removed = np.where(
        valid_cont, spec_win / np.maximum(continuum, 1e-5), 1.0
    )
    continuum_removed = np.clip(continuum_removed, 0.0, 1.0)
    absorption = 1.0 - continuum_removed

    depth = np.max(absorption, axis=1)
    area = np.trapezoid(absorption, x=wl_win, axis=1)

    min_idx = np.argmin(continuum_removed, axis=1)
    min_wl = _parabolic_subband_extremum(continuum_removed, wl_win, min_idx)

    has_signal = (np.max(continuum, axis=1) >= min_continuum_refl) & (depth > 1e-4)
    depth = np.where(has_signal, depth, 0.0)
    area = np.where(has_signal, area, 0.0)
    min_wl = np.where(has_signal, min_wl, wl_win[0])

    metrics = np.column_stack([depth, area, min_wl]).astype(np.float32)
    return metrics, wl_win, continuum_removed.astype(np.float32)


def compute_red_edge_metrics(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    window_nm: tuple[float, float] = (670.0, 760.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates continuum-removed chlorophyll depth, area, and sub-band REIP."""
    ab_metrics, wl_re, continuum_removed = compute_absorption_feature(
        spectra, wavelengths, window_nm
    )
    mask = (wavelengths >= window_nm[0]) & (wavelengths <= window_nm[1])
    spec_re = spectra[:, mask]

    first_deriv = np.gradient(spec_re, wl_re, axis=1)
    max_deriv_idx = np.argmax(first_deriv, axis=1)
    reip_nm = _parabolic_subband_extremum(-first_deriv, wl_re, max_deriv_idx)

    has_signal = np.any(spec_re > 1e-4, axis=1)
    reip_nm = np.where(has_signal, reip_nm, wl_re[0])

    metrics = np.column_stack([ab_metrics[:, 0], ab_metrics[:, 1], reip_nm]).astype(
        np.float32
    )
    return metrics, wl_re, continuum_removed


def analyze_diagnostic_windows(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    windows: dict[str, tuple[float, float]] = ABSORPTION_WINDOWS,
) -> dict[str, dict[str, np.ndarray]]:
    """Extracts 2D maps of band depth, area, and sub-band wavelength position."""
    h, w, c = cube.shape
    flat_spectra = cube.reshape(-1, c)
    results: dict[str, dict[str, np.ndarray]] = {}

    for name, bounds in windows.items():
        mask = (wavelengths >= bounds[0]) & (wavelengths <= bounds[1])
        if int(np.sum(mask)) < 4:
            continue

        if name == "chlorophyll_red_edge":
            metrics, wl_win, _ = compute_red_edge_metrics(
                flat_spectra, wavelengths, bounds
            )
            pos_key = "reip_nm"
        else:
            metrics, wl_win, _ = compute_absorption_feature(
                flat_spectra, wavelengths, bounds
            )
            pos_key = "min_wavelength_nm"

        results[name] = {
            "depth": metrics[:, 0].reshape(h, w),
            "area": metrics[:, 1].reshape(h, w),
            pos_key: metrics[:, 2].reshape(h, w),
            "wavelengths": wl_win,
        }

    return results


def extract_endmembers_atgp(
    cube: np.ndarray,
    n_endmembers: int = 4,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Extracts pure spectral endmembers via shape-normalized Orthogonal Subspace Projection."""
    _, _, c = cube.shape
    flat = cube.reshape(-1, c).astype(np.float64)

    if valid_mask is not None:
        valid_flat = flat[valid_mask.reshape(-1)]
    else:
        valid_flat = flat[np.any(flat > 1e-4, axis=1)]

    if len(valid_flat) < n_endmembers:
        raise ValueError("Insufficient valid pixels to extract ATGP endmembers.")

    band_activity = np.mean(valid_flat > 1e-3, axis=1)
    full_spec_mask = band_activity >= 0.80
    water_like_mask = (band_activity > 0.15) & (band_activity < 0.80)

    if np.sum(full_spec_mask) >= n_endmembers:
        pool = valid_flat[full_spec_mask]
    else:
        pool = valid_flat

    norms = np.linalg.norm(pool, axis=1)
    low_cap, high_cap = np.percentile(norms, [5.0, 99.5])
    filtered = pool[(norms >= low_cap) & (norms <= high_cap)]
    candidates = filtered if len(filtered) >= n_endmembers else pool

    cand_norms = np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12)
    unit_candidates = candidates / cand_norms

    reserve_water = (
        np.sum(water_like_mask) > 100 and np.sum(full_spec_mask) >= n_endmembers >= 3
    )
    n_land = n_endmembers - 1 if reserve_water else n_endmembers

    endmembers = np.zeros((n_endmembers, c), dtype=np.float64)
    unit_basis = np.zeros((n_land, c), dtype=np.float64)

    first_idx = int(np.argmax(np.linalg.norm(candidates, axis=1)))

    for k in range(n_land):
        if k == 0:
            target_idx = first_idx
        else:
            q_mat, _ = np.linalg.qr(unit_basis[:k].T)
            proj = (unit_candidates @ q_mat) @ q_mat.T
            residual_norms = np.sum((unit_candidates - proj) ** 2, axis=1)
            target_idx = int(np.argmax(residual_norms))

        target_unit = unit_candidates[target_idx]
        unit_basis[k] = target_unit

        cos_sim = unit_candidates @ target_unit
        top_k = min(15, len(candidates))
        nearest_idx = np.argpartition(cos_sim, -top_k)[-top_k:]
        endmembers[k] = candidates[nearest_idx].mean(axis=0)

    if reserve_water:
        water_pixels = valid_flat[water_like_mask]
        endmembers[-1] = np.median(water_pixels, axis=0)

    return endmembers.astype(np.float32)


def unmix_scene_atgp(
    cube: np.ndarray,
    n_endmembers: int = 4,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs ATGP endmember extraction followed by NNLS unmixing and RMSE mapping."""
    h, w, c = cube.shape
    if valid_mask is None:
        valid_mask = np.any(cube > 1e-4, axis=-1)

    endmembers = extract_endmembers_atgp(
        cube, n_endmembers=n_endmembers, valid_mask=valid_mask
    )
    design_matrix = endmembers.T.astype(np.float64)

    flat_pixels = cube.reshape(-1, c).astype(np.float64)
    valid_indices = np.nonzero(valid_mask.reshape(-1))[0]

    abundances = np.zeros((flat_pixels.shape[0], n_endmembers), dtype=np.float32)
    rmse_flat = np.zeros(flat_pixels.shape[0], dtype=np.float32)

    for i in valid_indices:
        pixel = flat_pixels[i]
        coeff, _ = nnls(design_matrix, pixel)
        recon = design_matrix @ coeff
        rmse_flat[i] = float(np.sqrt(np.mean((pixel - recon) ** 2)))
        total = coeff.sum()
        if total > 1e-8:
            abundances[i] = coeff / total

    return (
        abundances.reshape(h, w, n_endmembers),
        endmembers,
        rmse_flat.reshape(h, w),
    )


def unmix_nnls(
    cube: np.ndarray,
    gt_train: np.ndarray | None = None,
    endmember_ids: dict[str, int] | None = None,
    n_endmembers: int = 4,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Solves bounded non-negative least squares using labeled or ATGP endmembers."""
    h, w, c = cube.shape

    if gt_train is not None and np.any(gt_train > 0):
        unique_present = [int(cid) for cid in np.unique(gt_train) if cid > 0]
        if endmember_ids is None:
            if set(unique_present) >= {1, 2, 6, 8}:
                candidate_ids = {
                    "Vegetation": 2,
                    "Asphalt": 1,
                    "Bricks": 8,
                    "Bare Soil": 6,
                }
            else:
                candidate_ids = {f"Class_{cid}": cid for cid in unique_present[:4]}
        else:
            candidate_ids = endmember_ids

        basis_rows = []
        active_names = []
        for name, cid in candidate_ids.items():
            mask = gt_train == cid
            if np.any(mask):
                basis_rows.append(cube[mask].mean(axis=0))
                active_names.append(name)

        if not basis_rows:
            basis_rows = [
                cube[gt_train == cid].mean(axis=0) for cid in unique_present[:3]
            ]
            active_names = [f"Class_{cid}" for cid in unique_present[:3]]

        basis = np.vstack(basis_rows).astype(np.float32)
    else:
        basis = extract_endmembers_atgp(cube, n_endmembers=n_endmembers)
        active_names = [f"Endmember_{i + 1}" for i in range(len(basis))]

    design_matrix = basis.T.astype(np.float64)
    flat_pixels = cube.reshape(-1, c).astype(np.float64)
    abundances = np.zeros((flat_pixels.shape[0], len(active_names)), dtype=np.float32)

    valid_rows = np.nonzero(np.any(flat_pixels > 0.0, axis=1))[0]
    for i in valid_rows:
        abundances[i], _ = nnls(design_matrix, flat_pixels[i])

    row_sums = abundances.sum(axis=1, keepdims=True)
    abundances = np.divide(
        abundances, row_sums, out=np.zeros_like(abundances), where=row_sums > 1e-8
    )
    maps = {
        name: abundances[:, idx].reshape(h, w) for idx, name in enumerate(active_names)
    }
    return maps, basis


def train_olci_red_edge_estimator(
    olci_cube: np.ndarray,
    re_metrics_cube: np.ndarray,
    gt: np.ndarray,
    target_names: tuple[str, ...] = ("Band_Depth", "Absorption_Area", "REIP_nm"),
) -> tuple[Pipeline, dict[str, float]]:
    """Trains a regressor mapping multispectral bands to continuous hyperspectral metrics."""
    gt_train, gt_val, _ = split_spatial_gt(gt, train_frac=0.75)

    train_mask = gt_train.reshape(-1) > 0
    val_mask = gt_val.reshape(-1) > 0

    if not np.any(train_mask) or not np.any(val_mask):
        raise ValueError("Spatial split contains 0 valid pixels.")

    x_ms = olci_cube.reshape(-1, olci_cube.shape[-1])
    n_targets = re_metrics_cube.shape[-1]
    y_targets = re_metrics_cube.reshape(-1, n_targets)

    rng = np.random.default_rng(42)
    tr_idx = np.nonzero(train_mask)[0]
    if len(tr_idx) > 15000:
        tr_idx = rng.choice(tr_idx, size=15000, replace=False)

    regressor = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf_reg",
                RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            ),
        ]
    )
    regressor.fit(x_ms[tr_idx], y_targets[tr_idx])
    preds = regressor.predict(x_ms[val_mask])
    if preds.ndim == 1:
        preds = preds[:, np.newaxis]

    r2_scores = {
        name: float(r2_score(y_targets[val_mask, idx], preds[:, idx]))
        for idx, name in enumerate(target_names[:n_targets])
    }
    return regressor, r2_scores


def run_benchmark(
    hsi_cube: np.ndarray,
    convolved_cube: np.ndarray,
    re_metrics_cube: np.ndarray,
    gt: np.ndarray,
    sensor_label: str = "Convolved Multispectral",
) -> dict[str, tuple[float, float]]:
    gt_train, gt_val, _ = split_spatial_gt(gt, train_frac=0.75)
    x_tr_check, _ = prepare_tabular_data(convolved_cube, gt_train)
    x_va_check, _ = prepare_tabular_data(convolved_cube, gt_val)

    if len(x_tr_check) == 0 or len(x_va_check) == 0:
        raise ValueError("Selected region has no valid pixels in train or val split.")

    hybrid_cube = np.concatenate([hsi_cube, re_metrics_cube], axis=-1)
    n_bands = hsi_cube.shape[-1]
    n_ms = convolved_cube.shape[-1]
    n_pca = min(30, n_bands, len(x_tr_check))

    modalities = {
        f"{sensor_label} ({n_ms} bands)": (convolved_cube, None),
        f"Full HSI ({n_bands} bands, PCA-{n_pca})": (hsi_cube, n_pca),
        "HSI + Continuum Absorption Metrics": (hybrid_cube, n_pca),
    }

    scores: dict[str, tuple[float, float]] = {}
    for name, (arr, pca_comp) in modalities.items():
        x_tr, y_tr = prepare_tabular_data(arr, gt_train)
        x_va, y_va = prepare_tabular_data(arr, gt_val)

        steps = [("scaler", StandardScaler())]
        if pca_comp is not None:
            steps.append(("pca", PCA(n_components=pca_comp, random_state=42)))
        steps.append(
            ("rf", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1))
        )

        clf = Pipeline(steps)
        clf.fit(x_tr, y_tr)
        preds = clf.predict(x_va)

        oa = accuracy_score(y_va, preds) * 100.0
        f1 = f1_score(y_va, preds, average="macro", zero_division=0) * 100.0
        scores[name] = (oa, f1)
        print(f"{name:<38} | OA: {oa:5.2f}% | Macro F1: {f1:5.2f}%")

    return scores


def _insert_nan_at_spectral_gaps(
    wavelengths: np.ndarray, spectra: np.ndarray, max_step_nm: float = 35.0
) -> tuple[np.ndarray, np.ndarray]:
    gap_indices = np.nonzero(np.diff(wavelengths) > max_step_nm)[0] + 1
    if gap_indices.size == 0:
        return wavelengths, spectra

    wl_plot = np.insert(wavelengths.astype(np.float64), gap_indices, np.nan)
    spec_plot = np.insert(spectra.astype(np.float64), gap_indices, np.nan, axis=-1)
    return wl_plot, spec_plot


def plot_spectroscopy_summary(
    wavelengths: np.ndarray,
    endmembers: np.ndarray,
    window_results: dict[str, dict[str, np.ndarray]],
    rmse_map: np.ndarray,
    out_path: Path,
    convolved_centers: np.ndarray | None = None,
    convolved_endmembers: np.ndarray | None = None,
    sensor_name: str | None = None,
) -> None:
    """Renders a 4-panel physical spectroscopy and unmixing summary figure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    wl_plot, em_plot = _insert_nan_at_spectral_gaps(wavelengths, endmembers)

    ax0 = axes[0, 0]
    colors = ["#1b7837", "#d95f02", "#8c510a", "#2b5c8f", "#7570b3"]
    for idx, spec in enumerate(em_plot):
        col = colors[idx % len(colors)]
        ax0.plot(wl_plot, spec, lw=1.6, color=col, label=f"ATGP Endmember {idx + 1}")
        if convolved_centers is not None and convolved_endmembers is not None:
            ax0.scatter(
                convolved_centers,
                convolved_endmembers[idx],
                color=col,
                s=28,
                zorder=4,
            )

    title_a = "ATGP Autonomous Endmember Spectra"
    if sensor_name:
        title_a += f" (Dots: {sensor_name} Bandpass)"
    ax0.set_title(title_a, fontsize=11)
    ax0.set_xlabel("Wavelength (nm)")
    ax0.set_ylabel("Surface Reflectance")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[0, 1]
    if "chlorophyll_red_edge" in window_results:
        chl_depth = window_results["chlorophyll_red_edge"]["depth"]
        im1 = ax1.imshow(chl_depth, cmap="YlGn", vmin=0.0, vmax=0.85)
        plt.colorbar(
            im1, ax=ax1, fraction=0.046, pad=0.04, label="Continuum Band Depth"
        )
        ax1.set_title("Chlorophyll Red-Edge Absorption Depth (670-760 nm)", fontsize=11)
    ax1.axis("off")

    ax2 = axes[1, 0]
    if "clay_al_oh" in window_results:
        clay_depth = window_results["clay_al_oh"]["depth"]
        vmax_clay = max(float(np.percentile(clay_depth, 99)), 0.05)
        im2 = ax2.imshow(clay_depth, cmap="OrRd", vmin=0.0, vmax=vmax_clay)
        plt.colorbar(
            im2, ax=ax2, fraction=0.046, pad=0.04, label="Al-OH Band Depth (~2200 nm)"
        )
        ax2.set_title("SWIR Clay Mineral Al-OH Absorption (2160-2240 nm)", fontsize=11)
    elif "chlorophyll_red_edge" in window_results:
        reip_map = window_results["chlorophyll_red_edge"]["reip_nm"]
        chl_depth = window_results["chlorophyll_red_edge"]["depth"]
        masked_reip = np.where(chl_depth > 0.15, reip_map, np.nan)
        im2 = ax2.imshow(masked_reip, cmap="viridis", vmin=695.0, vmax=735.0)
        plt.colorbar(
            im2, ax=ax2, fraction=0.046, pad=0.04, label="REIP Wavelength (nm)"
        )
        ax2.set_title("Sub-Band Red-Edge Inflection Point (Veg Pixels)", fontsize=11)
    ax2.axis("off")

    ax3 = axes[1, 1]
    vmax_rmse = max(float(np.percentile(rmse_map, 99)), 1e-3)
    im3 = ax3.imshow(rmse_map, cmap="magma", vmin=0.0, vmax=vmax_rmse)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Spectral RMSE")
    ax3.set_title("ATGP + NNLS Unmixing Residual (RMSE)", fontsize=11)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VNIR-SWIR Continuum Spectroscopy, ATGP Unmixing, and Bandpass Simulation"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="data/PaviaU.mat",
        help="Path to PaviaU .mat file or an EnMAP L2A product directory",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        default=None,
        metavar=("R0", "R1", "C0", "C1"),
        help="Spatial bounding box crop (row_min row_max col_min col_max)",
    )
    parser.add_argument(
        "--endmembers",
        type=int,
        default=4,
        help="Number of autonomous ATGP endmembers to extract (default: 4)",
    )
    parser.add_argument(
        "--convolve",
        choices=["s2", "s3"],
        default=None,
        help="Optional bandpass degradation to Sentinel-2 MSI (s2) or Sentinel-3 OLCI (s3)",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    is_enmap_dir = source_path.is_dir()
    bbox = tuple(args.bbox) if args.bbox else None

    data, gt, wavelengths, _ = load_scene(source_path, bbox=bbox)
    valid_mask = np.any(data > 1e-4, axis=-1)
    valid_count = int(np.sum(valid_mask))

    print(f"\nScene:            {source_path.name}")
    print(
        f"Spatial grid:     {data.shape[0]} x {data.shape[1]} ({valid_count} valid pixels)"
    )
    print(
        f"Spectral range:   {data.shape[2]} bands ({wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm)"
    )

    window_results = analyze_diagnostic_windows(data, wavelengths)
    print("\n--- Continuum-Removed Absorption Features (Sub-Band Parabolic Fit) ---")
    for win_name, win_data in window_results.items():
        depth_vals = win_data["depth"][valid_mask]
        active = depth_vals > 0.05
        mean_depth = float(depth_vals[active].mean()) if np.any(active) else 0.0
        pos_key = "reip_nm" if "reip_nm" in win_data else "min_wavelength_nm"
        pos_vals = win_data[pos_key][valid_mask]
        mean_pos = float(pos_vals[active].mean()) if np.any(active) else 0.0
        print(
            f"  {win_name:<22} | Active Pixels: {int(np.sum(active)):>7} | "
            f"Mean Depth: {mean_depth:5.3f} | Mean {pos_key}: {mean_pos:6.2f} nm"
        )

    _, endmembers, rmse_map = unmix_scene_atgp(
        data, n_endmembers=args.endmembers, valid_mask=valid_mask
    )
    mean_rmse = float(rmse_map[valid_mask].mean()) if valid_count > 0 else 0.0
    print(
        f"\nATGP + NNLS Unmixing ({args.endmembers} endmembers) | Mean Spectral RMSE: {mean_rmse:.4f}"
    )

    convolved_centers = None
    convolved_endmembers = None
    sensor_label = None

    if args.convolve is not None:
        srf_specs = SENSOR_SRF_SPECS[args.convolve]
        sensor_label = "Sentinel-2 MSI" if args.convolve == "s2" else "Sentinel-3 OLCI"
        srf_weights, convolved_centers = build_srf_weights(wavelengths, srf_specs)
        convolved_cube = resample_cube(data, srf_weights)
        convolved_endmembers = endmembers @ srf_weights.T

        print(f"\n--- Cross-Sensor Bandpass Simulation ({sensor_label}) ---")
        target_layers = []
        target_names_list = []

        if "chlorophyll_red_edge" in window_results:
            target_layers.append(window_results["chlorophyll_red_edge"]["depth"])
            target_layers.append(window_results["chlorophyll_red_edge"]["reip_nm"])
            target_names_list.extend(["Chl_Band_Depth", "REIP_nm"])

        if "clay_al_oh" in window_results:
            target_layers.append(window_results["clay_al_oh"]["depth"])
            target_names_list.append("Clay_Al_OH_Depth")

        if target_layers:
            target_cube = np.stack(target_layers, axis=-1)
            valid_gt = valid_mask.astype(np.int64)
            _, r2_scores = train_olci_red_edge_estimator(
                convolved_cube,
                target_cube,
                valid_gt,
                target_names=tuple(target_names_list),
            )
            print(
                f"{sensor_label} -> Hyperspectral Parameter Retrieval (Spatial Val R2):"
            )
            for m_name, score in r2_scores.items():
                print(f"  {m_name:<22}: {score:6.3f}")

        if not is_enmap_dir:
            flat_pixels = data.reshape(-1, data.shape[-1])
            re_metrics_flat, _, _ = compute_red_edge_metrics(flat_pixels, wavelengths)
            re_metrics_cube = re_metrics_flat.reshape(data.shape[0], data.shape[1], 3)
            run_benchmark(
                data, convolved_cube, re_metrics_cube, gt, sensor_label=sensor_label
            )

    out_png = (
        Path("outputs/enmap_spectroscopy_summary.png")
        if is_enmap_dir
        else Path("outputs/pavia_spectroscopy_summary.png")
    )
    plot_spectroscopy_summary(
        wavelengths=wavelengths,
        endmembers=endmembers,
        window_results=window_results,
        rmse_map=rmse_map,
        out_path=out_png,
        convolved_centers=convolved_centers,
        convolved_endmembers=convolved_endmembers,
        sensor_name=sensor_label,
    )
    print(f"Summary figure saved to: {out_png}")


if __name__ == "__main__":
    main()
