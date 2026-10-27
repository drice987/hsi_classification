import numpy as np
from hsi.spectral_analysis import (
    SENTINEL2_MSI_BANDS,
    analyze_diagnostic_windows,
    build_srf_weights,
    compute_absorption_feature,
    compute_red_edge_metrics,
    extract_endmembers_atgp,
    get_wavelength_axis,
    resample_cube,
    unmix_nnls,
    unmix_scene_atgp,
)


def test_srf_resampling_olci_and_s2() -> None:
    wl_vnir = get_wavelength_axis(103, (430.0, 860.0))
    srf_olci, centers_olci = build_srf_weights(wl_vnir)

    assert srf_olci.shape == (15, 103)
    assert len(centers_olci) == 15
    np.testing.assert_allclose(srf_olci.sum(axis=1), 1.0, atol=1e-5)

    cube = np.ones((10, 12, 103), dtype=np.float32) * 0.42
    olci_cube = resample_cube(cube, srf_olci)
    assert olci_cube.shape == (10, 12, 15)
    np.testing.assert_allclose(olci_cube, 0.42, atol=1e-5)

    wl_full = get_wavelength_axis(180, (420.0, 2400.0))
    srf_s2, centers_s2 = build_srf_weights(wl_full, SENTINEL2_MSI_BANDS)
    assert srf_s2.shape == (12, 180)
    assert len(centers_s2) == 12


def test_compute_red_edge_and_clay_al_oh_metrics() -> None:
    wl_full = get_wavelength_axis(200, (420.0, 2400.0))

    red_edge = 0.05 + 0.35 / (1.0 + np.exp(-(wl_full - 715.0) / 8.0))
    clay_dip = 1.0 - 0.30 * np.exp(-0.5 * ((wl_full - 2205.0) / 12.0) ** 2)
    spectrum = (red_edge * clay_dip).astype(np.float32)

    re_metrics, wl_re, cr_re = compute_red_edge_metrics(
        spectrum[np.newaxis, :], wl_full
    )
    assert re_metrics.shape == (1, 3)
    assert cr_re.shape == (1, len(wl_re))
    assert 700.0 <= re_metrics[0, 2] <= 730.0

    clay_metrics, _, _ = compute_absorption_feature(
        spectrum[np.newaxis, :], wl_full, (2160.0, 2240.0)
    )
    assert clay_metrics[0, 0] > 0.20
    assert 2195.0 <= clay_metrics[0, 2] <= 2215.0

    cube = np.tile(spectrum, (6, 8, 1))
    win_maps = analyze_diagnostic_windows(cube, wl_full)
    assert "chlorophyll_red_edge" in win_maps
    assert "clay_al_oh" in win_maps
    assert win_maps["clay_al_oh"]["depth"].shape == (6, 8)


def test_subband_parabolic_wavelength_shift() -> None:
    wl_swir = np.arange(2100.0, 2300.0, 10.0, dtype=np.float32)

    spec_illite = 0.35 * (1.0 - 0.25 * np.exp(-0.5 * ((wl_swir - 2198.0) / 14.0) ** 2))
    spec_kaolinite = 0.35 * (
        1.0 - 0.25 * np.exp(-0.5 * ((wl_swir - 2206.0) / 14.0) ** 2)
    )
    spectra = np.vstack([spec_illite, spec_kaolinite]).astype(np.float32)

    metrics, _, _ = compute_absorption_feature(spectra, wl_swir, (2160.0, 2240.0))
    assert metrics[1, 2] - metrics[0, 2] > 5.0
    assert np.isclose(metrics[0, 2], 2198.0, atol=1.5)
    assert np.isclose(metrics[1, 2], 2206.0, atol=1.5)


def test_atgp_and_nnls_unmixing() -> None:
    rng = np.random.default_rng(42)
    cube = rng.uniform(0.05, 0.60, size=(8, 10, 50)).astype(np.float32)
    valid_mask = np.ones((8, 10), dtype=bool)
    valid_mask[0, 0] = False
    cube[0, 0, :] = 0.0

    endmembers = extract_endmembers_atgp(cube, n_endmembers=3, valid_mask=valid_mask)
    assert endmembers.shape == (3, 50)

    abund_cube, _, rmse_map = unmix_scene_atgp(
        cube, n_endmembers=3, valid_mask=valid_mask
    )
    assert abund_cube.shape == (8, 10, 3)
    assert rmse_map.shape == (8, 10)
    assert rmse_map[0, 0] == 0.0
    np.testing.assert_allclose(abund_cube[valid_mask].sum(axis=-1), 1.0, atol=1e-5)

    gt = np.zeros((8, 10), dtype=np.int64)
    gt[1:3, 1:3] = 1
    gt[3:5, 3:5] = 2
    maps, basis = unmix_nnls(cube, gt)
    assert basis.shape[1] == 50
    assert len(maps) == 2
