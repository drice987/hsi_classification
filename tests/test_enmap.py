from pathlib import Path

import numpy as np
import tifffile

from hsi.enmap import (
    ENMAP_NODATA,
    get_atmospheric_window_mask,
    load_enmap_l2a,
    parse_enmap_metadata,
)


def _write_mock_enmap_product(prod_dir: Path) -> None:
    wavelengths = np.linspace(420.0, 2420.0, 40)
    band_nodes = []
    for i, wl in enumerate(wavelengths, start=1):
        band_nodes.append(
            f'<bandID number="{i}">'
            f"<wavelengthCenterOfBand>{wl:.2f}</wavelengthCenterOfBand>"
            f"<FWHMOfBand>8.5</FWHMOfBand>"
            f"<GainOfBand>0.0001</GainOfBand>"
            f"<OffsetOfBand>0.0</OffsetOfBand>"
            f"</bandID>"
        )
    xml_content = (
        f"<level_X><specific><bandCharacterisation>"
        f"{''.join(band_nodes)}"
        f"</bandCharacterisation></specific></level_X>"
    )
    (prod_dir / "ENMAP01-L2A-METADATA.XML").write_text(xml_content, encoding="utf-8")

    h, w, c = 16, 20, 40
    raw_cube = np.full((c, h, w), 2500, dtype=np.int16)

    for b, wl in enumerate(wavelengths):
        if wl < 680.0:
            raw_cube[b, :8, :] = 400
        elif wl < 1300.0:
            raw_cube[b, :8, :] = 4800

    raw_cube[:, 0, 0] = ENMAP_NODATA
    tifffile.imwrite(prod_dir / "ENMAP01-L2A-SPECTRAL_IMAGE.TIF", raw_cube)

    cloud_mask = np.zeros((h, w), dtype=np.uint8)
    cloud_mask[1, 1] = 1
    tifffile.imwrite(prod_dir / "ENMAP01-L2A-QL_QUALITY_CLOUD.TIF", cloud_mask)

    q_classes = np.ones((h, w), dtype=np.uint8)
    q_classes[14:, :] = 2
    q_classes[2, 2] = 3
    tifffile.imwrite(prod_dir / "ENMAP01-L2A-QL_QUALITY_CLASSES.TIF", q_classes)


def test_parse_metadata_and_water_vapor_filter(tmp_path: Path) -> None:
    _write_mock_enmap_product(tmp_path)
    wl, fwhm, gains, offsets = parse_enmap_metadata(
        tmp_path / "ENMAP01-L2A-METADATA.XML"
    )

    assert len(wl) == 40
    assert np.allclose(fwhm, 8.5)
    assert np.allclose(gains, 1e-4)
    assert np.allclose(offsets, 0.0)

    keep = get_atmospheric_window_mask(wl)
    assert keep.dtype == bool
    assert int(np.sum(keep)) < 40


def test_load_enmap_l2a_masking_and_scaling(tmp_path: Path) -> None:
    _write_mock_enmap_product(tmp_path)
    scene = load_enmap_l2a(tmp_path, drop_water_vapor=True, bbox=(0, 16, 0, 15))

    assert scene.cube.shape[:2] == (16, 15)
    assert scene.cube.shape[2] == len(scene.wavelengths)

    assert not scene.valid_mask[0, 0]
    assert not scene.valid_mask[1, 1]
    assert not scene.valid_mask[2, 2]
    assert np.all(scene.cube[0, 0] == 0.0)
    assert np.all(scene.labels[~scene.valid_mask] == 0)

    assert scene.labels[4, 4] == 1
    assert scene.labels[15, 4] == 5
