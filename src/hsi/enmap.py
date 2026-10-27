import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

ENMAP_NODATA: int = -32768
WATER_VAPOR_WINDOWS_NM: tuple[tuple[float, float], ...] = (
    (1350.0, 1460.0),
    (1790.0, 1960.0),
)

ENMAP_SURFACE_CLASSES: list[str] = [
    "Dense Vegetation",
    "Sparse Vegetation",
    "Bare Soil / Rock",
    "Impervious / Low Albedo",
    "Water",
]


@dataclass
class EnMAPScene:
    cube: np.ndarray
    wavelengths: np.ndarray
    fwhm: np.ndarray
    valid_mask: np.ndarray
    quality_classes: np.ndarray
    labels: np.ndarray


def _resolve_product_dir(path: Path) -> Path:
    if list(path.glob("*METADATA.XML*")) or list(path.glob("*SPECTRAL_IMAGE*")):
        return path
    for child in path.iterdir():
        if child.is_dir() and (
            list(child.glob("*METADATA.XML*")) or list(child.glob("*SPECTRAL_IMAGE*"))
        ):
            return child
    return path


def _find_product_file(product_dir: Path, suffix: str) -> Path | None:
    matches = sorted(product_dir.glob(f"*{suffix.upper()}")) or sorted(
        product_dir.glob(f"*{suffix.lower()}")
    )
    return matches[0] if matches else None


def _parse_envi_header(hdr_path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    with open(hdr_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "=" in line and not line.lstrip().startswith(";"):
                k, v = line.split("=", 1)
                meta[k.strip().lower()] = v.strip().strip("{}").strip().lower()
    return meta


def parse_enmap_metadata(
    xml_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts center wavelengths, FWHM, gain, and offset arrays from METADATA.XML."""
    tree = ET.parse(Path(xml_path))
    root = tree.getroot()

    wavelengths: list[float] = []
    fwhms: list[float] = []
    gains: list[float] = []
    offsets: list[float] = []

    for parent in root.iter():
        if parent.tag.split("}")[-1].lower() == "bandcharacterisation":
            for band_node in parent:
                if band_node.tag.split("}")[-1].lower() not in ("bandid", "band"):
                    continue

                wl_val: float | None = None
                fwhm_val: float = 8.0
                gain_val: float = 1e-4
                offset_val: float = 0.0

                for child in band_node:
                    ctag = child.tag.split("}")[-1].lower()
                    text = child.text.strip() if child.text else ""
                    if not text:
                        continue
                    try:
                        if ctag in ("wavelengthcenterofband", "wavelength"):
                            wl_val = float(text)
                        elif ctag in ("fwhmofband", "fwhm"):
                            fwhm_val = float(text)
                        elif ctag in ("gainofband", "gain"):
                            gain_val = float(text)
                        elif ctag in ("offsetofband", "offset"):
                            offset_val = float(text)
                    except ValueError:
                        continue

                if wl_val is not None:
                    wavelengths.append(wl_val)
                    fwhms.append(fwhm_val)
                    gains.append(gain_val)
                    offsets.append(offset_val)
            break

    if not wavelengths:
        raise ValueError(f"No band characterisation nodes found in {xml_path}")

    return (
        np.array(wavelengths, dtype=np.float32),
        np.array(fwhms, dtype=np.float32),
        np.array(gains, dtype=np.float32),
        np.array(offsets, dtype=np.float32),
    )


def get_atmospheric_window_mask(
    wavelengths: np.ndarray,
    exclude_windows: tuple[tuple[float, float], ...] = WATER_VAPOR_WINDOWS_NM,
    bounds_nm: tuple[float, float] = (420.0, 2400.0),
    splice_nm: float = 950.0,
) -> np.ndarray:
    """Filters out deep water-vapor bands and splices the VNIR/SWIR detector overlap."""
    keep = (wavelengths >= bounds_nm[0]) & (wavelengths <= bounds_nm[1])
    for low_nm, high_nm in exclude_windows:
        keep &= ~((wavelengths >= low_nm) & (wavelengths <= high_nm))

    diffs = np.diff(wavelengths)
    if np.any(diffs < 0):
        split_idx = int(np.argmin(diffs)) + 1
        idx = np.arange(len(wavelengths))
        keep &= ((idx < split_idx) & (wavelengths <= splice_nm)) | (
            (idx >= split_idx) & (wavelengths > splice_nm)
        )
    return keep


def _read_single_band_tif(
    product_dir: Path, suffix_pattern: str, bbox: tuple[int, int, int, int] | None
) -> np.ndarray | None:
    tif_path = _find_product_file(product_dir, suffix_pattern)
    if tif_path is None:
        return None

    arr = np.squeeze(tifffile.imread(tif_path))

    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = arr[0]
        else:
            arr = arr[:, :, 0]

    if bbox is not None:
        r0, r1, c0, c1 = bbox
        arr = arr[r0:r1, c0:c1]

    return arr


def derive_radiometric_labels(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    valid_mask: np.ndarray,
    quality_classes: np.ndarray | None = None,
) -> np.ndarray:
    """Assigns coarse surface classes (1-5) when no external ground-truth raster is given."""
    h, w, _ = cube.shape
    labels = np.zeros((h, w), dtype=np.int64)

    idx_green = int(np.argmin(np.abs(wavelengths - 560.0)))
    idx_red = int(np.argmin(np.abs(wavelengths - 665.0)))
    idx_nir = int(np.argmin(np.abs(wavelengths - 865.0)))
    idx_swir = int(np.argmin(np.abs(wavelengths - 1650.0)))

    green = cube[:, :, idx_green]
    red = cube[:, :, idx_red]
    nir = cube[:, :, idx_nir]
    swir = cube[:, :, idx_swir]

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-7)
        ndwi = (green - nir) / (green + nir + 1e-7)
        swir_nir_ratio = swir / (nir + 1e-7)

    mean_refl = np.nanmean(cube, axis=-1)

    is_water = ndwi > 0.15
    if quality_classes is not None:
        is_water |= quality_classes == 2

    is_dense_veg = (~is_water) & (ndvi >= 0.50)
    is_sparse_veg = (~is_water) & (ndvi >= 0.25) & (ndvi < 0.50)
    is_soil = (
        (~is_water) & (ndvi < 0.25) & ((swir_nir_ratio >= 0.85) | (mean_refl >= 0.18))
    )
    is_impervious = (~is_water) & (ndvi < 0.25) & (~is_soil)

    labels[valid_mask & is_dense_veg] = 1
    labels[valid_mask & is_sparse_veg] = 2
    labels[valid_mask & is_soil] = 3
    labels[valid_mask & is_impervious] = 4
    labels[valid_mask & is_water] = 5

    return labels


def load_enmap_l2a(
    product_dir: Path | str,
    gt_tif_path: Path | str | None = None,
    drop_water_vapor: bool = True,
    bbox: tuple[int, int, int, int] | None = None,
) -> EnMAPScene:
    """Loads an EnMAP Level-2A product directory into a calibrated reflectance cube."""
    prod_path = _resolve_product_dir(Path(product_dir))
    if not prod_path.is_dir():
        raise FileNotFoundError(f"EnMAP product directory not found: {prod_path}")

    xml_path = _find_product_file(prod_path, "METADATA.XML")
    if xml_path is not None:
        wavelengths, fwhm, gains, offsets = parse_enmap_metadata(xml_path)
    else:
        wavelengths, fwhm, gains, offsets = None, None, None, None

    bsq_path = _find_product_file(prod_path, "SPECTRAL_IMAGE.BSQ")
    hdr_path = _find_product_file(prod_path, "SPECTRAL_IMAGE.HDR")
    tif_path = _find_product_file(prod_path, "SPECTRAL_IMAGE.TIF")

    if bsq_path is not None and hdr_path is not None:
        hdr = _parse_envi_header(hdr_path)
        lines = int(hdr.get("lines", 1000))
        samples = int(hdr.get("samples", 1000))
        bands = int(hdr.get("bands", 224))
        offset = int(hdr.get("header offset", 0))
        dtype_code = int(hdr.get("data type", 2))
        interleave = hdr.get("interleave", "bsq")
        endian = ">" if hdr.get("byte order", "0") == "1" else "<"

        type_map = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 12: "u2"}
        dtype_str = f"{endian}{type_map.get(dtype_code, 'i2')}"

        if interleave == "bil":
            mmap_shape = (lines, bands, samples)
        elif interleave == "bip":
            mmap_shape = (lines, samples, bands)
        else:
            mmap_shape = (bands, lines, samples)

        mmap = np.memmap(
            bsq_path, dtype=dtype_str, mode="r", offset=offset, shape=mmap_shape
        )

        if bbox is not None:
            r0, r1 = max(0, bbox[0]), min(lines, bbox[1])
            c0, c1 = max(0, bbox[2]), min(samples, bbox[3])
            if r0 >= r1 or c0 >= c1:
                raise ValueError(
                    f"Bounding box {bbox} is out of bounds for shape ({lines}, {samples})."
                )
            bbox = (r0, r1, c0, c1)
            if interleave == "bil":
                raw_img = np.transpose(np.array(mmap[r0:r1, :, c0:c1]), (0, 2, 1))
            elif interleave == "bip":
                raw_img = np.array(mmap[r0:r1, c0:c1, :])
            else:
                raw_img = np.transpose(np.array(mmap[:, r0:r1, c0:c1]), (1, 2, 0))
        else:
            if interleave == "bil":
                raw_img = np.transpose(np.array(mmap), (0, 2, 1))
            elif interleave == "bip":
                raw_img = np.array(mmap)
            else:
                raw_img = np.transpose(np.array(mmap), (1, 2, 0))

    elif tif_path is not None:
        raw_img = tifffile.imread(tif_path)
        if raw_img.ndim != 3:
            raise ValueError(f"Expected 3D spectral cube, got shape {raw_img.shape}")
        if (
            wavelengths is not None
            and raw_img.shape[0] == len(wavelengths)
            or raw_img.shape[0] < raw_img.shape[1]
            and raw_img.shape[0] < raw_img.shape[2]
        ):
            raw_img = np.transpose(raw_img, (1, 2, 0))

        if bbox is not None:
            r0, r1 = max(0, bbox[0]), min(raw_img.shape[0], bbox[1])
            c0, c1 = max(0, bbox[2]), min(raw_img.shape[1], bbox[3])
            bbox = (r0, r1, c0, c1)
            raw_img = raw_img[r0:r1, c0:c1, :]
    else:
        found = [p.name for p in prod_path.iterdir() if p.is_file()]
        raise FileNotFoundError(
            f"Could not locate EnMAP spectral image in {prod_path}. Found: {found}"
        )

    n_bands = raw_img.shape[-1]
    if wavelengths is None or len(wavelengths) != n_bands:
        wavelengths = np.linspace(420.0, 2450.0, n_bands, dtype=np.float32)
        fwhm = np.full(n_bands, 8.5, dtype=np.float32)
        gains = np.full(n_bands, 1e-4, dtype=np.float32)
        offsets = np.zeros(n_bands, dtype=np.float32)

    if drop_water_vapor:
        good_bands = get_atmospheric_window_mask(wavelengths)
        non_empty_bands = np.any(raw_img != ENMAP_NODATA, axis=(0, 1))
        if np.any(non_empty_bands):
            good_bands &= non_empty_bands

        raw_img = raw_img[:, :, good_bands]
        wavelengths = wavelengths[good_bands]
        fwhm = fwhm[good_bands]
        gains = gains[good_bands]
        offsets = offsets[good_bands]

    vnir_cols = wavelengths <= 900.0
    if np.any(vnir_cols):
        valid_mask = np.mean(raw_img[:, :, vnir_cols] != ENMAP_NODATA, axis=-1) >= 0.90
    else:
        valid_mask = np.mean(raw_img != ENMAP_NODATA, axis=-1) >= 0.90

    cube = (
        np.where(raw_img == ENMAP_NODATA, 0, raw_img).astype(np.float32)
        * gains[np.newaxis, np.newaxis, :]
        + offsets[np.newaxis, np.newaxis, :]
    )
    cube = np.where(raw_img == ENMAP_NODATA, 0.0, cube)
    cube = np.clip(cube, 0.0, 1.5)

    quality_classes = _read_single_band_tif(prod_path, "QL_QUALITY_CLASSES.TIF", bbox)
    if quality_classes is not None:
        valid_mask &= quality_classes != 3
    else:
        quality_classes = np.ones(valid_mask.shape, dtype=np.uint8)

    for mask_suffix in (
        "QL_QUALITY_CLOUD.TIF",
        "QL_QUALITY_CLOUDSHADOW.TIF",
        "QL_QUALITY_HAZE.TIF",
        "QL_QUALITY_SNOW.TIF",
    ):
        q_layer = _read_single_band_tif(prod_path, mask_suffix, bbox)
        if q_layer is not None:
            valid_mask &= q_layer == 0

    cirrus = _read_single_band_tif(prod_path, "QL_QUALITY_CIRRUS.TIF", bbox)
    if cirrus is not None:
        valid_mask &= cirrus <= 1

    cube[~valid_mask] = 0.0

    if gt_tif_path is not None:
        ext_gt = tifffile.imread(Path(gt_tif_path))
        if bbox is not None:
            r0, r1, c0, c1 = bbox
            ext_gt = ext_gt[r0:r1, c0:c1]
        labels = np.where(valid_mask, ext_gt.astype(np.int64), 0)
    else:
        labels = derive_radiometric_labels(
            cube, wavelengths, valid_mask, quality_classes
        )

    return EnMAPScene(
        cube=cube,
        wavelengths=wavelengths,
        fwhm=fwhm,
        valid_mask=valid_mask,
        quality_classes=quality_classes,
        labels=labels,
    )
