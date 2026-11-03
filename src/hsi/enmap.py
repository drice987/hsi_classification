import json
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
    "Built-up",
    "Water",
]

GEOTIFF_PIXEL_SCALE_TAG: int = 33550
GEOTIFF_TIEPOINT_TAG: int = 33922
GEOTIFF_GEOKEY_DIR_TAG: int = 34735


@dataclass(frozen=True)
class GeoMetadata:
    crs: str = "LOCAL"
    origin_x: float = 0.0
    origin_y: float = 0.0
    pixel_size_x: float = 30.0
    pixel_size_y: float = 30.0
    geokeys: tuple[int, ...] | None = None

    def crop(self, row_offset: int, col_offset: int) -> "GeoMetadata":
        return GeoMetadata(
            crs=self.crs,
            origin_x=self.origin_x + col_offset * self.pixel_size_x,
            origin_y=self.origin_y - row_offset * self.pixel_size_y,
            pixel_size_x=self.pixel_size_x,
            pixel_size_y=self.pixel_size_y,
            geokeys=self.geokeys,
        )


@dataclass
class EnMAPScene:
    cube: np.ndarray
    wavelengths: np.ndarray
    fwhm: np.ndarray
    valid_mask: np.ndarray
    quality_classes: np.ndarray
    labels: np.ndarray
    geo: GeoMetadata | None = None


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


def _parse_geo_from_envi_hdr(hdr: dict[str, str]) -> GeoMetadata | None:
    map_info = hdr.get("map info")
    if not map_info:
        return None
    parts = [p.strip() for p in map_info.split(",")]
    if len(parts) < 7:
        return None
    try:
        proj_name = parts[0].upper()
        ref_x = float(parts[1]) - 1.0
        ref_y = float(parts[2]) - 1.0
        tie_x = float(parts[3])
        tie_y = float(parts[4])
        dx = float(parts[5])
        dy = float(parts[6])

        crs_label = "EPSG:4326"
        if "UTM" in proj_name and len(parts) > 7:
            try:
                zone = int(parts[7].strip())
                hemisphere = parts[8].strip().upper() if len(parts) > 8 else "NORTH"
                base = 32700 if hemisphere.startswith("S") else 32600
                crs_label = f"EPSG:{base + zone}"
            except ValueError:
                crs_label = "EPSG:32632"
        elif proj_name in ("GEOGRAPHIC", "GEOGRAPHIC LAT/LON", "LAT/LON"):
            crs_label = "EPSG:4326"
        else:
            crs_label = proj_name

        return GeoMetadata(
            crs=crs_label,
            origin_x=tie_x - ref_x * dx,
            origin_y=tie_y + ref_y * dy,
            pixel_size_x=dx,
            pixel_size_y=dy,
        )
    except ValueError:
        return None


def _parse_geo_from_tiff(tif_path: Path) -> GeoMetadata | None:
    with tifffile.TiffFile(tif_path) as tif:
        tags = tif.pages[0].tags
        scale_tag = tags.get(GEOTIFF_PIXEL_SCALE_TAG)
        tie_tag = tags.get(GEOTIFF_TIEPOINT_TAG)
        geokey_tag = tags.get(GEOTIFF_GEOKEY_DIR_TAG)

        if scale_tag is None and tie_tag is None and geokey_tag is None:
            return None

        dx, dy = 30.0, 30.0
        if scale_tag is not None and len(scale_tag.value) >= 2:
            dx = float(scale_tag.value[0])
            dy = float(scale_tag.value[1])

        ox, oy = 0.0, 0.0
        if tie_tag is not None and len(tie_tag.value) >= 6:
            i_px, j_px = float(tie_tag.value[0]), float(tie_tag.value[1])
            ox = float(tie_tag.value[3]) - i_px * dx
            oy = float(tie_tag.value[4]) + j_px * dy

        geokeys: tuple[int, ...] | None = None
        crs = "GEOTIFF"
        if geokey_tag is not None:
            geokeys = tuple(int(v) for v in geokey_tag.value)
            for idx in range(4, len(geokeys) - 3, 4):
                if geokeys[idx] in (3072, 2048) and geokeys[idx + 1] == 0:
                    crs = f"EPSG:{geokeys[idx + 3]}"
                    break

        return GeoMetadata(
            crs=crs,
            origin_x=ox,
            origin_y=oy,
            pixel_size_x=dx,
            pixel_size_y=dy,
            geokeys=geokeys,
        )


def export_geotiff(
    out_path: Path | str,
    data: np.ndarray,
    geo: GeoMetadata | None = None,
    band_names: list[str] | None = None,
    nodata: float = 0.0,
) -> Path:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))

    extratags: list[tuple[int, str, int, tuple[float | int, ...], bool]] = []
    meta_dict: dict[str, object] = {"nodata": nodata}
    if band_names is not None:
        meta_dict["band_names"] = band_names

    if geo is not None:
        meta_dict["crs"] = geo.crs
        scale_vals = (float(geo.pixel_size_x), float(geo.pixel_size_y), 0.0)
        tie_vals = (0.0, 0.0, 0.0, float(geo.origin_x), float(geo.origin_y), 0.0)
        extratags.append((GEOTIFF_PIXEL_SCALE_TAG, "d", 3, scale_vals, True))
        extratags.append((GEOTIFF_TIEPOINT_TAG, "d", 6, tie_vals, True))
        if geo.geokeys is not None:
            extratags.append(
                (GEOTIFF_GEOKEY_DIR_TAG, "H", len(geo.geokeys), geo.geokeys, True)
            )

    tifffile.imwrite(
        target,
        arr,
        photometric="minisblack",
        description=json.dumps(meta_dict),
        extratags=extratags,
    )
    return target


def _local_tag(element: ET.Element) -> str:
    return element.tag.split("}")[-1].lower()


def parse_enmap_metadata(
    xml_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts band center wavelengths, FWHM, gains, and offsets from EnMAP METADATA.XML."""
    root = ET.parse(Path(xml_path)).getroot()

    wavelengths: list[float] = []
    fwhms: list[float] = []
    gains: list[float] = []
    offsets: list[float] = []

    for parent in root.iter():
        if _local_tag(parent) == "bandcharacterisation":
            for node in parent:
                if _local_tag(node) not in ("bandid", "band"):
                    continue

                band_data: dict[str, float] = {}
                for child in node:
                    tag = _local_tag(child)
                    text = (child.text or "").strip()
                    if text:
                        try:
                            band_data[tag] = float(text)
                        except ValueError:
                            pass

                wl = band_data.get("wavelengthcenterofband") or band_data.get(
                    "wavelength"
                )
                if wl is not None:
                    wavelengths.append(wl)
                    fwhms.append(
                        band_data.get("fwhmofband", band_data.get("fwhm", 8.5))
                    )
                    gains.append(
                        band_data.get("gainofband", band_data.get("gain", 1e-4))
                    )
                    offsets.append(
                        band_data.get("offsetofband", band_data.get("offset", 0.0))
                    )
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

    is_dense_veg = (~is_water) & (ndvi >= 0.35)
    is_sparse_veg = (~is_water) & (ndvi >= 0.18) & (ndvi < 0.35)
    is_soil = (
        (~is_water) & (ndvi < 0.18) & ((swir_nir_ratio >= 0.80) | (mean_refl >= 0.15))
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
    geo_meta: GeoMetadata | None = None

    if bsq_path is not None and hdr_path is not None:
        hdr = _parse_envi_header(hdr_path)
        geo_meta = _parse_geo_from_envi_hdr(hdr)
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

        r0, r1, c0, c1 = 0, lines, 0, samples
        if bbox is not None:
            r0, r1 = max(0, bbox[0]), min(lines, bbox[1])
            c0, c1 = max(0, bbox[2]), min(samples, bbox[3])
            if r0 >= r1 or c0 >= c1:
                raise ValueError(
                    f"Bounding box {bbox} is out of bounds for shape ({lines}, {samples})."
                )
            bbox = (r0, r1, c0, c1)
            if geo_meta is not None:
                geo_meta = geo_meta.crop(r0, c0)

        if interleave == "bil":
            raw_img = np.transpose(np.array(mmap[r0:r1, :, c0:c1]), (0, 2, 1))
        elif interleave == "bip":
            raw_img = np.array(mmap[r0:r1, c0:c1, :])
        else:
            raw_img = np.transpose(np.array(mmap[:, r0:r1, c0:c1]), (1, 2, 0))

    elif tif_path is not None:
        geo_meta = _parse_geo_from_tiff(tif_path)
        raw_img = tifffile.imread(tif_path)
        if raw_img.ndim != 3:
            raise ValueError(f"Expected 3D spectral cube, got shape {raw_img.shape}")

        is_band_first = (
            wavelengths is not None and raw_img.shape[0] == len(wavelengths)
        ) or (
            raw_img.shape[0] < raw_img.shape[1] and raw_img.shape[0] < raw_img.shape[2]
        )
        if is_band_first:
            raw_img = np.transpose(raw_img, (1, 2, 0))

        if bbox is not None:
            r0, r1 = max(0, bbox[0]), min(raw_img.shape[0], bbox[1])
            c0, c1 = max(0, bbox[2]), min(raw_img.shape[1], bbox[3])
            bbox = (r0, r1, c0, c1)
            if geo_meta is not None:
                geo_meta = geo_meta.crop(r0, c0)
            raw_img = raw_img[r0:r1, c0:c1, :]
    else:
        found = [p.name for p in prod_path.iterdir() if p.is_file()]
        raise FileNotFoundError(
            f"Could not locate EnMAP spectral image in {prod_path}. Found: {found}"
        )

    n_bands = raw_img.shape[-1]
    if wavelengths is None or len(wavelengths) != n_bands:
        raise ValueError(
            f"Metadata band count ({len(wavelengths) if wavelengths is not None else 0}) "
            f"does not match image cube bands ({n_bands}). Check METADATA.XML parsing."
        )

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
        ext_gt = np.squeeze(tifffile.imread(Path(gt_tif_path)))

        if ext_gt.shape != valid_mask.shape:
            if bbox is not None:
                r0, r1, c0, c1 = bbox
                ext_gt = ext_gt[r0:r1, c0:c1]

        if ext_gt.shape != valid_mask.shape:
            raise ValueError(
                f"External ground truth shape {ext_gt.shape} does not match "
                f"scene shape {valid_mask.shape}."
            )

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
        geo=geo_meta,
    )
