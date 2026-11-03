import argparse
import math
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import CRSError
from rasterio.warp import reproject, transform

from .enmap import GeoMetadata, load_enmap_l2a

WORLDCOVER_S3_TEMPLATE: str = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile_id}_Map.tif"
)

WORLDCOVER_TO_PIPELINE: dict[int, int] = {
    10: 1,  # Tree cover -> Dense Vegetation
    20: 2,  # Shrubland -> Sparse Vegetation
    30: 2,  # Grassland -> Sparse Vegetation
    40: 1,  # Cropland -> Dense Vegetation
    50: 4,  # Built-up -> Built-up
    60: 3,  # Bare / sparse vegetation -> Bare Soil / Rock
    80: 5,  # Permanent water bodies -> Water
    90: 2,  # Herbaceous wetland -> Sparse Vegetation
    95: 1,  # Mangroves -> Dense Vegetation
    100: 2,  # Moss and lichen -> Sparse Vegetation
}


def parse_crs(crs_str: str) -> CRS:
    """Parses a CRS string, translating legacy ENVI UTM strings if needed."""
    try:
        return CRS.from_user_input(crs_str)
    except (CRSError, ValueError):
        pass

    m = re.search(
        r"UTM\s*(?:ZONE)?\s*(\d+)\s*,?\s*([NS]|NORTH|SOUTH)?",
        crs_str,
        re.IGNORECASE,
    )
    if m:
        zone = int(m.group(1))
        hemi = m.group(2) or "N"
        base = 32700 if hemi.upper().startswith("S") else 32600
        return CRS.from_epsg(base + zone)

    raise ValueError(f"Could not parse spatial reference from: '{crs_str}'")


def _get_tile_id(lat: float, lon: float) -> str:
    tile_lat = int(math.floor(lat / 3.0) * 3)
    tile_lon = int(math.floor(lon / 3.0) * 3)

    lat_str = f"N{tile_lat:02d}" if tile_lat >= 0 else f"S{abs(tile_lat):02d}"
    lon_str = f"E{tile_lon:03d}" if tile_lon >= 0 else f"W{abs(tile_lon):03d}"
    return f"{lat_str}{lon_str}"


def get_intersecting_worldcover_tiles(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> list[str]:
    lat_starts = range(
        int(math.floor(min_lat / 3.0) * 3),
        int(math.floor(max_lat / 3.0) * 3) + 3,
        3,
    )
    lon_starts = range(
        int(math.floor(min_lon / 3.0) * 3),
        int(math.floor(max_lon / 3.0) * 3) + 3,
        3,
    )

    tile_ids = [_get_tile_id(lat, lon) for lat in lat_starts for lon in lon_starts]
    return sorted(set(tile_ids))


def fetch_aligned_worldcover(
    enmap_dir: Path | str,
    out_tif_path: Path | str = "data/aligned_worldcover.tif",
    bbox: tuple[int, int, int, int] | None = None,
) -> Path:
    out_file = Path(out_tif_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    scene = load_enmap_l2a(enmap_dir, bbox=bbox)
    if scene.geo is None:
        raise ValueError(
            "EnMAP scene does not contain spatial georeferencing metadata."
        )

    geo: GeoMetadata = scene.geo
    h, w, _ = scene.cube.shape

    dst_crs = parse_crs(geo.crs)
    dst_transform = rasterio.transform.from_origin(
        geo.origin_x, geo.origin_y, geo.pixel_size_x, geo.pixel_size_y
    )

    x_min, x_max = geo.origin_x, geo.origin_x + w * geo.pixel_size_x
    y_min, y_max = geo.origin_y - h * geo.pixel_size_y, geo.origin_y
    xs = [x_min, x_max, x_min, x_max]
    ys = [y_min, y_min, y_max, y_max]

    lons, lats = transform(dst_crs, CRS.from_epsg(4326), xs, ys)
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    tile_ids = get_intersecting_worldcover_tiles(min_lon, min_lat, max_lon, max_lat)
    cog_urls = [
        f"/vsicurl/{WORLDCOVER_S3_TEMPLATE.format(tile_id=t)}" for t in tile_ids
    ]

    print(f"Target grid: {w}x{h} pixels ({geo.pixel_size_x}m GSD, {dst_crs})")
    print(f"WGS84 extent: [{min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}]")
    print(
        f"Streaming {len(cog_urls)} intersecting WorldCover COG(s): {', '.join(tile_ids)}"
    )

    raw_aligned = np.zeros((h, w), dtype=np.uint8)

    for url in cog_urls:
        try:
            with rasterio.open(url) as src:
                tile_buf = np.zeros((h, w), dtype=np.uint8)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=tile_buf,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.mode,
                )
                mask = tile_buf > 0
                raw_aligned[mask] = tile_buf[mask]
        except rasterio.errors.RasterioError as e:
            print(f"Warning: could not stream {url}: {e}")

    mapped_labels = np.zeros((h, w), dtype=np.int64)
    for wc_val, pipe_val in WORLDCOVER_TO_PIPELINE.items():
        mapped_labels[raw_aligned == wc_val] = pipe_val

    mapped_labels[~scene.valid_mask] = 0

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": rasterio.int64,
        "crs": dst_crs,
        "transform": dst_transform,
        "nodata": 0,
    }

    with rasterio.open(out_file, "w", **profile) as dst:
        dst.write(mapped_labels, 1)

    print(f"Aligned ground truth saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream and reproject ESA WorldCover 10m to an EnMAP scene grid."
    )
    parser.add_argument(
        "enmap_dir",
        type=Path,
        help="Path to EnMAP Level-2A product directory",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        default=None,
        metavar=("R0", "R1", "C0", "C1"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/aligned_worldcover.tif"))
    args = parser.parse_args()

    bbox_tuple = tuple(args.bbox) if args.bbox else None
    fetch_aligned_worldcover(args.enmap_dir, out_tif_path=args.out, bbox=bbox_tuple)
