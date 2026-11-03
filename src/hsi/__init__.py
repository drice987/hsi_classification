from importlib.metadata import PackageNotFoundError, version

from .data_loader import (
    CLASS_NAMES,
    SceneData,
    load_pavia_data,
    load_scene,
    normalize_data,
    split_spatial_gt,
)
from .dataset import (
    PaviaInferenceDataset,
    PaviaPatchDataset,
    PaviaSSLDataset,
    PixelInferenceDataset,
    RandomSpatialPatchDataset,
    SpatialPatchDataset,
    SSLNeighborhoodDataset,
)
from .enmap import (
    ENMAP_SURFACE_CLASSES,
    EnMAPScene,
    GeoMetadata,
    export_geotiff,
    load_enmap_l2a,
)
from .model import (
    MaskedConvolutionalAutoencoder,
    SpatialSpectralCNN,
    SSLClassifier,
    get_device,
)

try:
    __version__ = version("hsi")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "CLASS_NAMES",
    "ENMAP_SURFACE_CLASSES",
    "EnMAPScene",
    "GeoMetadata",
    "MaskedConvolutionalAutoencoder",
    "PaviaInferenceDataset",
    "PaviaPatchDataset",
    "PaviaSSLDataset",
    "PixelInferenceDataset",
    "RandomSpatialPatchDataset",
    "SSLClassifier",
    "SSLNeighborhoodDataset",
    "SceneData",
    "SpatialPatchDataset",
    "SpatialSpectralCNN",
    "__version__",
    "export_geotiff",
    "get_device",
    "load_enmap_l2a",
    "load_pavia_data",
    "load_scene",
    "normalize_data",
    "split_spatial_gt",
]
