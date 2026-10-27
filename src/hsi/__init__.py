from importlib.metadata import PackageNotFoundError, version

from .data_loader import (
    CLASS_NAMES,
    load_pavia_data,
    load_scene,
    normalize_data,
    split_spatial_gt,
)
from .dataset import PaviaInferenceDataset, PaviaPatchDataset, PaviaSSLDataset
from .enmap import ENMAP_SURFACE_CLASSES, EnMAPScene, load_enmap_l2a
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
    "MaskedConvolutionalAutoencoder",
    "PaviaInferenceDataset",
    "PaviaPatchDataset",
    "PaviaSSLDataset",
    "SSLClassifier",
    "SpatialSpectralCNN",
    "__version__",
    "get_device",
    "load_enmap_l2a",
    "load_pavia_data",
    "load_scene",
    "normalize_data",
    "split_spatial_gt",
]
