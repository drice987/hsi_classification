from importlib.metadata import PackageNotFoundError, version

from .data_loader import CLASS_NAMES, load_pavia_data, normalize_data, split_spatial_gt
from .dataset import PaviaPatchDataset
from .model import SpatialSpectralCNN, get_device

try:
    __version__ = version("hsi")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "CLASS_NAMES",
    "PaviaPatchDataset",
    "SpatialSpectralCNN",
    "__version__",
    "get_device",
    "load_pavia_data",
    "normalize_data",
    "split_spatial_gt",
]