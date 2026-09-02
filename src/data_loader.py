import scipy.io as sio
import numpy as np
from pathlib import Path

def load_pavia_data(
    data_path: Path | str = "data/PaviaU.mat", 
    gt_path: Path | str = "data/PaviaU_gt.mat"
) -> tuple[np.ndarray, np.ndarray]:
    """Loads raw Pavia University data cube and ground truth labels."""
    data = sio.loadmat(data_path)['paviaU']
    gt = sio.loadmat(gt_path)['paviaU_gt']
    
    return data.astype(np.float32), gt.astype(np.int64)