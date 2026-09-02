import torch
from torch.utils.data import Dataset
import numpy as np

class PaviaPatchDataset(Dataset):
    def __init__(self, data: np.ndarray, gt: np.ndarray, patch_size: int = 9):

        self.patch_size = patch_size
        pad = patch_size // 2

        self.padded_data = np.pad(
            data,
            pad_width=((pad, pad), (pad, pad), (0, 0)),
            mode='reflect'
        )

        y_coords, x_coords = np.where(gt > 0)
        self.coordinates = list(zip(y_coords, x_coords))
        self.labels = gt[y_coords, x_coords] - 1  

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        y, x = self.coordinates[idx]
        pad = self.patch_size // 2
        
        y_pad, x_pad = y + pad, x + pad

        patch = self.padded_data[
            y_pad - pad : y_pad + pad + 1,
            x_pad - pad : x_pad + pad + 1,
            :
        ]

        patch = np.transpose(patch, (2, 0, 1))

        return (
            torch.tensor(patch, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


if __name__ == "__main__":
    from data_loader import load_pavia_data
    
    data, gt = load_pavia_data()
    dataset = PaviaPatchDataset(data, gt, patch_size=9)
    
    patch, label = dataset[0]
    
    assert patch.shape == (103, 9, 9), f"Wrong shape: {patch.shape}"
    assert patch.dtype == torch.float32, f"Wrong patch dtype: {patch.dtype}"
    assert label.dtype == torch.long, f"Wrong label dtype: {label.dtype}"
    assert 0 <= label.item() <= 8, f"Label out of bounds: {label.item()}"
    
    print(f"Passed. Total samples: {len(dataset)}")