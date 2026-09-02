import torch
import torch.nn as nn

class SpatialSpectralCNN(nn.Module):
    def __init__(self, in_channels: int = 103, num_classes: int = 9):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SpatialSpectralCNN(in_channels=103, num_classes=9)
    dummy_batch = torch.randn(4, 103, 9, 9)
    
    output = model(dummy_batch)
    
    assert output.shape == (4, 9), f"Expected shape (4, 9), got {output.shape}"
    assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"
    assert not torch.isnan(output).any(), "NaN values detected in forward pass"
    
    print("Pass")