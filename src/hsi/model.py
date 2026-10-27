from pathlib import Path

import torch
from torch import nn


def get_device() -> torch.device:
    """Selects CUDA, Apple Silicon MPS, or CPU execution device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SpatialSpectralCNN(nn.Module):
    """2D convolutional network for spatial-spectral patch classification."""

    def __init__(self, in_channels: int = 103, num_classes: int = 9) -> None:
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

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MaskedConvolutionalAutoencoder(nn.Module):
    """Spectral-spatial bottleneck autoencoder for self-supervised reconstruction."""

    def __init__(self, in_channels: int = 103, bottleneck_dim: int = 64) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, bottleneck_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


class SSLClassifier(nn.Module):
    """Classification head coupled to a pretrained convolutional encoder."""

    def __init__(self, pretrained_encoder: nn.Module, num_classes: int = 9) -> None:
        super().__init__()
        self.encoder = pretrained_encoder
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_pretrained_mae(
    mae: MaskedConvolutionalAutoencoder,
    weights_path: Path | str,
    device: torch.device,
) -> MaskedConvolutionalAutoencoder:
    """Loads either a full MAE state dict or an encoder-only state dict."""
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("encoder.") for k in state_dict):
        mae.load_state_dict(state_dict, strict=False)
    else:
        mae.encoder.load_state_dict(state_dict)
    return mae
