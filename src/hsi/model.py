from pathlib import Path

import torch
from torch import nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SpatialSpectralCNN(nn.Module):
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
    """Spectral-spatial bottleneck autoencoder with a learnable spatial mask token."""

    def __init__(
        self,
        in_channels: int = 103,
        bottleneck_dim: int = 64,
        mask_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim
        self.mask_ratio = mask_ratio

        self.mask_token = nn.Parameter(torch.empty(1, in_channels, 1, 1))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

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

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
        spatial_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio

        if spatial_mask is not None:
            x_masked = torch.where(spatial_mask, self.mask_token.expand_as(x), x)
        elif self.training and ratio > 0.0:
            b, _, h, w = x.shape
            rand_mask = torch.rand(b, 1, h, w, device=x.device) < ratio
            x_masked = torch.where(rand_mask, self.mask_token.expand_as(x), x)
        else:
            x_masked = x

        latent = self.encoder(x_masked)
        return self.decoder(latent)


class SSLClassifier(nn.Module):
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        num_classes: int,
        hidden_dim: int = 128,
        use_center_pixel: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = pretrained_encoder
        self.use_center_pixel = use_center_pixel
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        if self.use_center_pixel:
            c_h = feat.shape[-2] // 2
            c_w = feat.shape[-1] // 2
            pooled = feat[:, :, c_h, c_w]
        else:
            pooled = torch.flatten(self.pool(feat), 1)

        return self.classifier(pooled)


def load_pretrained_mae(
    mae: MaskedConvolutionalAutoencoder,
    weights_path: Path | str,
    device: torch.device,
) -> MaskedConvolutionalAutoencoder:
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if any(k.startswith("encoder.") for k in state_dict):
        mae.load_state_dict(state_dict, strict=False)
    else:
        mae.encoder.load_state_dict(state_dict)
    return mae
