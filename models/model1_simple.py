from __future__ import annotations

from typing import List

import torch
from torch import nn


def _feature_stage(
    in_channels: int,
    out_channels: int,
    spatial_dropout: float = 0.0,
) -> List[nn.Module]:
    layers: List[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if spatial_dropout > 0:
        layers.append(nn.Dropout2d(p=spatial_dropout))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


class SimpleCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 15,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.features = nn.Sequential(
            *_feature_stage(in_channels, 32),
            *_feature_stage(32, 64, spatial_dropout=0.10),
            *_feature_stage(64, 128, spatial_dropout=0.15),
            *_feature_stage(128, 256, spatial_dropout=0.20),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


def get_model1_simple(
    num_classes: int = 15,
    dropout: float = 0.4,
    in_channels: int = 3,
) -> SimpleCNN:
    return SimpleCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = get_model1_simple()
    model.eval()
    sample = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(sample)
    print(f"[SimpleCNN] output shape: {tuple(logits.shape)}")
