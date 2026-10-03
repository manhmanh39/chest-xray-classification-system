from __future__ import annotations

import torch
from torch import nn


class ConvBNReLU(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiPathParallelBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")

        branch_channels = out_channels // 4

        self.branch1 = ConvBNReLU(in_channels, branch_channels, 1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, branch_channels, 1),
            ConvBNReLU(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, branch_channels, 1),
            ConvBNReLU(branch_channels, branch_channels, 3, padding=1),
            ConvBNReLU(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, branch_channels, 1),
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = (
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
        )
        merged = torch.cat(branch_outputs, dim=1)
        return self.relu(merged + self.shortcut(x))


class ComplexCNN(nn.Module):

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

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, 1, 1),
            ConvBNReLU(32, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage1 = nn.Sequential(
            MultiPathParallelBlock(64, 128),
            MultiPathParallelBlock(128, 128),
            nn.Dropout2d(p=0.10),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            MultiPathParallelBlock(128, 256),
            MultiPathParallelBlock(256, 256),
            nn.Dropout2d(p=0.15),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            MultiPathParallelBlock(256, 512),
            MultiPathParallelBlock(512, 512),
            nn.Dropout2d(p=0.20),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        return self.classifier(x)


def get_model2_complex(
    num_classes: int = 15,
    dropout: float = 0.4,
    in_channels: int = 3,
) -> ComplexCNN:
    return ComplexCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = get_model2_complex()
    model.eval()
    sample = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(sample)
    print(f"[ComplexCNN] output shape: {tuple(logits.shape)}")
