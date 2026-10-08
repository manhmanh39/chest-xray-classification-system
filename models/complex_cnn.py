import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiPathParallelBlock(nn.Module):
    """4 parallel branches -> concat -> 1x1 fusion -> residual add."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")
        branch_channels = out_channels // 4

        self.branch1 = ConvBNReLU(in_channels, branch_channels, 1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, branch_channels, 1),
            ConvBNReLU(branch_channels, branch_channels, 3),
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, branch_channels, 1),
            ConvBNReLU(branch_channels, branch_channels, 3),
            ConvBNReLU(branch_channels, branch_channels, 3),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_channels, branch_channels, 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branches = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        merged = self.fusion(torch.cat(branches, dim=1))
        return self.relu(merged + self.shortcut(x))


class ComplexCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            ConvBNReLU(32, 64),
        )
        self.stage1 = nn.Sequential(
            MultiPathParallelBlock(64, 128),
            MultiPathParallelBlock(128, 128),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            MultiPathParallelBlock(128, 256),
            MultiPathParallelBlock(256, 256),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.MaxPool2d(2),
            MultiPathParallelBlock(256, 512),
            MultiPathParallelBlock(512, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.classifier(x)