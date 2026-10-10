import torch.nn as nn


def feature_stage(in_channels, out_channels, dropout=0):
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            feature_stage(3, 32),
            feature_stage(32, 64),
            feature_stage(64, 128),
            feature_stage(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)