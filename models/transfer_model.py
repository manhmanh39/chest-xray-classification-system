from __future__ import annotations

from typing import Literal, TypeAlias, cast

import torch
from torch import nn
from torchvision import models

BackboneName: TypeAlias = Literal[
    "resnet18",
    "resnet50",
    "mobilenet_v3",
    "efficientnet_b0",
    "convnext_tiny",
]

SUPPORTED_BACKBONES: tuple[str, ...] = (
    "resnet18",
    "resnet50",
    "mobilenet_v3",
    "efficientnet_b0",
    "convnext_tiny",
)


class TransferLearningModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 15,
        backbone_name: BackboneName = "resnet50",
        pretrained: bool = True,
        freeze_base: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if backbone_name not in SUPPORTED_BACKBONES:
            supported = ", ".join(SUPPORTED_BACKBONES)
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. Expected one of: {supported}"
            )

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.backbone = self._build_backbone(
            backbone_name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

        if freeze_base:
            self.freeze_backbone()

    @staticmethod
    def _build_backbone(
        backbone_name: BackboneName,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ) -> nn.Module:
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
            return backbone

        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
            return backbone

        if backbone_name == "mobilenet_v3":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_large(weights=weights)
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
            return backbone

        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
            return backbone

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        in_features = backbone.classifier[2].in_features
        backbone.classifier[2] = nn.Linear(in_features, num_classes)
        return backbone

    def _classification_head(self) -> nn.Module:
        if hasattr(self.backbone, "fc"):
            return cast(nn.Module, self.backbone.fc)
        return cast(nn.Module, self.backbone.classifier)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self._classification_head().parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def get_model3_transfer(
    num_classes: int = 15,
    backbone_name: BackboneName = "resnet50",
    pretrained: bool = True,
    freeze_base: bool = True,
    dropout: float = 0.3,
) -> TransferLearningModel:
    return TransferLearningModel(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=pretrained,
        freeze_base=freeze_base,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = get_model3_transfer(
        backbone_name="resnet18",
        pretrained=False,
        freeze_base=False,
    )
    model.eval()
    sample = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(sample)
    print(f"[TransferLearningModel] output shape: {tuple(logits.shape)}")
