import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, dropout=0.3, pretrained=True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)
        features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True