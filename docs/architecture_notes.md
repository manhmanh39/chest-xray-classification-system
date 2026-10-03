# Model Architecture Notes

This note summarizes the three models used in the chest X-ray classification project and the main reasons behind their design. All models return 15 logits for multi-label classification.

## 1. Task setup

- **Task:** multi-label classification of chest X-ray images.
- **Input:** an RGB image tensor with shape `B x 3 x H x W` (normally `224 x 224`).
- **Output:** a tensor with shape `B x 15`, one logit for each label.
- **Loss:** `nn.BCEWithLogitsLoss`.

The models do not apply sigmoid inside `forward()`. `BCEWithLogitsLoss` already combines sigmoid and binary cross-entropy in a numerically stable way, so sigmoid is only needed later when probabilities are required.

---

## 2. SimpleCNN

**File:** `models/model1_simple.py`  
**Factory key:** `get_model("simple")`

`SimpleCNN` is the baseline model. Its structure is fully sequential, using repeated convolution, normalization, activation, and pooling stages before the classifier.

```text
Input: 3 x 224 x 224
   |
   |-- Conv 3 -> 32 -> BatchNorm -> ReLU -> MaxPool
   |      output: 32 x 112 x 112
   |
   |-- Conv 32 -> 64 -> BatchNorm -> ReLU -> Dropout2d(0.10) -> MaxPool
   |      output: 64 x 56 x 56
   |
   |-- Conv 64 -> 128 -> BatchNorm -> ReLU -> Dropout2d(0.15) -> MaxPool
   |      output: 128 x 28 x 28
   |
   |-- Conv 128 -> 256 -> BatchNorm -> ReLU -> Dropout2d(0.20) -> MaxPool
   |      output: 256 x 14 x 14
   |
   |-- AdaptiveAvgPool2d(1 x 1)
   |
   `-- Linear 256 -> 128 -> BatchNorm -> ReLU -> Dropout(0.4)
          -> Linear 128 -> 15
```

### Why it is designed this way

- The number of channels increases from `32 -> 64 -> 128 -> 256` while pooling reduces the spatial size. This lets later layers represent more complex features without keeping large feature maps throughout the network.
- `Dropout2d` is used in the convolutional part to regularize feature channels. The dropout rate increases slightly in deeper stages.
- `AdaptiveAvgPool2d((1, 1))` makes the classifier independent of a fixed feature-map size and avoids flattening a large convolutional tensor directly.
- The final `256 -> 128 -> 15` head adds one small hidden layer before the 15 output logits.

---

## 3. ComplexCNN

**File:** `models/model2_complex.py`  
**Factory key:** `get_model("complex")`

`ComplexCNN` combines sequential stages with a parallel multi-path block. The idea is to let one block process the same feature map through paths with different receptive fields before merging them again.

Chest X-ray findings can appear at different spatial scales. Some patterns cover a relatively large region, while others are more localized. Using several paths gives the model more flexibility than relying on a single convolution path everywhere.

### MultiPathParallelBlock

For an input with `C_in` channels, the block produces `C_out` channels. `C_out` must be divisible by four because the output channels are split equally across four branches.

                              Input
                                │
      ┌─────────────────────────┼─────────────────────────┬─────────────────────────┐
      │                         │                         │                         │
      ▼                         ▼                         ▼                         ▼
   Branch 1                  Branch 2                  Branch 3                  Branch 4
     1x1                    1x1 -> 3x3              1x1 -> 3x3               3x3 MaxPool
                                                        -> 3x3                  -> 1x1
      │                         │                         │                         │
      └───────────────┬─────────┴───────────────┬─────────┴───────────────┬─────────┘
                      │                         │                         │
                      └───────────────────── Concatenate ──────────────────┘
                                              │
                                              ▼
                                   + Residual Shortcut
                                              │
                                              ▼
                                             ReLU

The four branches have different roles:

- **Branch 1:** `1x1` convolution for channel projection.
- **Branch 2:** `1x1` followed by `3x3` for local features.
- **Branch 3:** `1x1` followed by two `3x3` convolutions. Two stacked `3x3` layers give an effective `5x5` receptive field. Considering only the spatial kernels, this uses `18` weights per channel pair instead of `25` for one `5x5` kernel.
- **Branch 4:** `3x3` max pooling with stride 1 and padding 1, followed by a `1x1` convolution.

A residual shortcut is added after concatenation. If the input and output channel counts are equal, the shortcut is an identity mapping. Otherwise, a `1x1` convolution with batch normalization is used to match the channel count.

### Overall structure

```text
Input
  |
Stem: Conv 3 -> 32 -> Conv 32 -> 64 -> MaxPool
  |
Stage 1: MultiPathBlock 64 -> 128
         MultiPathBlock 128 -> 128
         Dropout2d(0.10) -> MaxPool
  |
Stage 2: MultiPathBlock 128 -> 256
         MultiPathBlock 256 -> 256
         Dropout2d(0.15) -> MaxPool
  |
Stage 3: MultiPathBlock 256 -> 512
         MultiPathBlock 512 -> 512
         Dropout2d(0.20)
  |
AdaptiveAvgPool2d(1 x 1)
  |
Linear 512 -> 256 -> BatchNorm -> ReLU -> Dropout(0.4)
  |
Linear 256 -> 15
```

With a `224 x 224` input, the main spatial sizes are approximately `224 -> 112 -> 56 -> 28` through the three stages.

---

## 4. TransferLearningModel

**File:** `models/transfer_model.py`  
**Factory key:** `get_model("transfer", ...)`

The third model uses a pretrained torchvision backbone and replaces its original classifier with a 15-class head.

### Supported backbones

| Backbone | Pretrained weights | Feature dimension | Replaced classifier |
|---|---:|---:|---|
| `resnet18` | `ResNet18_Weights.DEFAULT` | 512 | `model.fc` |
| `resnet50` | `ResNet50_Weights.DEFAULT` | 2048 | `model.fc` |
| `mobilenet_v3` | `MobileNet_V3_Large_Weights.DEFAULT` | 960 | `model.classifier[0]` |
| `efficientnet_b0` | `EfficientNet_B0_Weights.DEFAULT` | 1280 | `model.classifier[1]` |
| `convnext_tiny` | `ConvNeXt_Tiny_Weights.DEFAULT` | 768 | `model.classifier[2]` |

### Freeze and fine-tune flow

When `freeze_base=True`, the backbone parameters are frozen and only the replacement classification head remains trainable. This is useful for the first stage of transfer learning.

Later, `unfreeze_backbone()` can enable gradients for the whole network so the backbone can be fine-tuned with a smaller learning rate.

---

## 5. Model factory

The package exposes one common function for creating all three models:

```python
from models import get_model

simple_model = get_model("simple", num_classes=15)
complex_model = get_model("complex", num_classes=15, dropout=0.4)
transfer_model = get_model(
    "transfer",
    num_classes=15,
    backbone_name="resnet50",
    pretrained=True,
    freeze_base=True,
)
```

The factory normalizes the model name with `strip()` and `lower()`. Unsupported names raise a `ValueError` instead of silently selecting another model.

---

## 6. Model tests

`tests/test_models.py` checks the model package independently from the training pipeline. The current tests cover:

- output shape `(batch_size, 15)` for all three architectures;
- transfer-learning models with `pretrained=False`, so tests do not depend on internet access;
- correct model selection through `get_model()`;
- clear errors for invalid model names and invalid multi-path channel settings.

These tests are intended as smoke and regression checks for the model code. Training quality is evaluated separately using the training and evaluation pipeline.
