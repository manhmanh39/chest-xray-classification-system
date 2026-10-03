"""Smoke tests for the three model architectures and the public factory."""

from __future__ import annotations

import pytest
import torch

from models import ComplexCNN, SimpleCNN, TransferLearningModel, get_model


def _forward_shape(model: torch.nn.Module, image_size: int = 64) -> tuple[int, ...]:
    model.eval()
    batch = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        output = model(batch)
    return tuple(output.shape)


@pytest.mark.parametrize(
    ("model_class", "kwargs"),
    [
        (SimpleCNN, {}),
        (ComplexCNN, {}),
    ],
)
def test_custom_models_return_expected_shape(model_class, kwargs) -> None:
    model = model_class(num_classes=15, **kwargs)
    assert _forward_shape(model) == (2, 15)


def test_transfer_model_returns_expected_shape_without_downloading_weights() -> None:
    model = TransferLearningModel(
        num_classes=15,
        backbone_name="resnet18",
        pretrained=False,
        freeze_base=False,
    )
    assert _forward_shape(model) == (2, 15)


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("simple", SimpleCNN),
        ("complex", ComplexCNN),
        ("transfer", TransferLearningModel),
    ],
)
def test_factory_returns_expected_model(name: str, expected_type: type) -> None:
    kwargs = {"pretrained": False, "backbone_name": "resnet18"} if name == "transfer" else {}
    model = get_model(name, num_classes=15, **kwargs)
    assert isinstance(model, expected_type)


def test_factory_normalizes_whitespace_and_case() -> None:
    model = get_model("  SIMPLE  ", num_classes=15)
    assert isinstance(model, SimpleCNN)


def test_factory_rejects_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="Expected one of: simple, complex, transfer"):
        get_model("not-a-model")


def test_parallel_block_validates_channel_count() -> None:
    from models.model2_complex import MultiPathParallelBlock

    with pytest.raises(ValueError, match="divisible by 4"):
        MultiPathParallelBlock(in_channels=32, out_channels=130)
