from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from .model1_simple import SimpleCNN, get_model1_simple
from .model2_complex import ComplexCNN, get_model2_complex
from .transfer_model import TransferLearningModel, get_model3_transfer

ModelBuilder = Callable[..., nn.Module]

_MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "simple": get_model1_simple,
    "complex": get_model2_complex,
    "transfer": get_model3_transfer,
}


def get_model(model_name: str, num_classes: int = 15, **kwargs: Any) -> nn.Module:
    key = model_name.strip().lower()
    try:
        builder = _MODEL_BUILDERS[key]
    except KeyError as exc:
        valid_names = ", ".join(_MODEL_BUILDERS)
        raise ValueError(
            f"Unknown model '{model_name}'. Expected one of: {valid_names}"
        ) from exc
    return builder(num_classes=num_classes, **kwargs)


__all__ = [
    "SimpleCNN",
    "ComplexCNN",
    "TransferLearningModel",
    "get_model",
    "get_model1_simple",
    "get_model2_complex",
    "get_model3_transfer",
]
