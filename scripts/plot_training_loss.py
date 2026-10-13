import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import config


model_titles = {
    "simple": "Simple CNN",
    "complex": "Complex CNN",
    "transfer": "DenseNet121 Transfer Learning",
}


def load_history(model_name):
    path = config.output_dir / f"{model_name}_history.json"
    if not path.is_file():
        raise ValueError(f"Missing training history: {path}")

    with path.open(encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list) or not history:
        raise ValueError(f"Training history is empty or invalid: {path}")

    expected_epochs = list(range(1, len(history) + 1))
    epochs = [row.get("epoch") for row in history if isinstance(row, dict)]
    if len(epochs) != len(history) or epochs != expected_epochs:
        raise ValueError(f"Epoch records are incomplete or out of order: {path}")

    for row in history:
        for key in ("train_loss", "val_loss"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {key} at epoch {row['epoch']}: {path}")

    return history


def main():
    histories = {name: load_history(name) for name in model_titles}
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for axis, (model_name, history) in zip(axes, histories.items()):
        epochs = [row["epoch"] for row in history]
        marker_interval = max(1, len(epochs) // 10)
        axis.plot(
            epochs,
            [row["train_loss"] for row in history],
            color="#252525",
            linestyle="-",
            marker="o",
            markevery=marker_interval,
            markersize=5,
            linewidth=2,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="Training loss",
        )
        axis.plot(
            epochs,
            [row["val_loss"] for row in history],
            color="#858585",
            linestyle="--",
            marker="s",
            markevery=marker_interval,
            markersize=4.5,
            linewidth=2,
            markerfacecolor="#E2E2E2",
            markeredgecolor="#707070",
            markeredgewidth=1,
            label="Validation loss",
        )
        axis.set_title(model_titles[model_name], fontsize=16)
        axis.set_xlabel("Epoch", fontsize=14)
        axis.set_xticks(epochs[:: max(1, len(epochs) // 10)])
        axis.tick_params(axis="both", labelsize=12)
        axis.grid(linestyle=":", color="#B8B8B8", alpha=0.55)
        axis.set_axisbelow(True)

        if model_name == "transfer":
            unfreeze_epoch = config.freeze_epochs + 1
            axis.axvline(
                unfreeze_epoch - 0.5,
                color="#A0A0A0",
                linestyle="-.",
                linewidth=1.8,
                label=f"Backbone unfreezes (epoch {unfreeze_epoch})",
            )

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[-1].legend(loc="best", fontsize=10, frameon=False)
    figure.suptitle("Training and validation loss by epoch", fontsize=18, y=1.02)
    figure.tight_layout()

    output_path = config.output_dir / "training_loss_by_epoch.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot plot incomplete training histories: {error}") from error