import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import config


model_titles = {
    "simple": "Simple CNN",
    "complex": "Complex CNN",
    "transfer": "DenseNet121",
}
model_grays = {
    "simple": "#404040",
    "complex": "#858585",
    "transfer": "#C4C4C4",
}
metrics = {
    "roc_auc": "Macro ROC-AUC",
    "f1_score": "Macro F1",
    "f1_micro": "Micro F1",
}


def load_runs(model_name):
    path = config.output_dir / f"multi_run_median_{model_name}.json"
    if not path.is_file():
        raise ValueError(f"Missing multi-run results: {path}")

    with path.open(encoding="utf-8") as file:
        results = json.load(file)

    runs = results.get("runs")
    if not isinstance(runs, list) or len(runs) != 3:
        raise ValueError(f"Expected exactly three saved runs in {path}")

    for run in runs:
        for metric in metrics:
            value = run.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing or invalid {metric} in {path}")
            if not 0 <= value <= 1:
                raise ValueError(f"{metric} is outside [0, 1] in {path}")

    return runs


def main():
    runs_by_model = {name: load_runs(name) for name in model_titles}
    means = {name: [] for name in model_titles}
    standard_deviations = {name: [] for name in model_titles}

    for model_name, runs in runs_by_model.items():
        for metric in metrics:
            values = [run[metric] for run in runs]
            means[model_name].append(statistics.mean(values))
            standard_deviations[model_name].append(statistics.pstdev(values))

    positions = np.arange(len(metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12, 7))

    for index, model_name in enumerate(model_titles):
        offset = (index - 1) * width
        axis.bar(
            positions + offset,
            means[model_name],
            width,
            yerr=standard_deviations[model_name],
            capsize=4,
            color=model_grays[model_name],
            edgecolor="black",
            ecolor="#303030",
            label=model_titles[model_name],
            linewidth=0.75,
        )

    axis.set_xticks(positions, metrics.values(), fontsize=14)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score", fontsize=15)
    axis.tick_params(axis="y", labelsize=13)
    axis.set_title("Model comparison across three runs", fontsize=19, pad=14)
    axis.legend(frameon=False, ncols=3, loc="upper left", fontsize=12)
    axis.grid(axis="y", linestyle=":", color="#B8B8B8", alpha=0.55)
    axis.set_axisbelow(True)
    figure.tight_layout()

    output_path = config.output_dir / "model_metrics_comparison.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot plot model metrics: {error}") from error