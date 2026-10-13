import argparse
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
model_grays = {
    "simple": "#404040",
    "complex": "#858585",
    "transfer": "#C4C4C4",
}


def load_per_class_auc(model_name):
    input_path = config.output_dir / f"multi_run_median_{model_name}.json"
    with open(input_path, encoding="utf-8") as file:
        results = json.load(file)

    per_class = results.get("per_class_summary")
    if not isinstance(per_class, dict) or not per_class:
        raise ValueError(f"Missing per-class results in {input_path}")

    class_names = list(per_class)
    means, standard_deviations = [], []
    for class_name in class_names:
        summary = per_class[class_name].get("roc_auc", per_class[class_name].get("auc"))
        if not isinstance(summary, dict):
            raise ValueError(f"Missing ROC-AUC summary for {class_name} in {input_path}")
        mean = summary.get("mean")
        standard_deviation = summary.get("std")
        if (
            not isinstance(mean, (int, float))
            or not math.isfinite(mean)
            or not isinstance(standard_deviation, (int, float))
            or not math.isfinite(standard_deviation)
        ):
            raise ValueError(f"Invalid ROC-AUC summary for {class_name} in {input_path}")
        means.append(mean)
        standard_deviations.append(standard_deviation)

    return class_names, means, standard_deviations


def plot_models(model_names):
    results = {name: load_per_class_auc(name) for name in model_names}
    class_names = results[model_names[0]][0]
    if any(results[name][0] != class_names for name in model_names[1:]):
        raise ValueError("Per-class label order differs between model results")

    positions = list(range(len(class_names)))
    figure, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(7.2 * len(model_names), 8.5),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for axis, model_name in zip(axes, model_names):
        _, means, standard_deviations = results[model_name]
        axis.barh(
            positions,
            means,
            xerr=standard_deviations,
            color=model_grays[model_name],
            edgecolor="black",
            linewidth=0.65,
            ecolor="#303030",
            capsize=3,
        )
        axis.set_yticks(positions, class_names)
        axis.invert_yaxis()
        axis.set_xlim(0.5, 1.0)
        axis.set_title(model_titles[model_name], fontsize=17, pad=12)
        axis.tick_params(axis="both", labelsize=13)
        axis.grid(axis="x", linestyle=":", color="#B8B8B8", alpha=0.55)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("Disease label", fontsize=15)
    figure.supxlabel("Test ROC-AUC (mean +/- std across seeds)", fontsize=16)
    figure.suptitle("Per-class test ROC-AUC by model", fontsize=20)
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    suffix = "comparison" if len(model_names) > 1 else model_names[0]
    output_path = config.output_dir / f"per_class_auc_{suffix}.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path} ({len(model_names)} models, {len(class_names)} classes)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-disease AUC mean and standard deviation from multi-seed results."
    )
    parser.add_argument(
        "--model",
        choices=["all", "simple", "complex", "transfer"],
        default="all",
    )
    args = parser.parse_args()

    model_names = list(model_titles) if args.model == "all" else [args.model]
    try:
        plot_models(model_names)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot plot per-class AUC: {error}") from error


if __name__ == "__main__":
    main()