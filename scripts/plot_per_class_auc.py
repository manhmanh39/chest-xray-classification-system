import argparse
import json
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
model_colors = {
    "simple": "#2878A5",
    "complex": "#D47A36",
    "transfer": "#4F8A62",
}


def plot_model(model_name):
    input_path = config.output_dir / f"multi_run_median_{model_name}.json"
    with open(input_path, encoding="utf-8") as file:
        results = json.load(file)

    per_class = results["per_class_summary"]
    class_names = list(per_class)
    means = [per_class[name]["auc"]["mean"] for name in class_names]
    standard_deviations = [per_class[name]["auc"]["std"] for name in class_names]
    positions = list(range(len(class_names)))

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        positions,
        means,
        xerr=standard_deviations,
        color=model_colors[model_name],
        alpha=0.88,
        capsize=3,
    )
    axis.set_yticks(positions, class_names)
    axis.invert_yaxis()
    axis.set_xlim(0.5, 1.0)
    axis.set_xlabel("Test AUC (mean +/- std across seeds)")
    axis.set_title(f"{model_titles[model_name]}: per-disease test AUC")
    axis.grid(axis="x", linestyle=":", alpha=0.45)
    axis.set_axisbelow(True)
    figure.tight_layout()

    output_path = config.output_dir / f"per_class_auc_{model_name}.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path} ({len(class_names)} classes)")


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
    for model_name in model_names:
        plot_model(model_name)


if __name__ == "__main__":
    main()