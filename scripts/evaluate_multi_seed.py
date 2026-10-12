import argparse
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import config
from evaluate import calculate_metrics, predict_test_set
from train import per_class_auc


def summarize(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def normalize_metric_key(key):
    alias_map = {
        "auc": "roc_auc",
        "sensitivity": "recall",
        "recall": "recall",
        "specificity": "specificity",
        "roc_auc": "roc_auc",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "label_accuracy": "label_accuracy",
        "subset_accuracy": "subset_accuracy",
    }
    return alias_map.get(key, key)


def normalize_run(run):
    normalized = {}
    for key, value in run.items():
        normalized[normalize_metric_key(key)] = value
    return normalized


def evaluate_model(model_name, threshold):
    output_path = config.output_dir / f"multi_run_median_{model_name}.json"
    with open(output_path, encoding="utf-8") as file:
        results = json.load(file)

    per_class_runs = []
    for run in results["runs"]:
        normalized_run = normalize_run(run)
        checkpoint_path = config.checkpoint_dir / f"{model_name}_seed{normalized_run['seed']}.pth"
        probabilities, labels = predict_test_set(
            checkpoint_path, batch_size=config.batch_size
        )
        metrics = calculate_metrics(labels, probabilities, threshold)
        for name, value in metrics.items():
            saved_value = normalized_run.get(name)
            if saved_value is None and name == "roc_auc":
                saved_value = normalized_run.get("auc")
            if saved_value is None and name == "recall":
                saved_value = normalized_run.get("sensitivity")
            if saved_value is not None and not np.isclose(value, saved_value, rtol=0, atol=1e-5):
                raise ValueError(
                    f"{checkpoint_path.name}: {name}={value} does not match "
                    f"the saved multi-seed result {saved_value}"
                )

        labels = labels.astype(int)
        predictions = (probabilities >= threshold).astype(int)
        auc_by_class = per_class_auc(labels, probabilities)
        per_class = []
        for index, class_name in enumerate(config.class_names):
            class_labels = labels[:, index]
            class_predictions = predictions[:, index]
            positives = int(class_labels.sum())
            negatives = len(class_labels) - positives
            true_positive = int(((class_predictions == 1) & (class_labels == 1)).sum())
            true_negative = int(((class_predictions == 0) & (class_labels == 0)).sum())
            per_class.append(
                {
                    "class": class_name,
                    "support": positives,
                    "roc_auc": auc_by_class[class_name],
                    "recall": true_positive / positives if positives else 0.0,
                    "specificity": true_negative / negatives if negatives else 0.0,
                }
            )

        normalized_run["per_class"] = per_class
        results["runs"][results["runs"].index(run)] = normalized_run
        per_class_runs.append(per_class)

    results["per_class_summary"] = {}
    for index, class_name in enumerate(config.class_names):
        class_runs = [run[index] for run in per_class_runs]
        results["per_class_summary"][class_name] = {
            "support": class_runs[0]["support"],
            "roc_auc": summarize([run["roc_auc"] for run in class_runs]),
            "recall": summarize([run["recall"] for run in class_runs]),
            "specificity": summarize([run["specificity"] for run in class_runs]),
        }

    temporary_path = output_path.with_suffix(".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    temporary_path.replace(output_path)
    print(f"Updated {output_path} with per-disease results for {len(per_class_runs)} seeds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "complex", "transfer"], required=True)
    parser.add_argument("--threshold", type=float, default=config.threshold)
    args = parser.parse_args()
    evaluate_model(args.model, args.threshold)


if __name__ == "__main__":
    main()