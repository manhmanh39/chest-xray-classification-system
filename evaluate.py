import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

import config
from dataset import get_dataloaders
from train import create_model, macro_auc, per_class_auc


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probabilities, labels = [], []
    for images, targets in tqdm(loader, desc="predict", leave=False):
        outputs = model(images.to(config.DEVICE))
        probabilities.append(torch.sigmoid(outputs).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def calculate_metrics(labels, probabilities, threshold=config.THRESHOLD):
    labels = labels.astype(int)
    predictions = (probabilities >= threshold).astype(int)
    true_positive = ((predictions == 1) & (labels == 1)).sum()
    true_negative = ((predictions == 0) & (labels == 0)).sum()
    return {
        "auc": macro_auc(labels, probabilities),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "sensitivity": float(true_positive / labels.sum()),
        "specificity": float(true_negative / (1 - labels).sum()),
        "label_accuracy": float((predictions == labels).mean()),
        "subset_accuracy": float((predictions == labels).all(axis=1).mean()),
    }


def predict_test_set(checkpoint_path, batch_size):
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model = create_model(checkpoint["model_name"], checkpoint["dropout"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(config.DEVICE)
    loaders, _ = get_dataloaders(batch_size=batch_size)
    return predict(model, loaders["test"])


def evaluate_checkpoint(checkpoint_path, batch_size=config.BATCH_SIZE, threshold=config.THRESHOLD):
    probabilities, labels = predict_test_set(checkpoint_path, batch_size)
    return calculate_metrics(labels, probabilities, threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=config.THRESHOLD)
    args = parser.parse_args()

    probabilities, labels = predict_test_set(args.checkpoint, args.batch_size)
    result = calculate_metrics(labels, probabilities, args.threshold)
    result["threshold"] = args.threshold
    result["per_class_auc"] = per_class_auc(labels.astype(int), probabilities)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / "evaluation.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
