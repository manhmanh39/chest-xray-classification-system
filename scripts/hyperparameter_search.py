import argparse
import json
import sys
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader, Subset

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import config
from preprocessing import get_dataloaders
from train import fit, set_seed


def objective(trial, model_name, epochs, sample_ratio):
    batch_sizes = [32, 64] if model_name == "simple" else [16, 32]
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", batch_sizes),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "sgd"]),
    }

    set_seed(config.seed)
    loaders, pos_weight = get_dataloaders(batch_size=params["batch_size"])
    dataset = loaders["train"].dataset
    indices = torch.randperm(len(dataset))[: int(len(dataset) * sample_ratio)].tolist()
    loaders["train"] = DataLoader(
        Subset(dataset, indices),
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )

    history = fit(model_name, loaders, pos_weight, epochs, params, trial=trial)
    return max(row["val_auc"] for row in history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "complex", "transfer"], required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sample_ratio", type=float, default=0.3)
    args = parser.parse_args()

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        sampler=sampler,
    )
    study.optimize(
        lambda trial: objective(trial, args.model, args.epochs, args.sample_ratio),
        n_trials=args.trials,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"best_params_{args.model}.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=2)
    print(f"Best validation AUC: {study.best_value:.4f}")
    print(f"Best parameters saved to {path}")


if __name__ == "__main__":
    main()
