import argparse
import json

import optuna
import torch
from torch.utils.data import DataLoader, Subset

import config
from dataset import get_dataloaders
from train import fit, set_seed


def objective(trial, model_name, epochs, sample_ratio):
    if model_name == "simple":
        batch_sizes = [32, 64]

        # Based on the best Simple CNN result found so far.
        learning_rate_range = (5e-5, 5e-4)
        weight_decay_range = (1e-7, 3e-5)

    else:
        batch_sizes = [16, 32]

        # Based on the best Complex CNN result found so far.
        learning_rate_range = (5e-5, 4e-4)
        weight_decay_range = (1e-7, 5e-5)

    params = {
        "learning_rate": trial.suggest_float( "learning_rate", *learning_rate_range, log=True, ),
        "batch_size": trial.suggest_categorical( "batch_size", batch_sizes,),
        "weight_decay": trial.suggest_float( "weight_decay", *weight_decay_range, log=True,),
        "dropout": trial.suggest_float("dropout", 0.05, 0.30,),
        "optimizer": "adamw",
        }

    set_seed(config.SEED)

    loaders, pos_weight = get_dataloaders( batch_size=params["batch_size"], seed=config.SEED,)

    dataset = loaders["train"].dataset

    indices = torch.randperm( len(dataset), generator=torch.Generator().manual_seed(config.SEED),)[: int(len(dataset) * sample_ratio)].tolist()

    loaders["train"] = DataLoader(
        Subset(dataset, indices),
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.NUM_WORKERS > 0,
        drop_last=True,
    )

    history = fit( model_name, loaders, pos_weight, epochs, params, trial=None,)

    return max(row["val_auc"] for row in history)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["simple", "complex"], required=True,)

    parser.add_argument( "--trials", type=int, default=20,)

    parser.add_argument( "--epochs", type=int, default=15,)

    parser.add_argument("--sample_ratio", type=float, default=0.3,)

    args = parser.parse_args()

    sampler = optuna.samplers.TPESampler(seed=config.SEED,)

    study = optuna.create_study( direction="maximize", sampler=sampler, pruner=optuna.pruners.NopPruner(),)

    if args.model == "simple":
        study.enqueue_trial({
            "learning_rate": 1.2858e-4,
            "batch_size": 32,
            "weight_decay": 1.523e-6,
            "dropout": 0.115,
        })

    else:
        study.enqueue_trial({
            "learning_rate": 1.2477e-4,
            "batch_size": 16,
            "weight_decay": 9.815e-5,
            "dropout": 0.168,
        })

    study.optimize(
        lambda trial: objective(
            trial,
            args.model,
            args.epochs,
            args.sample_ratio,
        ),
        n_trials=args.trials,
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = config.OUTPUT_DIR / f"best_params_{args.model}.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=2)

    print()
    print(f"Best validation AUC: {study.best_value:.4f}")
    print(f"Best parameters saved to {path}")
    print("Best parameters:")

    for name, value in study.best_params.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()