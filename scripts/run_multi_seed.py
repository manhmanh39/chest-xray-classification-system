import argparse
import json

import numpy as np

import config
from evaluate import evaluate_checkpoint
from train import get_params, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "complex", "transfer"], required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--base_seed", type=int, default=config.SEED)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    params = get_params(args.model)
    results = []
    for seed in range(args.base_seed, args.base_seed + args.runs):
        checkpoint, _ = train(args.model, args.epochs, seed, params, f"{args.model}_seed{seed}.pth")
        results.append(evaluate_checkpoint(checkpoint, params["batch_size"]))

    summary = {}
    for key in results[0]:
        values = [run[key] for run in results]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        print(f"{key:<16} mean {summary[key]['mean']:.4f} | std {summary[key]['std']:.4f}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / f"multi_seed_{args.model}.json", "w", encoding="utf-8") as file:
        json.dump({"runs": results, "summary": summary}, file, indent=2)


if __name__ == "__main__":
    main()
