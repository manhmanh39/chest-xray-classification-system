import argparse
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import config
from evaluate import evaluate_checkpoint
from train import get_params, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "complex", "transfer"], required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--base_seed", type=int, default=config.seed)
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    params = get_params(args.model)
    results = []

    for i in range(args.runs):
        seed = args.base_seed + i
        print(f"\nRun {i + 1}/{args.runs} | Seed: {seed}")

        checkpoint, _ = train(
            args.model, args.epochs, seed, params,
            f"{args.model}_seed{seed}.pth"
        )
        metrics = evaluate_checkpoint(checkpoint, params["batch_size"])
        results.append({"seed": seed, **metrics})

        print("Test metrics:")
        for name, value in metrics.items():
            print(f"  {name:<16} {value:.4f}")

    metric_names = [name for name in results[0] if name != "seed"]
    summary = {}

    for name in metric_names:
        values = [run[name] for run in results]
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }

    print("\nSummary across runs:")
    for name, stats in summary.items():
        print(
            f"{name:<16} mean {stats['mean']:.4f} | "
            f"std {stats['std']:.4f} | "
            f"median {stats['median']:.4f}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"multi_run_median_{args.model}.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            {"model": args.model, "runs": results, "summary": summary},
            file,
            indent=2,
        )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()