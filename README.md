# Chest X-Ray Multi-Label Classification

Multi-label classification of 15 thoracic findings (14 diseases plus "No finding") on VinBigData chest X-rays, with three models compared on one fixed split.

## Layout

```text
models/        SimpleCNN, ComplexCNN, TransferLearningModel and get_model()
data/          prepare_dataset.py (DICOM -> PNG + stratified split), dataset.py (loaders)
training/      settings.py (config + tuned hparams), engine.py, train.py
evaluation/    metrics.py, evaluate.py
scripts/       hyperparameter_search.py, run_multi_seed.py
utils/         seed.py
tests/         test_models.py, test_data.py, test_training_eval.py
docs/          architecture_notes.md
main.py        train + evaluate all models and print a summary table
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

Download the VinBigData competition files and place `train.csv` in `datasets/raw/`. Then build the split:

```bash
python -m data.prepare_dataset --zip_path <path-to-vinbigdata.zip>
```

- A label is positive when any radiologist boxed that finding. An image with no disease box is "No finding".
- The split is multilabel-stratified (`iterative-stratification`), 80/10/10, so rare findings appear in every split. It is fixed by `--seed` (default 202601).
- Images are saved as grayscale PNG (224 px, percentile-normalised, MONOCHROME1 inverted). Stale PNGs from an earlier split are deleted, so splits never leak.
- `datasets/processed/split_summary.csv` lists the positive count per class and split. Use it in the report.

If the processed split is missing, training stops with a message telling you to run the command above.

## Training

```bash
python -m training.train --model simple
python -m training.train --model transfer --backbone resnet50
```

Learning rate, batch size, weight decay, optimizer and dropout are read from `outputs/best_hparams_<model>.json` when it exists. Every entry point (`training.train`, `main.py`, `run_multi_seed`) goes through `resolve_train_config`, so they all use the same values. Pass `--no_tuned` to ignore the file; explicit CLI flags always win. The best checkpoint is chosen by validation macro AUC.

```bash
python -m scripts.hyperparameter_search --model simple --n_trials 30
python -m scripts.run_multi_seed --model transfer --n_runs 5
```

## Evaluation

```bash
python -m evaluation.evaluate --checkpoint checkpoints/transfer_seed202601_best.pth
python main.py --model all
```

Reported on the test split: macro AUC, macro/micro F1, mean sensitivity and specificity, per-label accuracy and subset (exact-match) accuracy. Per-class results are saved to `outputs/eval_<model>.json` and `outputs/per_class_auc_<model>.png`.

Checkpoints are selected on validation only. Report mean and std (and median) over several seeds on the fixed split rather than the best single run.

## Tests

```bash
python -m pytest -q
```

## Contributions

| Member | Area |
|---|---|
| Ánh | Model architectures, factory, model tests, architecture notes |
| Mai Anh | Data preparation, loaders, config, seeding, dependencies, data tests |
| Khánh Dương | Training, evaluation, tuning, multi-seed runner, main pipeline, README |

## References

- VinBigData Chest X-ray Abnormalities Detection (Kaggle).
- Pretrained backbones from `torchvision.models`.
