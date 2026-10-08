import argparse
import json
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import SGD, AdamW
from tqdm import tqdm

import config
from dataset import get_dataloaders
from models.complex_cnn import ComplexCNN
from models.simple_cnn import SimpleCNN
from models.transfer import TransferLearningModel

USE_AMP = config.DEVICE.type == "cuda"
PATIENCE = 6

DEFAULT_PARAMS = {
    "learning_rate": config.LEARNING_RATE,
    "batch_size": config.BATCH_SIZE,
    "weight_decay": config.WEIGHT_DECAY,
    "dropout": 0.4,
    "optimizer": "adamw",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_params(model_name):
    params = dict(DEFAULT_PARAMS)
    path = config.OUTPUT_DIR / f"best_params_{model_name}.json"
    if path.exists():
        with open(path, encoding="utf-8") as file:
            params.update(json.load(file))
    return params


def create_model(name, dropout, pretrained=True):
    if name == "simple":
        return SimpleCNN(config.NUM_CLASSES, dropout)
    if name == "complex":
        return ComplexCNN(config.NUM_CLASSES, dropout)
    if name == "transfer":
        return TransferLearningModel(config.NUM_CLASSES, dropout, pretrained)
    raise ValueError(f"Unknown model: {name}")


def create_optimizer(model, name, learning_rate, weight_decay):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def to_device(images, labels):
    images = images.to(config.DEVICE, non_blocking=True, memory_format=torch.channels_last)
    return images, labels.to(config.DEVICE, non_blocking=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_loss, count = 0.0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = to_device(images, labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(config.DEVICE.type, enabled=USE_AMP):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        count += images.size(0)
    return total_loss / count


def per_class_auc(labels, probabilities):
    result = {}
    for index, name in enumerate(config.CLASS_NAMES):
        column = labels[:, index]
        has_both = len(np.unique(column)) == 2
        result[name] = float(roc_auc_score(column, probabilities[:, index])) if has_both else None
    return result


def macro_auc(labels, probabilities):
    scores = [s for s in per_class_auc(labels, probabilities).values() if s is not None]
    return float(np.mean(scores))


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, count = 0.0, 0
    probabilities, labels = [], []
    for images, targets in loader:
        images, targets = to_device(images, targets)
        with torch.autocast(config.DEVICE.type, enabled=USE_AMP):
            outputs = model(images)
        outputs = outputs.float()
        total_loss += criterion(outputs, targets).item() * images.size(0)
        count += images.size(0)
        probabilities.append(torch.sigmoid(outputs).cpu().numpy())
        labels.append(targets.cpu().numpy())
    auc = macro_auc(np.concatenate(labels), np.concatenate(probabilities))
    return total_loss / count, auc


def fit(model_name, loaders, pos_weight, epochs, params, checkpoint_path=None, trial=None, patience=None):
    model = create_model(model_name, params["dropout"]).to(config.DEVICE, memory_format=torch.channels_last)
    if model_name == "transfer":
        model.freeze_backbone()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config.DEVICE))
    optimizer = create_optimizer(model, params["optimizer"], params["learning_rate"], params["weight_decay"])
    scaler = torch.amp.GradScaler(config.DEVICE.type, enabled=USE_AMP)

    best_auc, wait, history = 0.0, 0, []
    for epoch in range(1, epochs + 1):
        if model_name == "transfer" and epoch == config.FREEZE_EPOCHS + 1:
            model.unfreeze_backbone()
            optimizer = create_optimizer(model, params["optimizer"], config.UNFREEZE_LR, params["weight_decay"])

        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler)
        val_loss, val_auc = validate(model, loaders["val"], criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})

        is_best = val_auc > best_auc
        wait = 0 if is_best else wait + 1
        if is_best:
            best_auc = val_auc
            if checkpoint_path is not None:
                torch.save(
                    {"model": model.state_dict(), "model_name": model_name,
                     "dropout": params["dropout"], "epoch": epoch, "val_auc": val_auc},
                    checkpoint_path,
                )

        print(
            f"Epoch {epoch}/{epochs} - train loss: {train_loss:.4f} - "
            f"val loss: {val_loss:.4f} - val AUC: {val_auc:.4f}{' *' if is_best else ''}"
        )
        if trial is not None:
            trial.report(best_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if patience is not None and wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    return history


def train(model_name, epochs=config.EPOCHS, seed=config.SEED, params=None, checkpoint_name=None):
    params = params or get_params(model_name)
    set_seed(seed)
    loaders, pos_weight = get_dataloaders(batch_size=params["batch_size"], seed=seed)

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.CHECKPOINT_DIR / (checkpoint_name or f"{model_name}_best.pth")
    history = fit(model_name, loaders, pos_weight, epochs, params, checkpoint_path, patience=PATIENCE)
    return checkpoint_path, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "complex", "transfer"], required=True)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"])
    args = parser.parse_args()

    params = get_params(args.model)
    overrides = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
    }
    params.update({k: v for k, v in overrides.items() if v is not None})

    print(params)
    checkpoint_path, history = train(args.model, args.epochs, args.seed, params)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / f"{args.model}_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()