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
from preprocessing import get_dataloaders
from models.complex_cnn import ComplexCNN
from models.simple_cnn import SimpleCNN
from models.transfer import TransferLearningModel

use_amp = config.device.type == "cuda"
patience = 6

default_params = {
    "learning_rate": config.learning_rate,
    "batch_size": config.batch_size,
    "weight_decay": config.weight_decay,
    "dropout": 0.4,
    "optimizer": "adamw",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_params(model_name):
    params = dict(default_params)
    path = config.output_dir / f"best_params_{model_name}.json"
    if path.exists():
        with open(path, encoding="utf-8") as file:
            params.update(json.load(file))
    return params


def create_model(name, dropout, pretrained=True):
    if name == "simple":
        return SimpleCNN(config.num_classes, dropout)
    if name == "complex":
        return ComplexCNN(config.num_classes, dropout)
    if name == "transfer":
        return TransferLearningModel(config.num_classes, dropout, pretrained)
    raise ValueError(f"Unknown model: {name}")


def create_optimizer(model, name, learning_rate, weight_decay):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def to_device(images, labels):
    images = images.to(config.device, non_blocking=True, memory_format=torch.channels_last)
    return images, labels.to(config.device, non_blocking=True)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_loss, count = 0.0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = to_device(images, labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(config.device.type, enabled=use_amp):
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
    for index, name in enumerate(config.class_names):
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
        with torch.autocast(config.device.type, enabled=use_amp):
            outputs = model(images)
        outputs = outputs.float()
        total_loss += criterion(outputs, targets).item() * images.size(0)
        count += images.size(0)
        probabilities.append(torch.sigmoid(outputs).cpu().numpy())
        labels.append(targets.cpu().numpy())
    auc = macro_auc(np.concatenate(labels), np.concatenate(probabilities))
    return total_loss / count, auc


def fit(model_name, loaders, pos_weight, epochs, params, checkpoint_path=None, trial=None, patience=None):
    model = create_model(model_name, params["dropout"]).to(config.device, memory_format=torch.channels_last)
    if model_name == "transfer":
        model.freeze_backbone()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config.device))
    optimizer = create_optimizer(model, params["optimizer"], params["learning_rate"], params["weight_decay"])
    scaler = torch.amp.GradScaler(config.device.type, enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=params["learning_rate"] * 0.01
    )
    best_auc, wait, history = 0.0, 0, []
    for epoch in range(1, epochs + 1):
        if model_name == "transfer" and epoch == config.freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = create_optimizer(model, params["optimizer"], config.unfreeze_lr, params["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - config.freeze_epochs),
                eta_min=config.unfreeze_lr * 0.01,
            )
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler)
        scheduler.step()
        val_loss, val_auc = validate(model, loaders["val"], criterion)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "lr": current_lr,
        })
        is_best = val_auc > best_auc
        wait = 0 if is_best else wait + 1
        if is_best:
            best_auc = val_auc
            if checkpoint_path is not None:
                torch.save({
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "dropout": params["dropout"],
                    "epoch": epoch,
                    "val_auc": val_auc,
                }, checkpoint_path)
        marker = " *" if is_best else ""
        print(
            f"Epoch {epoch}/{epochs} (lr: {current_lr:.2e}) - "
            f"train loss: {train_loss:.4f} - val loss: {val_loss:.4f} - "
            f"val AUC: {val_auc:.4f}{marker}"
        )
        if trial is not None:
            trial.report(best_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if patience is not None and wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    return history


def train(model_name, epochs=config.epochs, seed=config.seed, params=None, checkpoint_name=None):
    params = params or get_params(model_name)
    set_seed(seed)
    loaders, pos_weight = get_dataloaders(batch_size=params["batch_size"], seed=seed)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.checkpoint_dir / (checkpoint_name or f"{model_name}_best.pth")
    history = fit(model_name, loaders, pos_weight, epochs, params, checkpoint_path, patience=patience)
    return checkpoint_path, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "complex", "transfer"], required=True)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--seed", type=int, default=config.seed)
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
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with open(config.output_dir / f"{args.model}_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    print(f"Best model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()