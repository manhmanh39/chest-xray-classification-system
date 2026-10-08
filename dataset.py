import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size=config.IMAGE_SIZE):
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomRotation(10),
        T.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        normalize,
    ])

    eval_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize,
    ])

    return train_transform, eval_transform


class ChestXrayDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.from_numpy(self.labels[index])
        return image, label


def load_split(data_dir, split):
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{split}.csv"
    image_dir = data_dir / split / "images"

    if not csv_path.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Processed '{split}' split not found in {data_dir}.")

    table = pd.read_csv(csv_path)

    missing = [name for name in config.CLASS_NAMES if name not in table.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    paths = [image_dir / f"{image_id}.png" for image_id in table["image_id"]]
    missing_images = [path for path in paths if not path.exists()]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images are missing. Example: {missing_images[0]}"
        )

    labels = table[config.CLASS_NAMES].to_numpy(dtype=np.float32)
    return paths, labels


def compute_pos_weight(labels):
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    weights = negatives / np.clip(positives, 1, None)
    weights = np.clip(weights, None, config.MAX_POS_WEIGHT)

    return torch.tensor(weights, dtype=torch.float32)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_dataloaders(
    data_dir=config.PROCESSED_DATA_DIR,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    seed=config.SEED,
):
    train_transform, eval_transform = build_transforms()

    datasets = {}

    for split in ("train", "val", "test"):
        paths, labels = load_split(data_dir, split)
        transform = train_transform if split == "train" else eval_transform
        datasets[split] = ChestXrayDataset(paths, labels, transform)

    pos_weight = compute_pos_weight(datasets["train"].labels)

    loaders = {}

    for split, dataset in datasets.items():
        shuffle = split == "train"

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=make_generator(seed) if shuffle else None,
            drop_last=shuffle,
        )

    return loaders, pos_weight