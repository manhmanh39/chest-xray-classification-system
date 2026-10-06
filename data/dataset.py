from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config
from utils.seed import make_generator, seed_worker

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SPLITS = ("train", "val", "test")


def build_transforms(image_size=config.IMAGE_SIZE):
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomRotation(10),
        T.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        normalize,
    ])
    evaluation = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        normalize,
    ])
    return train, evaluation


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
        return image, torch.from_numpy(self.labels[index])


def load_split(data_dir, split):
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{split}.csv"
    image_dir = data_dir / split / "images"
    if not csv_path.exists() or not image_dir.is_dir():
        raise FileNotFoundError(
            f"Processed '{split}' split not found in {data_dir}. Build it with: "
            "python -m data.prepare_dataset --zip_path <vinbigdata.zip>"
        )
    table = pd.read_csv(csv_path)
    missing_columns = [name for name in config.CLASS_NAMES if name not in table.columns]
    if missing_columns:
        raise ValueError(f"{csv_path} is missing label columns: {missing_columns}")
    paths = [image_dir / f"{image_id}.png" for image_id in table["image_id"]]
    absent = [path for path in paths if not path.exists()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} images listed in {csv_path} are missing, e.g. {absent[0]}")
    labels = table[config.CLASS_NAMES].to_numpy(dtype=np.float32)
    return paths, labels


def compute_pos_weight(labels):
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    weight = np.clip(negatives / np.clip(positives, 1, None), None, config.MAX_POS_WEIGHT)
    return torch.tensor(weight, dtype=torch.float32)


def build_datasets(data_dir=config.PROCESSED_DATA_DIR, image_size=config.IMAGE_SIZE):
    train_transform, eval_transform = build_transforms(image_size)
    datasets = {}
    for split in SPLITS:
        paths, labels = load_split(data_dir, split)
        transform = train_transform if split == "train" else eval_transform
        datasets[split] = ChestXrayDataset(paths, labels, transform)
    return datasets, compute_pos_weight(datasets["train"].labels)


def make_loader(dataset, batch_size, shuffle, num_workers, seed):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed) if shuffle else None,
        drop_last=shuffle,
    )


def get_dataloaders(data_dir=config.PROCESSED_DATA_DIR, batch_size=config.BATCH_SIZE,
                    num_workers=config.NUM_WORKERS, seed=config.SEED):
    datasets, pos_weight = build_datasets(data_dir)
    loaders = {
        split: make_loader(dataset, batch_size, split == "train", num_workers, seed)
        for split, dataset in datasets.items()
    }
    return loaders, pos_weight
