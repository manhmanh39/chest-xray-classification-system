import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "datasets" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "datasets" / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs"

CLASS_NAMES = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening",
    "Pneumothorax", "Pulmonary fibrosis", "No finding",
]
NUM_CLASSES = len(CLASS_NAMES)
NO_FINDING_CLASS_ID = CLASS_NAMES.index("No finding")

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = min(20, os.cpu_count() or 1)
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
MAX_POS_WEIGHT = 30.0

FREEZE_EPOCHS = 3
UNFREEZE_LR = 1e-5

SEED = 202601
SPLIT_SEED = 2026
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
