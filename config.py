import os
from pathlib import Path

import torch

base_dir = Path(__file__).resolve().parent
raw_data_dir = base_dir / "datasets" / "raw"
processed_data_dir = base_dir / "datasets" / "processed"
checkpoint_dir = base_dir / "checkpoints"
output_dir = base_dir / "outputs"

class_names = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis", "No finding",
]

num_classes = len(class_names)
no_finding_class_id = class_names.index("No finding")
image_size = 224
batch_size = 16
num_workers = min(20, os.cpu_count() or 1)
epochs = 30
learning_rate = 1e-3
weight_decay = 1e-4
threshold = 0.5
max_pos_weight = 30.0
freeze_epochs = 3
unfreeze_lr = 1e-5
seed = 202601
split_seed = 2026
val_ratio = 0.1
test_ratio = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")