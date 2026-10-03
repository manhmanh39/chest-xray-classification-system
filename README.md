# Chest X-Ray Multi-Label Classification System

A modular deep learning framework for automated thoracic disease classification from Chest X-Ray radiographs (15 classes) using PyTorch.

## Repository Architecture

```text
chest-xray-classification-system/
│
├── models/                         # Model architectures & unified factory
│   ├── __init__.py                 # Factory API: get_model()
│   ├── model1_simple.py            # Baseline sequential CNN
│   ├── model2_complex.py           # Multi-path parallel residual CNN
│   └── transfer_model.py           # Pretrained backbone wrappers
│
├── tests/                          # Automated testing suite
│   └── test_models.py              # Unit & smoke tests for model modules
│
├── docs/                           # Technical documentation
│   └── architecture_notes.md       # CNN design specifications & mathematical formulation
│
├── data/                           # Data loading, augmentations & split routines 
│   └── .gitkeep
│
├── training/                       # Training loops, loss functions & optimizers 
│   └── .gitkeep
│
├── evaluation/                     # Metric evaluation, confusion matrix & reports
│   └── .gitkeep
│
├── utils/                          # Checkpoint managers, seed control & logging
│   └── .gitkeep
│
├── scripts/                        # Utility scripts (download, inference, hyperparameter search)
│   └── .gitkeep
│
└── README.md                       # Project overview & structure
```

## Quickstart (Model Validation)
Run smoke tests across all three model architectures:
```bash
python -m pytest -q tests/test_models.py
```
