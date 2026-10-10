# Chest X-Ray Classification System

This project is developed by Group 10 for the Deep Learning course. It focuses on multi-label classification of chest X-ray images using the VinBigData Chest X-ray dataset.

The project implements and compares three models: Simple CNN, Complex CNN, and Transfer Learning.

## Group Members

* Nguyễn Thị Mai Anh
* Phạm Thị Ngọc Ánh
* Phạm Khánh Dương
* Nguyễn Khánh Huyền

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/manhmanh39/chest-xray-classification-system.git
cd chest-xray-classification-system
```

### 2. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you use a virtual environment, activate it before installing the dependencies.

## Dataset

This project uses the VinBigData Chest X-ray dataset for multi-label classification of 14 thoracic diseases and the `No finding` class.

### 1. Download the prepared dataset

Run the following command from the project root directory:

```bash
python download_dataset.py
```

The script downloads the prepared dataset from [Google Drive](https://drive.google.com/file/d/1qpvIWLy3OzO_c5RaapVtt9xsDf9PEF0b/view?usp=sharing) and extracts it into the `datasets/` directory.

The expected directory structure is:

```text
datasets/
├── raw/
│   └── train.csv
└── processed/
    ├── train/
    ├── val/
    ├── test/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── split_summary.csv
```

Make sure the dataset has been downloaded and extracted successfully before training.

### 2. Prepare the dataset manually

Alternatively, download the original dataset from [Kaggle](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data).

Place the original dataset files in the appropriate directories under `datasets/`. If you have the original training ZIP file, run:

```bash
python generate_data.py --zip_path path/to/train.zip
```

Replace `path/to/train.zip` with the actual path to the ZIP file. The script generates the processed dataset and splits the data into training, validation, and test sets.

If you use the prepared dataset from Google Drive, you do not need to generate the dataset again unless you want to recreate the splits or preprocessing outputs.

## Models

The project includes three models:

* **Simple CNN:** A basic convolutional neural network with convolutional and pooling layers.
* **Complex CNN:** A CNN with parallel branches for feature extraction.
* **Transfer Learning:** A model that fine-tunes a pretrained DenseNet121 backbone.

## Training

Run the following command from the project root directory to train a model:

```bash
python train.py --model simple
```

The supported model options are:

* `simple`
* `complex`
* `transfer`

For example, to train the Complex CNN:

```bash
python train.py --model complex
```

To train the Transfer Learning model:

```bash
python train.py --model transfer
```

You can customize training settings using command-line arguments. For example:

```bash
python train.py --model complex --epochs 30 --batch_size 16 --lr 0.0005
```

Run the following command to see the available arguments:

```bash
python train.py --help
```

## Evaluation

Evaluate a trained model using the evaluation script:

```bash
python evaluate.py --checkpoint checkpoints/complex_best.pth
```

Replace the checkpoint path with the checkpoint you want to evaluate.

The project uses the following evaluation metrics:

* **AUC:** Area Under the ROC Curve
* **F1-score:** Macro and micro F1-score
* **Sensitivity:** Proportion of positive cases correctly identified
* **Specificity:** Proportion of negative cases correctly identified
* **Label accuracy:** Accuracy calculated across individual labels
* **Subset accuracy:** Proportion of samples for which all labels are predicted correctly

The default classification threshold is `0.5`.

## Project Structure

```text
chest-xray-classification-system/
├── checkpoints/              # Saved model checkpoints
├── datasets/
│   ├── raw/                   # Original dataset files
│   └── processed/             # Processed images and dataset splits
├── models/                    # Model architectures
├── outputs/                   # Evaluation results and experiment outputs
├── scripts/                   # Helper and experiment scripts
├── .gitignore
├── config.py                  # Project configuration
├── download_dataset.py        # Downloads the prepared dataset
├── evaluate.py                # Evaluates trained models
├── generate_data.py           # Generates and splits the dataset
├── preprocessing.py           # Data preprocessing utilities
├── requirements.txt           # Python dependencies
├── train.py                   # Model training entry point
└── README.md
```

## Notes

* Run commands from the project root directory.
* Ensure the dataset is available in the expected directories before training or evaluation.
* Check `config.py` for the default dataset paths, class names, and training configuration.
* The `checkpoints/` and `outputs/` directories store model checkpoints and experiment results.
* The actual dataset files and trained checkpoints may not be included in the Git repository.
