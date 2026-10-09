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

First, download the project from GitHub:

```bash
git clone https://github.com/manhmanh39/chest-xray-classification-system.git
cd chest-xray-classification-system
```

### 2. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment, activate it before installing the dependencies.

## Dataset

This project uses the VinBigData Chest X-ray dataset for multi-label classification of 14 thoracic diseases and the "No Finding" class.

You can download the prepared dataset automatically using the provided script.

### 1. Download the dataset

Run the following command from the project root directory:

```bash
python download_dataset.py
```

The script downloads the dataset ZIP file from [Google Drive](https://drive.google.com/file/d/1qpvIWLy3OzO_c5RaapVtt9xsDf9PEF0b/view?usp=sharing) and extracts the files into the `datasets/` directory.

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

Alternatively, you can download the original dataset from [Kaggle](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data).

Place the original `train.csv` file in `datasets/raw/`, then run:

```bash
python generate_data.py --zip_path path/to/train.zip
```

Replace `path/to/train.zip` with the actual path to the original dataset ZIP file. The script processes the DICOM images and creates the training, validation, and test sets.

**Note:** If you use the prepared dataset from Google Drive, you do not need to run the preprocessing script again.

## Models

The project includes three models:

* **Simple CNN:** A basic convolutional neural network with convolutional and pooling layers.
* **Complex CNN:** A CNN with parallel branches for feature extraction.
* **Transfer Learning:** A model that uses a pretrained network as its backbone.

## Training

After downloading the dataset, you can train a model using:

```bash
python main.py --model simple
```

Replace `simple` with another supported model name to train a different model.

Training settings, including the learning rate, batch size, number of epochs, and model configuration, can be adjusted using the available configuration options.

## Evaluation

The models are evaluated using classification metrics, including:

* AUC (Area Under the ROC Curve)
* F1-score
* Sensitivity
* Specificity
* Accuracy

These metrics are used to compare the performance of the models on the chest X-ray classification task.

## Project Structure

```text
chest-xray-classification-system/
├── models/
├── data/
├── training/
├── evaluation/
├── scripts/
├── utils/
├── tests/
├── docs/
├── datasets/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── download_dataset.py
├── generate_data.py
├── main.py
├── requirements.txt
└── README.md
```

The project contains the model implementations, data processing utilities, training and evaluation scripts, tests, and documentation.

## Notes

* Download the repository before running the dataset download script.
* Make sure the dataset is available in the expected directory before training.
* The dataset ZIP file does not need to be downloaded manually if `download_dataset.py` is configured correctly.
