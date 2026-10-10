from pathlib import Path
import zipfile

import gdown

dataset_url = "https://drive.google.com/file/d/1qpvIWLy3OzO_c5RaapVtt9xsDf9PEF0b/view?usp=sharing"

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir / "datasets"
zip_path = root_dir / "datasets.zip"


def main():
    processed_dir = dataset_dir / "processed"

    # Skip downloading if the dataset is already prepared
    splits = ("train", "val", "test")
    if all((processed_dir / split).is_dir() for split in splits):
        print("Dataset already exists. Skipping download.")
        return

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")

    result = gdown.download(url=dataset_url, output=str(zip_path), quiet=False)

    if result is None:
        raise RuntimeError("Download failed. Check the Google Drive link and permissions.")

    print("Extracting dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    zip_path.unlink(missing_ok=True)

    print("Dataset downloaded and extracted successfully.")
    print(f"Dataset location: {dataset_dir}")


if __name__ == "__main__":
    main()