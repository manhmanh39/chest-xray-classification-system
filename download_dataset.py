
from pathlib import Path
import zipfile
import gdown

DATASET_URL = "https://drive.google.com/file/d/1qpvIWLy3OzO_c5RaapVtt9xsDf9PEF0b/view?usp=sharing"

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "datasets"
ZIP_PATH = ROOT_DIR / "datasets.zip"


def main():
    processed_dir = DATASET_DIR / "processed"

    # Skip downloading if the dataset is already prepared
    if (
        (processed_dir / "train").is_dir()
        and (processed_dir / "val").is_dir()
        and (processed_dir / "test").is_dir()
    ):
        print("Dataset already exists. Skipping download.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset...")

    result = gdown.download(
        url=DATASET_URL,
        output=str(ZIP_PATH),
        quiet=False,
    )

    if result is None:
        raise RuntimeError(
            "Download failed. Check the Google Drive link and permissions."
        )

    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    ZIP_PATH.unlink(missing_ok=True)

    print("Dataset downloaded and extracted successfully.")
    print(f"Dataset location: {DATASET_DIR}")


if __name__ == "__main__":
    main()