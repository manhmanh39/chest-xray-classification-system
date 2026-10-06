import argparse
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from tqdm import tqdm

try:
    from pydicom.pixels import apply_voi_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_voi_lut

import config


def build_label_table(csv_path):
    annotations = pd.read_csv(csv_path)
    labels = pd.crosstab(annotations["image_id"], annotations["class_id"]).clip(upper=1)
    labels = labels.reindex(columns=range(config.NUM_CLASSES), fill_value=0)
    has_finding = labels.iloc[:, : config.NO_FINDING_CLASS_ID].sum(axis=1) > 0
    labels[config.NO_FINDING_CLASS_ID] = (~has_finding).astype(int)
    labels.columns = config.CLASS_NAMES
    return labels.reset_index()


def stratified_split(labels, val_ratio, test_ratio, seed):
    targets = labels[config.CLASS_NAMES].to_numpy()
    indices = np.arange(len(labels))
    holdout = val_ratio + test_ratio

    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    train_idx, holdout_idx = next(first.split(indices, targets))

    second = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio / holdout, random_state=seed
    )
    val_rel, test_rel = next(second.split(holdout_idx, targets[holdout_idx]))
    return {
        "train": labels.iloc[train_idx],
        "val": labels.iloc[holdout_idx[val_rel]],
        "test": labels.iloc[holdout_idx[test_rel]],
    }


def dicom_to_png(task):
    image_id, zip_path, out_path, image_size, overwrite = task
    if out_path.exists() and not overwrite:
        return image_id, True
    try:
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(f"train/{image_id}.dicom") as handle:
                dicom = pydicom.dcmread(io.BytesIO(handle.read()))
        pixels = apply_voi_lut(dicom.pixel_array, dicom).astype(np.float32)
        if str(getattr(dicom, "PhotometricInterpretation", "")) == "MONOCHROME1":
            pixels = pixels.max() - pixels
        low, high = np.percentile(pixels, [0.5, 99.5])
        if high <= low:
            low, high = pixels.min(), pixels.max()
        scaled = np.clip((pixels - low) / max(high - low, 1e-6), 0.0, 1.0)
        image = cv2.resize(
            (scaled * 255).astype(np.uint8),
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(out_path), image)
        return image_id, True
    except Exception as error:
        print(f"[prepare] failed on {image_id}: {error}")
        return image_id, False


def remove_stale_images(image_dir, keep_ids):
    for path in image_dir.glob("*.png"):
        if path.stem not in keep_ids:
            path.unlink()


def summarize(frames):
    summary = pd.DataFrame(
        {name: frame[config.CLASS_NAMES].sum() for name, frame in frames.items()}
    )
    summary["total"] = summary.sum(axis=1)
    return summary


def prepare_dataset(zip_path, csv_path, output_dir, image_size, val_ratio, test_ratio,
                    seed, num_workers, overwrite):
    zip_path, csv_path, output_dir = Path(zip_path), Path(csv_path), Path(output_dir)
    for required in (zip_path, csv_path):
        if not required.exists():
            raise FileNotFoundError(f"File not found: {required}")

    labels = build_label_table(csv_path)
    with zipfile.ZipFile(zip_path) as archive:
        available = {
            Path(name).stem
            for name in archive.namelist()
            if name.startswith("train/") and name.endswith(".dicom")
        }
    labels = labels[labels["image_id"].isin(available)].reset_index(drop=True)
    print(f"[prepare] {len(labels)} images with labels and DICOM files")

    frames = stratified_split(labels, val_ratio, test_ratio, seed)
    kept = {}
    for split, frame in frames.items():
        image_dir = output_dir / split / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        remove_stale_images(image_dir, set(frame["image_id"]))

        tasks = [
            (image_id, str(zip_path), image_dir / f"{image_id}.png", image_size, overwrite)
            for image_id in frame["image_id"]
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(tqdm(pool.map(dicom_to_png, tasks, chunksize=16),
                                total=len(tasks), desc=split))
        converted = {image_id for image_id, ok in results if ok}
        kept[split] = frame[frame["image_id"].isin(converted)]
        kept[split].to_csv(output_dir / f"{split}.csv", index=False)

    summary = summarize(kept)
    summary.to_csv(output_dir / "split_summary.csv")
    print(summary.to_string())
    print(f"[prepare] done: {output_dir.resolve()}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the train/val/test split from VinBigData DICOMs")
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--csv_path", default=str(config.RAW_DATA_DIR / "train.csv"))
    parser.add_argument("--output_dir", default=str(config.PROCESSED_DATA_DIR))
    parser.add_argument("--image_size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument("--val_ratio", type=float, default=config.VAL_RATIO)
    parser.add_argument("--test_ratio", type=float, default=config.TEST_RATIO)
    parser.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prepare_dataset(
        args.zip_path, args.csv_path, args.output_dir, args.image_size,
        args.val_ratio, args.test_ratio, args.seed, args.num_workers, args.overwrite,
    )


if __name__ == "__main__":
    main()
