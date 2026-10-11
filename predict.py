import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import config
from preprocessing import build_transforms
from train import create_model


def predict_image(checkpoint_path, image_path, threshold):
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model = create_model(
        checkpoint["model_name"], checkpoint["dropout"], pretrained=False
    )
    model.load_state_dict(checkpoint["model"])
    model.to(config.device)
    model.eval()

    _, eval_transform = build_transforms()
    with Image.open(image_path) as image:
        image_tensor = eval_transform(image.convert("RGB")).unsqueeze(0)
    image_tensor = image_tensor.to(config.device)

    with torch.inference_mode():
        probabilities = torch.sigmoid(model(image_tensor))[0].cpu().tolist()

    class_probabilities = dict(zip(config.class_names, probabilities))
    predicted_labels = [
        name for name, probability in class_probabilities.items()
        if probability >= threshold
    ]
    return {
        "image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "predicted_labels": predicted_labels,
        "probabilities": class_probabilities,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict chest X-ray labels using a saved PyTorch checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=config.threshold)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    result = predict_image(args.checkpoint, args.image, args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()