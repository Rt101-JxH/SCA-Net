import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from lib.sca_net import SCANet
from utils.data import InferenceDataset


DEFAULT_DATASETS = [
    "CVC-300",
    "CVC-ClinicDB",
    "Kvasir",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/sca_net/epoch_100.pth")
    parser.add_argument("--data-root", type=str, default="./data/TestDataset")
    parser.add_argument("--save-dir", type=str, default="./results/sca_net")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    return parser.parse_args()


def load_checkpoint_file(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(checkpoint_path: str, device: torch.device) -> SCANet:
    model = SCANet(pretrained_backbone=False).to(device)
    state_dict = load_checkpoint_file(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def normalize_prediction(prediction: np.ndarray) -> np.ndarray:
    min_value = prediction.min()
    max_value = prediction.max()
    normalized = (prediction - min_value) / (max_value - min_value + 1e-8)
    return (normalized * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    data_root = Path(args.data_root)
    save_root = Path(args.save_dir)

    for dataset_name in args.datasets:
        image_root = data_root / dataset_name / "images"
        mask_root = data_root / dataset_name / "masks"
        if not image_root.exists():
            print(f"Skipping {dataset_name}: {image_root} not found.")
            continue

        dataset = InferenceDataset(
            image_root=image_root,
            image_size=args.image_size,
            mask_root=mask_root if mask_root.exists() else None,
        )
        dataset_save_root = save_root / dataset_name
        dataset_save_root.mkdir(parents=True, exist_ok=True)

        print(f"Predicting on {dataset_name}...")
        for image_tensor, original_size, output_name in dataset:
            image_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(image_tensor)[0]
                prediction = F.interpolate(
                    prediction,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
                prediction = torch.sigmoid(prediction).squeeze().cpu().numpy()

            Image.fromarray(normalize_prediction(prediction)).save(dataset_save_root / output_name)


if __name__ == "__main__":
    main()
