import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from utils.metrics import calculate_binary_metrics, enhanced_measure, structure_measure, weighted_f_measure


DEFAULT_DATASETS = [
    "CVC-300",
    "CVC-ClinicDB",
    "Kvasir",
    "CVC-ColonDB",
    "ETIS-LaribPolypDB",
]

METRIC_HEADERS = [
    "meanDice",
    "meanIoU",
    "wFm",
    "Sm",
    "meanEm",
    "mae",
    "maxEm",
    "maxDice",
    "maxIoU",
    "meanSen",
    "maxSen",
    "meanSpe",
    "maxSpe",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=str, default="./results/sca_net")
    parser.add_argument("--gt-root", type=str, default="./data/TestDataset")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--method-name", type=str, default="SCA-Net")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    return parser.parse_args()


def _format_cell(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _format_table(rows: list[list[object]], headers: list[str]) -> str:
    table = [headers] + rows
    widths = [max(len(_format_cell(row[index])) for row in table) for index in range(len(headers))]
    lines = []

    for row_index, row in enumerate(table):
        cells = [_format_cell(cell).ljust(widths[index]) for index, cell in enumerate(row)]
        lines.append(" | ".join(cells))
        if row_index == 0:
            lines.append("-+-".join("-" * width for width in widths))

    return "\n".join(lines)


def evaluate_dataset(pred_root: Path, gt_root: Path, device: torch.device) -> list[float]:
    pred_files = sorted(os.listdir(pred_root))
    gt_files = sorted(os.listdir(gt_root))
    thresholds = torch.linspace(1, 0, 256, device=device)

    threshold_f = torch.zeros((len(pred_files), len(thresholds)), device=device)
    threshold_e = torch.zeros_like(threshold_f)
    threshold_iou = torch.zeros_like(threshold_f)
    threshold_sen = torch.zeros_like(threshold_f)
    threshold_spe = torch.zeros_like(threshold_f)
    threshold_dice = torch.zeros_like(threshold_f)

    structure_scores = torch.zeros(len(pred_files), device=device)
    weighted_f_scores = torch.zeros(len(pred_files), device=device)
    mae_scores = torch.zeros(len(pred_files), device=device)

    for index, (pred_name, gt_name) in enumerate(zip(pred_files, gt_files)):
        prediction = torch.tensor(
            np.array(Image.open(pred_root / pred_name)),
            dtype=torch.float32,
            device=device,
        )
        target = torch.tensor(
            np.array(Image.open(gt_root / gt_name)),
            dtype=torch.float32,
            device=device,
        )

        if prediction.dim() == 3:
            prediction = prediction[..., 0]
        if target.dim() == 3:
            target = target[..., 0]

        target = ((target / 255.0) > 0.5).float()
        prediction = prediction / 255.0

        structure_scores[index] = structure_measure(prediction, target)
        weighted_f_scores[index] = weighted_f_measure(prediction, target)
        mae_scores[index] = torch.mean(torch.abs(target - prediction))

        for threshold_index, threshold in enumerate(thresholds):
            precision, recall, specificity, dice, f_measure, iou = calculate_binary_metrics(
                prediction,
                target,
                threshold.item(),
            )
            threshold_f[index, threshold_index] = f_measure
            threshold_e[index, threshold_index] = enhanced_measure((prediction >= threshold).float(), target)
            threshold_iou[index, threshold_index] = iou
            threshold_sen[index, threshold_index] = recall
            threshold_spe[index, threshold_index] = specificity
            threshold_dice[index, threshold_index] = dice

    return [
        threshold_dice.mean(dim=0).mean().item(),
        threshold_iou.mean(dim=0).mean().item(),
        weighted_f_scores.mean().item(),
        structure_scores.mean().item(),
        threshold_e.mean(dim=0).mean().item(),
        mae_scores.mean().item(),
        threshold_e.mean(dim=0).max().item(),
        threshold_dice.mean(dim=0).max().item(),
        threshold_iou.mean(dim=0).max().item(),
        threshold_sen.mean(dim=0).mean().item(),
        threshold_sen.mean(dim=0).max().item(),
        threshold_spe.mean(dim=0).mean().item(),
        threshold_spe.mean(dim=0).max().item(),
    ]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    save_root = Path(args.save_dir) if args.save_dir else pred_root
    save_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset_name in args.datasets:
        dataset_pred_root = pred_root / dataset_name
        dataset_gt_root = gt_root / dataset_name / "masks"
        if not dataset_pred_root.exists():
            print(f"Skipping {dataset_name}: {dataset_pred_root} not found.")
            continue

        metrics = evaluate_dataset(dataset_pred_root, dataset_gt_root, device)
        rows.append([dataset_name] + metrics)

        with open(save_root / f"result_{dataset_name}.csv", "a", newline="") as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(["method"] + METRIC_HEADERS)
            writer.writerow([args.method_name] + [f"{value:.4f}" for value in metrics])

    print(_format_table(rows, ["dataset"] + METRIC_HEADERS))


if __name__ == "__main__":
    main()
