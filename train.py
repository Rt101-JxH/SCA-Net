import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from lib.sca_net import SCANet
from utils.data import create_train_loader
from utils.losses import DeepSupervisionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-power", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-norm", type=float, default=0.5)
    parser.add_argument("--train-path", type=str, default="./data/TrainDataset")
    parser.add_argument("--save-dir", type=str, default="./checkpoints/sca_net")
    parser.add_argument("--seed", type=int, default=725)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.amp.GradScaler(enabled=use_amp)


def get_autocast_context(use_amp: bool):
    if not use_amp:
        return nullcontext
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):

        def autocast_context():
            try:
                return torch.amp.autocast(device_type="cuda")
            except TypeError:
                return torch.amp.autocast("cuda")

        return autocast_context
    return torch.amp.autocast


def build_size_labels(mask: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    batch_size = mask.shape[0]
    foreground_ratio = mask.view(batch_size, -1).mean(dim=1)
    return (foreground_ratio > threshold).float().unsqueeze(1)


def save_checkpoint(path: Path, model: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    segmentation_criterion: DeepSupervisionLoss,
    routing_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
    grad_norm: float,
    accumulation_steps: int,
    use_amp: bool,
) -> None:
    model.train()
    scaler = create_grad_scaler(use_amp)
    autocast_context = get_autocast_context(use_amp)

    running_total = 0.0
    running_segmentation = 0.0
    running_routing = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        size_labels = build_size_labels(masks).to(device)

        with autocast_context():
            output1, output2, output3, output4, output5, routing_logits = model(images)
            main_loss, aux1_loss, aux2_loss, aux3_loss, aux4_loss, _ = segmentation_criterion(
                (output1, output2, output3, output4, output5), masks
            )
            segmentation_loss = main_loss + aux1_loss + aux2_loss + aux3_loss + aux4_loss
            routing_loss = routing_criterion(routing_logits, size_labels)
            loss = segmentation_loss + routing_loss

        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()

        should_step = step % accumulation_steps == 0 or step == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_total += loss.item()
        running_segmentation += segmentation_loss.item()
        running_routing += routing_loss.item()

        if step % 20 == 0 or step == len(loader):
            mean_total = running_total / step
            mean_segmentation = running_segmentation / step
            mean_routing = running_routing / step
            print(
                f"{datetime.now()} Epoch [{epoch:03d}/{epochs:03d}] "
                f"Step [{step:04d}/{len(loader):04d}] "
                f"loss={mean_total:.4f} seg={mean_segmentation:.4f} route={mean_routing:.4f}"
            )

    scheduler.step()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    accumulation_steps = max(1, args.accumulation_steps)

    model = SCANet(pretrained_backbone=args.pretrained_backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda current_epoch: 1.0 - (current_epoch / max(1, args.epochs)) ** args.lr_power,
    )
    segmentation_criterion = DeepSupervisionLoss()
    routing_criterion = nn.BCEWithLogitsLoss()

    train_loader = create_train_loader(
        image_root=Path(args.train_path) / "images",
        mask_root=Path(args.train_path) / "masks",
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        augment=True,
    )

    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * accumulation_steps}")
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters())}")

    save_dir = Path(args.save_dir)
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            segmentation_criterion=segmentation_criterion,
            routing_criterion=routing_criterion,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            grad_norm=args.grad_norm,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
        )

    save_checkpoint(save_dir / f"epoch_{args.epochs}.pth", model)
    print(f"Final checkpoint saved to: {save_dir / f'epoch_{args.epochs}.pth'}")


if __name__ == "__main__":
    main()
