"""
train.py

Phase 2 training script: trains IRISBaseline on the paired NoisyLR/GT
data using Charbonnier loss (a smooth L1 variant, standard in SR/
restoration work -- avoids the non-smooth gradient of pure L1 at 0
while staying more robust to outliers than MSE).

Per the roadmap, this stays intentionally simple:
    - single loss term (pixel-level Charbonnier)
    - no degradation encoder, no adaptive routing, no augmentation simulator
    - fixed seed, fixed train/val split (from dataset.py) so every later
      experiment (Phase 3+) is compared against the SAME validation set

Usage:
    python train.py --data_root "C:\\Users\\sekuh\\Desktop\\semicon\\train" --epochs 30

Outputs:
    checkpoints/best.pt        -- best val PSNR checkpoint
    checkpoints/last.pt        -- most recent epoch checkpoint
    checkpoints/log.csv        -- per-epoch train/val loss + PSNR/SSIM
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import make_train_val_split
from model import IRISBaseline
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """sqrt((pred - target)^2 + eps^2), averaged over all elements."""
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    PSNR computed on CLIPPED predictions (clip to [0,1] to match GT's
    valid range) -- this is evaluation-time clipping only, never
    applied during training/backprop. data_range=1.0 since GT is [0,1].
    """
    pred_clipped = pred.clamp(0.0, 1.0)
    mse = torch.mean((pred_clipped - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


@torch.no_grad()
def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Lightweight single-scale SSIM (Gaussian window), computed on clipped
    predictions, averaged over the batch. Implemented directly (no extra
    dependency) using a fixed 11x11 Gaussian window -- standard SSIM
    settings (C1=(0.01)^2, C2=(0.03)^2 for data_range=1.0).
    """
    pred_clipped = pred.clamp(0.0, 1.0)

    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0).to(pred.device)  # (1,1,11,11)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    mu_x = torch.nn.functional.conv2d(pred_clipped, window_2d, padding=pad)
    mu_y = torch.nn.functional.conv2d(target, window_2d, padding=pad)

    mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x_sq = torch.nn.functional.conv2d(pred_clipped * pred_clipped, window_2d, padding=pad) - mu_x_sq
    sigma_y_sq = torch.nn.functional.conv2d(target * target, window_2d, padding=pad) - mu_y_sq
    sigma_xy = torch.nn.functional.conv2d(pred_clipped * target, window_2d, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return ssim_map.mean().item()


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, total_psnr, total_ssim, n_batches = 0.0, 0.0, 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "val"
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)

            pred = model(noisy)
            loss = charbonnier_loss(pred, gt)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_psnr += compute_psnr(pred.detach(), gt)
            total_ssim += compute_ssim(pred.detach(), gt)
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "psnr": total_psnr / n_batches,
        "ssim": total_ssim / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train IRISBaseline (Phase 2)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_set, val_set = make_train_val_split(
        args.data_root, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"Train pairs: {len(train_set)} | Val pairs: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = IRISBaseline(channels=64, num_res_blocks=8).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_psnr", "val_ssim",
            "lr", "epoch_time_sec",
        ])

    best_val_psnr = -float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step()

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss {train_metrics['loss']:.5f} | "
            f"val_loss {val_metrics['loss']:.5f} | "
            f"val_psnr {val_metrics['psnr']:.2f} dB | "
            f"val_ssim {val_metrics['ssim']:.4f} | "
            f"{epoch_time:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_metrics["loss"], val_metrics["loss"],
                val_metrics["psnr"], val_metrics["ssim"],
                current_lr, round(epoch_time, 2),
            ])

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(),
             "val_psnr": val_metrics["psnr"], "args": vars(args)},
            checkpoint_dir / "last.pt",
        )

        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_psnr": val_metrics["psnr"], "args": vars(args)},
                checkpoint_dir / "best.pt",
            )
            print(f"  -> new best val PSNR: {best_val_psnr:.2f} dB (checkpoint saved)")

    print()
    print(f"Training complete. Best val PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoints saved to: {checkpoint_dir.resolve()}")
    print(f"Log saved to: {log_path.resolve()}")


if __name__ == "__main__":
    main()