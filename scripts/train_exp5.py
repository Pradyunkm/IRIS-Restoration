"""
train_exp5.py

Experiment 5: IRISConditioned (degradation-aware FiLM conditioning) +
CombinedLoss, trained on REAL PAIRS ONLY (matching Experiment 3's exact
setup: same loss, same data, same optimizer/schedule). This isolates
degradation-aware conditioning as the single new variable against
Experiment 3.

Usage:
    python train_exp5.py --data_root "C:\\Users\\sekuh\\Desktop\\semicon\\train" --epochs 100 --batch_size 8

Recommended first step: run a short smoke test (--epochs 5) to confirm
training is stable (loss decreasing, no NaNs) before committing to a
full 100-epoch run, since this is a new, more complex architecture.
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_train_val_split
from model_conditioned import IRISConditioned
from losses import CombinedLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_clipped = pred.clamp(0.0, 1.0)
    mse = torch.mean((pred_clipped - target) ** 2).item()
    if mse == 0:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


@torch.no_grad()
def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_clipped = pred.clamp(0.0, 1.0)
    window_size, sigma = 11, 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0).to(pred.device)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
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


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, total_psnr, total_ssim, n_batches = 0.0, 0.0, 0.0, 0
    total_pixel, total_struct, total_edge = 0.0, 0.0, 0.0

    context = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "val"
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)

            pred = model(noisy)
            loss, components = loss_fn(pred, gt)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_pixel += components["pixel"]
            total_struct += components["struct"]
            total_edge += components["edge"]
            total_psnr += compute_psnr(pred.detach(), gt)
            total_ssim += compute_ssim(pred.detach(), gt)
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "pixel": total_pixel / n_batches,
        "struct": total_struct / n_batches,
        "edge": total_edge / n_batches,
        "psnr": total_psnr / n_batches,
        "ssim": total_ssim / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train IRISConditioned (Experiment 5)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_exp5")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=112)
    parser.add_argument("--num_res_blocks", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--lambda_pixel", type=float, default=1.0)
    parser.add_argument("--lambda_struct", type=float, default=0.2)
    parser.add_argument("--lambda_edge", type=float, default=0.3)
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

    model = IRISConditioned(
        channels=args.channels, num_res_blocks=args.num_res_blocks, embed_dim=args.embed_dim
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    loss_fn = CombinedLoss(
        lambda_pixel=args.lambda_pixel,
        lambda_struct=args.lambda_struct,
        lambda_edge=args.lambda_edge,
    ).to(device)
    print(f"Loss weights: pixel={args.lambda_pixel}, struct={args.lambda_struct}, edge={args.lambda_edge}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_pixel", "val_struct", "val_edge",
            "val_psnr", "val_ssim", "lr", "epoch_time_sec",
        ])

    best_val_psnr = -float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
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
                val_metrics["pixel"], val_metrics["struct"], val_metrics["edge"],
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
    print(f"(Exp 3, no conditioning, was: 26.62 dB batch-pooled / 29.05 dB clean per-image)")
    print(f"Checkpoints saved to: {checkpoint_dir.resolve()}")
    print(f"Log saved to: {log_path.resolve()}")


if __name__ == "__main__":
    main()