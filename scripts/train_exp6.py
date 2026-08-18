"""
train_exp6.py

Experiment 6: NAFNet backbone + FFT loss + AdamW + EMA + geometric augmentation.

This is the highest-accuracy configuration for a <6 GB GPU setup, stacking
every improvement that has a positive expected value based on the prior
five experiments:

  Architecture : NAFNet-style UNet (SimpleGate, ChannelAttention, no BN)
  Loss         : Charbonnier + SSIM + Edge + FFT (frequency-domain)
  Optimizer    : AdamW (weight_decay=1e-4) + linear warmup + cosine LR
  Regularizer  : EMA (Exponential Moving Average) of model weights
  Augmentation : Random horizontal/vertical flip + 90°/180°/270° rotation
  Epochs       : 200 (vs 100 in Exp 5)

Why each change helps:
  NAFNet : SimpleGate avoids dead neurons; channel attention recalibrates
           feature importance; UNet multi-scale skip connections recover
           fine detail lost during downsampling.
  FFT    : Penalizes frequency-domain errors → sharper high-freq detail.
  AdamW  : Decoupled weight decay → better generalization than Adam+L2.
  EMA    : Ensemble of historical weights → consistent ~+0.3 dB PSNR
           with zero extra inference cost.
  Augment: Random flips/rotations double effective dataset size while
           being safe (degradation is spatially isotropic).

Usage:
    python train_exp6.py --data_root "D:\VSProjects\IRIS---Restoration-main\dataset\train\train" --epochs 200 --batch_size 8

Smoke test first (5 epochs):
    python train_exp6.py --data_root "D:\VSProjects\IRIS---Restoration-main\dataset\train\train" --epochs 5 --batch_size 8
"""

import argparse
import csv
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import IRISPairedDataset, make_train_val_split
from model_nafnet import NAFNetIRIS
from losses import CombinedLoss


# =============================================================================
#  Reproducibility
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
#  Geometric augmentation wrapper
# =============================================================================

class AugmentedPairedDataset(Dataset):
    """
    Wraps IRISPairedDataset and applies consistent geometric augmentation
    to BOTH noisy input and GT target.

    Augmentations (all valid for IRIS since degradation is spatially uniform):
      - Random horizontal flip (50%)
      - Random vertical flip (50%)
      - Random 90° rotation (k ∈ {0, 1, 2, 3}, 25% each)

    GT is at 256×256 and noisy is at 128×128, so the same random choice is
    applied independently to ensure consistency — rotations are applied
    before any shape-dependent ops so both tensors are consistent.
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        noisy  = sample["noisy"]   # (1, 128, 128)
        gt     = sample["gt"]      # (1, 256, 256)

        # Choose augmentation once — apply consistently to both
        hflip  = random.random() < 0.5
        vflip  = random.random() < 0.5
        k_rot  = random.randint(0, 3)         # 0=no-op, 1=90°, 2=180°, 3=270°

        if hflip:
            noisy = torch.flip(noisy, dims=[2])
            gt    = torch.flip(gt,    dims=[2])
        if vflip:
            noisy = torch.flip(noisy, dims=[1])
            gt    = torch.flip(gt,    dims=[1])
        if k_rot > 0:
            noisy = torch.rot90(noisy, k=k_rot, dims=[1, 2])
            gt    = torch.rot90(gt,    k=k_rot, dims=[1, 2])

        return {"noisy": noisy, "gt": gt, "file_id": sample["file_id"]}


# =============================================================================
#  EMA helper
# =============================================================================

class EMA:
    """
    Exponential Moving Average of model parameters.

    After training, swap to the EMA weights for evaluation and checkpointing:
        ema.apply_shadow()   # load averaged weights into model
        ... evaluate ...
        ema.restore()        # restore original weights for continued training
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = deepcopy(model.state_dict())
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        """Load EMA weights into model (for eval/checkpoint)."""
        self.backup = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)

    def restore(self):
        """Restore original training weights."""
        self.model.load_state_dict(self.backup)
        self.backup = {}


# =============================================================================
#  Metrics
# =============================================================================

@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_clipped = pred.clamp(0.0, 1.0)
    mse = torch.mean((pred_clipped - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


@torch.no_grad()
def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_clipped = pred.clamp(0.0, 1.0)
    window_size, sigma = 11, 1.5
    coords   = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g        = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g        = (g / g.sum()).unsqueeze(0)
    window   = (g.t() @ g).unsqueeze(0).unsqueeze(0).to(pred.device)
    C1, C2  = 0.01 ** 2, 0.03 ** 2
    pad     = window_size // 2

    mu_x  = F.conv2d(pred_clipped,             window, padding=pad)
    mu_y  = F.conv2d(target,                   window, padding=pad)
    s_x   = F.conv2d(pred_clipped * pred_clipped, window, padding=pad) - mu_x * mu_x
    s_y   = F.conv2d(target * target,          window, padding=pad) - mu_y * mu_y
    s_xy  = F.conv2d(pred_clipped * target,    window, padding=pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * s_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (s_x + s_y + C2)
    )
    return ssim_map.mean().item()


# =============================================================================
#  Training / validation loop
# =============================================================================

def run_epoch(model, loader, loss_fn, optimizer, device, train: bool, ema=None):
    model.train() if train else model.eval()
    total_loss  = 0.0
    total_psnr  = 0.0
    total_ssim  = 0.0
    comp_sums   = {"pixel": 0.0, "struct": 0.0, "edge": 0.0, "freq": 0.0}
    n_batches   = 0

    ctx  = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "val"

    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            noisy = batch["noisy"].to(device)
            gt    = batch["gt"].to(device)

            pred = model(noisy)
            loss, components = loss_fn(pred, gt)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # Update EMA after every optimizer step (correct approach)
                # With decay=0.999 and ~360 steps/epoch this gives a
                # meaningful half-life of ~693 steps (~2 epochs)
                if ema is not None:
                    ema.update()

            total_loss += loss.item()
            for k in comp_sums:
                comp_sums[k] += components.get(k, 0.0)
            total_psnr += compute_psnr(pred.detach(), gt)
            total_ssim += compute_ssim(pred.detach(), gt)
            n_batches  += 1

    metrics = {
        "loss":   total_loss  / n_batches,
        "psnr":   total_psnr  / n_batches,
        "ssim":   total_ssim  / n_batches,
    }
    for k in comp_sums:
        metrics[k] = comp_sums[k] / n_batches
    return metrics


# =============================================================================
#  LR schedule: linear warmup + cosine annealing
# =============================================================================

def get_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear warmup
        # cosine from warmup_epochs → total_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
#  Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train NAFNetIRIS (Experiment 6) — NAFNet + FFT loss + EMA"
    )
    parser.add_argument("--data_root",       type=str,
                        default=r"D:\VSProjects\IRIS---Restoration-main\dataset\train\train",
                        help="Path to train folder containing NoisyLR/ and GT/ subfolders")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--warmup_epochs",   type=int,   default=5)
    parser.add_argument("--ema_decay",       type=float, default=0.999)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--val_fraction",    type=float, default=0.1)
    parser.add_argument("--checkpoint_dir",  type=str,   default="checkpoints_exp6")
    parser.add_argument("--num_workers",     type=int,   default=2)
    # NAFNet hyperparams
    parser.add_argument("--width",           type=int,   default=32,
                        help="Base channel width (32 → ~17M params, safe for <6 GB GPU)")
    parser.add_argument("--middle_blks",     type=int,   default=1)
    # Loss weights
    parser.add_argument("--lambda_pixel",    type=float, default=1.0)
    parser.add_argument("--lambda_struct",   type=float, default=0.15)
    parser.add_argument("--lambda_edge",     type=float, default=0.2)
    parser.add_argument("--lambda_freq",     type=float, default=0.05)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU            : {props.name} ({props.total_memory // 1024**2} MB VRAM)")

    # --- Data ----------------------------------------------------------------
    train_base, val_set = make_train_val_split(
        args.data_root, val_fraction=args.val_fraction, seed=args.seed
    )
    train_set = AugmentedPairedDataset(train_base)
    print(f"Train pairs    : {len(train_set)} (with geometric augmentation)")
    print(f"Val pairs      : {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Model ---------------------------------------------------------------
    model = NAFNetIRIS(
        width=args.width,
        enc_blks=[1, 1, 1, 28],
        middle_blks=args.middle_blks,
        dec_blks=[1, 1, 1, 1],
    ).to(device)
    print(f"Parameters     : {model.count_parameters():,}")

    ema = EMA(model, decay=args.ema_decay)

    # --- Loss ----------------------------------------------------------------
    loss_fn = CombinedLoss(
        lambda_pixel=args.lambda_pixel,
        lambda_struct=args.lambda_struct,
        lambda_edge=args.lambda_edge,
        lambda_freq=args.lambda_freq,
    ).to(device)
    print(
        f"Loss weights   : pixel={args.lambda_pixel}, struct={args.lambda_struct}, "
        f"edge={args.lambda_edge}, freq={args.lambda_freq}"
    )

    # --- Optimizer & scheduler -----------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs)
    print(f"Optimizer      : AdamW lr={args.lr}, wd={args.weight_decay}")
    print(f"LR schedule    : {args.warmup_epochs}-epoch warmup + cosine annealing")
    print(f"EMA decay      : {args.ema_decay}")

    # --- Checkpointing -------------------------------------------------------
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "log.csv"

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "val_loss",
            "val_pixel", "val_struct", "val_edge", "val_freq",
            "val_psnr", "val_ssim",
            "val_psnr_ema", "val_ssim_ema",
            "lr", "epoch_time_sec",
        ])

    best_val_psnr = -float("inf")
    print()

    # --- Training loop -------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True, ema=ema)
        # EMA is now updated per-batch inside run_epoch — no extra call needed here

        # --- Val: normal weights ---
        val_m   = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)

        # --- Val: EMA weights ---
        ema.apply_shadow()
        val_m_ema = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        ema.restore()

        scheduler.step()

        elapsed        = time.time() - t0
        current_lr     = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_m['loss']:.5f} | "
            f"val {val_m['loss']:.5f} | "
            f"PSNR {val_m['psnr']:.2f} dB | "
            f"EMA-PSNR {val_m_ema['psnr']:.2f} dB | "
            f"SSIM {val_m_ema['ssim']:.4f} | "
            f"{elapsed:.1f}s"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_m["loss"], val_m["loss"],
                val_m["pixel"], val_m["struct"], val_m["edge"], val_m["freq"],
                val_m["psnr"], val_m["ssim"],
                val_m_ema["psnr"], val_m_ema["ssim"],
                current_lr, round(elapsed, 2),
            ])

        # Save last checkpoint (normal weights)
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(),
             "ema_state": ema.shadow, "val_psnr": val_m["psnr"], "args": vars(args)},
            ckpt_dir / "last.pt",
        )

        # Save best checkpoint (EMA weights, as they evaluate better)
        if val_m_ema["psnr"] > best_val_psnr:
            best_val_psnr = val_m_ema["psnr"]
            # Save EMA state as the model_state for easy loading in run.py
            ema.apply_shadow()
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_psnr": best_val_psnr, "args": vars(args)},
                ckpt_dir / "best.pt",
            )
            ema.restore()
            print(f"  -> new best EMA val PSNR: {best_val_psnr:.2f} dB  (checkpoint saved)")

    print()
    print(f"Training complete. Best EMA val PSNR: {best_val_psnr:.2f} dB")
    print(f"(Exp 5 baseline was: 29.07 dB)")
    print(f"Checkpoints saved to: {ckpt_dir.resolve()}")
    print(f"Log saved to        : {log_path.resolve()}")


if __name__ == "__main__":
    main()
