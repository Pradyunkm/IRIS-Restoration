"""
auto_retrain.py

Automated retraining pipeline for the IRIS IRISConditioned restoration model.

This script:
  1. Watches a staging folder (new_data/) for new paired NoisyLR + GT files.
  2. Moves matched pairs into the training dataset.
  3. Fine-tunes the IRISConditioned model from the current best checkpoint.
  4. Evaluates the new model on the validation set.
  5. Replaces the production model ONLY if the new model has higher PSNR.
  6. Archives old models and logs every cycle to retrain_history.csv.

Usage:
    # Daemon mode — watches continuously
    python auto_retrain.py

    # One-shot — run once then exit
    python auto_retrain.py --once

    # Dry run — simulate without training
    python auto_retrain.py --dry_run

    # Custom settings
    python auto_retrain.py --retrain_epochs 30 --poll_interval 120 --lr 1e-5
"""

import argparse
import csv
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IRISPairedDataset, make_train_val_split
from model_conditioned import IRISConditioned
from losses import CombinedLoss


# =============================================================================
#  Constants & defaults
# =============================================================================

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_ROOT      = PROJECT_ROOT / "dataset" / "train" / "train"
NEW_DATA_DIR   = PROJECT_ROOT / "new_data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_exp5"
ARCHIVE_DIR    = CHECKPOINT_DIR / "archive"
PRODUCTION_DIR = PROJECT_ROOT / "Team-LML" / "models"
HISTORY_CSV    = CHECKPOINT_DIR / "retrain_history.csv"


# =============================================================================
#  Reproducibility
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    C1, C2   = 0.01 ** 2, 0.03 ** 2
    pad      = window_size // 2

    mu_x  = F.conv2d(pred_clipped, window, padding=pad)
    mu_y  = F.conv2d(target, window, padding=pad)
    s_x   = F.conv2d(pred_clipped * pred_clipped, window, padding=pad) - mu_x * mu_x
    s_y   = F.conv2d(target * target, window, padding=pad) - mu_y * mu_y
    s_xy  = F.conv2d(pred_clipped * target, window, padding=pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * s_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (s_x + s_y + C2)
    )
    return ssim_map.mean().item()


# =============================================================================
#  Training / validation loop
# =============================================================================

def run_epoch(model, loader, loss_fn, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches  = 0

    ctx  = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "val"

    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            noisy = batch["noisy"].to(device)
            gt    = batch["gt"].to(device)

            pred = model(noisy)
            loss, _ = loss_fn(pred, gt)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_psnr += compute_psnr(pred.detach(), gt)
            total_ssim += compute_ssim(pred.detach(), gt)
            n_batches  += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "psnr": total_psnr / max(n_batches, 1),
        "ssim": total_ssim / max(n_batches, 1),
    }


# =============================================================================
#  Data watcher — detect new paired files
# =============================================================================

def find_new_pairs(new_data_dir: Path) -> list:
    """
    Scan new_data/NoisyLR/ and new_data/GT/ for matching .npy filenames.
    Returns a sorted list of file stems that have a match in both directories.
    """
    noisy_dir = new_data_dir / "NoisyLR"
    gt_dir    = new_data_dir / "GT"

    if not noisy_dir.is_dir() or not gt_dir.is_dir():
        return []

    noisy_stems = {p.stem for p in noisy_dir.glob("*.npy") if not p.name.startswith("._")}
    gt_stems    = {p.stem for p in gt_dir.glob("*.npy") if not p.name.startswith("._")}
    matched     = sorted(noisy_stems & gt_stems)
    return matched


def ingest_new_data(new_data_dir: Path, data_root: Path, file_stems: list) -> int:
    """
    Move matched pairs from new_data/ into the training dataset.
    Returns the number of pairs successfully ingested.
    """
    src_noisy = new_data_dir / "NoisyLR"
    src_gt    = new_data_dir / "GT"
    dst_noisy = data_root / "NoisyLR"
    dst_gt    = data_root / "GT"

    ingested = 0
    for stem in file_stems:
        noisy_src = src_noisy / f"{stem}.npy"
        gt_src    = src_gt / f"{stem}.npy"
        noisy_dst = dst_noisy / f"{stem}.npy"
        gt_dst    = dst_gt / f"{stem}.npy"

        if noisy_dst.exists() or gt_dst.exists():
            print(f"  [SKIP] {stem}.npy already exists in training set")
            continue

        try:
            noisy_arr = np.load(noisy_src)
            gt_arr    = np.load(gt_src)
            if noisy_arr.shape != (128, 128):
                print(f"  [SKIP] {stem}.npy NoisyLR has unexpected shape {noisy_arr.shape}")
                continue
            if gt_arr.shape != (256, 256):
                print(f"  [SKIP] {stem}.npy GT has unexpected shape {gt_arr.shape}")
                continue
        except Exception as e:
            print(f"  [SKIP] {stem}.npy failed validation: {e}")
            continue

        shutil.move(str(noisy_src), str(noisy_dst))
        shutil.move(str(gt_src),    str(gt_dst))
        ingested += 1

    return ingested


# =============================================================================
#  Load current best checkpoint
# =============================================================================

def load_best_checkpoint(checkpoint_dir: Path, device: torch.device):
    """
    Load best.pt and return (model, best_psnr, saved_args).
    Returns (None, -inf, {}) if no checkpoint found.
    """
    best_path = checkpoint_dir / "best.pt"
    if not best_path.exists():
        print("  [WARN] No best.pt found — will train from scratch")
        return None, -float("inf"), {}

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    best_psnr  = ckpt.get("val_psnr", -float("inf"))

    channels       = saved_args.get("channels", 112)
    num_res_blocks = saved_args.get("num_res_blocks", 16)
    embed_dim      = saved_args.get("embed_dim", 32)

    model = IRISConditioned(
        channels=channels,
        num_res_blocks=num_res_blocks,
        embed_dim=embed_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded best.pt (PSNR: {best_psnr:.2f} dB, "
          f"channels={channels}, blocks={num_res_blocks})")
    return model, best_psnr, saved_args


# =============================================================================
#  Archive & hot-swap
# =============================================================================

def archive_and_swap(
    checkpoint_dir: Path,
    production_dir: Path,
    archive_dir: Path,
    new_model_state: dict,
    new_psnr: float,
    new_ssim: float,
    epoch: int,
    saved_args: dict,
):
    """Archive old best.pt and save new model as best.pt + production copy."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    production_dir.mkdir(parents=True, exist_ok=True)

    best_path = checkpoint_dir / "best.pt"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if best_path.exists():
        archive_path = archive_dir / f"best_{timestamp}.pt"
        shutil.copy2(str(best_path), str(archive_path))
        print(f"  Archived old model -> {archive_path.name}")

    torch.save(
        {
            "epoch": epoch,
            "model_state": new_model_state,
            "val_psnr": new_psnr,
            "val_ssim": new_ssim,
            "args": saved_args,
            "retrained_at": timestamp,
        },
        best_path,
    )
    print(f"  Saved new best.pt (PSNR: {new_psnr:.2f} dB)")

    prod_path = production_dir / "best.pt"
    shutil.copy2(str(best_path), str(prod_path))
    print(f"  Copied to production -> {prod_path}")


# =============================================================================
#  History logger
# =============================================================================

def init_history_csv(path: Path):
    """Create retrain_history.csv with header if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "num_new_files", "total_train_size",
                "old_psnr", "new_psnr", "old_ssim", "new_ssim",
                "model_replaced", "epochs", "retrain_time_sec",
            ])


def log_history(
    path: Path, num_new: int, total_train: int,
    old_psnr: float, new_psnr: float,
    old_ssim: float, new_ssim: float,
    replaced: bool, epochs: int, elapsed: float,
):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(),
            num_new, total_train,
            f"{old_psnr:.4f}", f"{new_psnr:.4f}",
            f"{old_ssim:.6f}", f"{new_ssim:.6f}",
            replaced, epochs, f"{elapsed:.1f}",
        ])


# =============================================================================
#  Core retrain cycle
# =============================================================================

def retrain_cycle(args) -> bool:
    """
    Run one retrain cycle:
      1. Check for new data
      2. Ingest it
      3. Fine-tune IRISConditioned
      4. Evaluate
      5. Hot-swap if better

    Returns True if retraining was triggered, False if no new data found.
    """
    new_data_dir   = Path(args.new_data_dir)
    data_root      = Path(args.data_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    production_dir = Path(args.production_dir)
    archive_dir    = Path(args.archive_dir)
    history_csv    = Path(args.history_csv)

    # --- Step 1: Check for new data ---
    new_pairs = find_new_pairs(new_data_dir)
    if not new_pairs:
        return False

    if len(new_pairs) < args.min_new_files:
        print(f"  Found {len(new_pairs)} new pair(s), waiting for at least {args.min_new_files}")
        return False

    print(f"\n{'='*60}")
    print(f"  AUTO-RETRAIN TRIGGERED")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  New data pairs found: {len(new_pairs)}")
    print(f"{'='*60}\n")

    # --- Step 2: Ingest new data ---
    if args.dry_run:
        print(f"  [DRY RUN] Would ingest {len(new_pairs)} files")
        ingested = len(new_pairs)
    else:
        ingested = ingest_new_data(new_data_dir, data_root, new_pairs)
        print(f"  Ingested {ingested} new pair(s) into training set")

    if ingested == 0 and not args.dry_run:
        print("  No new files were actually ingested (all duplicates?). Skipping retrain.")
        return False

    # --- Step 3: Set up training ---
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model, old_psnr, saved_args = load_best_checkpoint(checkpoint_dir, device)

    if model is None:
        channels       = getattr(args, "channels", 112)
        num_res_blocks = getattr(args, "num_res_blocks", 16)
        embed_dim      = getattr(args, "embed_dim", 32)
        model = IRISConditioned(
            channels=channels,
            num_res_blocks=num_res_blocks,
            embed_dim=embed_dim,
        ).to(device)
        saved_args = {
            "channels": channels,
            "num_res_blocks": num_res_blocks,
            "embed_dim": embed_dim,
        }

    train_base, val_set = make_train_val_split(
        str(data_root), val_fraction=args.val_fraction, seed=args.seed
    )
    total_train_size = len(train_base)
    print(f"  Total training pairs: {total_train_size}")
    print(f"  Validation pairs: {len(val_set)}")

    if args.dry_run:
        print(f"  [DRY RUN] Would fine-tune for {args.retrain_epochs} epochs at lr={args.lr}")
        print(f"  [DRY RUN] Current best PSNR: {old_psnr:.2f} dB")
        log_history(
            history_csv, ingested, total_train_size,
            old_psnr, 0.0, 0.0, 0.0,
            False, 0, 0.0,
        )
        return True

    train_loader = DataLoader(
        train_base, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Step 4: Fine-tune ---
    loss_fn = CombinedLoss(
        lambda_pixel=1.0, lambda_struct=0.2,
        lambda_edge=0.3, lambda_freq=0.0,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.retrain_epochs
    )

    print(f"\n  Fine-tuning for {args.retrain_epochs} epochs (lr={args.lr}) ...")
    t0 = time.time()

    best_retrain_psnr  = -float("inf")
    best_retrain_ssim  = 0.0
    best_retrain_state = None
    best_retrain_epoch = 0

    for epoch in range(1, args.retrain_epochs + 1):
        epoch_t0 = time.time()

        train_m = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
        scheduler.step()

        elapsed = time.time() - epoch_t0
        print(
            f"  Epoch {epoch:3d}/{args.retrain_epochs} | "
            f"train {train_m['loss']:.5f} | "
            f"val {val_m['loss']:.5f} | "
            f"PSNR {val_m['psnr']:.2f} dB | "
            f"SSIM {val_m['ssim']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_m["psnr"] > best_retrain_psnr:
            best_retrain_psnr  = val_m["psnr"]
            best_retrain_ssim  = val_m["ssim"]
            best_retrain_epoch = epoch
            import copy
            best_retrain_state = copy.deepcopy(model.state_dict())

    total_time = time.time() - t0
    print(f"\n  Fine-tuning complete in {total_time:.1f}s")
    print(f"  Best retrain PSNR: {best_retrain_psnr:.2f} dB (epoch {best_retrain_epoch})")

    # --- Step 5: Compare & hot-swap ---
    replaced = False
    if best_retrain_psnr > old_psnr:
        print(f"\n  >>> NEW MODEL IS BETTER! <<<")
        print(f"     Old PSNR: {old_psnr:.4f} dB")
        print(f"     New PSNR: {best_retrain_psnr:.4f} dB  (+{best_retrain_psnr - old_psnr:.4f})")
        archive_and_swap(
            checkpoint_dir, production_dir, archive_dir,
            best_retrain_state, best_retrain_psnr, best_retrain_ssim,
            best_retrain_epoch, saved_args,
        )
        replaced = True
    else:
        print(f"\n  New model is NOT better -- keeping current model")
        print(f"     Old PSNR: {old_psnr:.4f} dB")
        print(f"     New PSNR: {best_retrain_psnr:.4f} dB  ({best_retrain_psnr - old_psnr:+.4f})")

    # --- Step 6: Log ---
    old_ssim = 0.0
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists() and not replaced:
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            old_ssim = ckpt.get("val_ssim", 0.0)
        except Exception:
            pass

    log_history(
        history_csv, ingested, total_train_size,
        old_psnr, best_retrain_psnr,
        old_ssim, best_retrain_ssim,
        replaced, args.retrain_epochs, total_time,
    )
    print(f"  Logged to {history_csv}")

    return True


# =============================================================================
#  Main — daemon loop or one-shot
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Auto-retrain IRIS IRISConditioned model when new data arrives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    parser.add_argument("--data_root",      type=str, default=str(DATA_ROOT))
    parser.add_argument("--new_data_dir",   type=str, default=str(NEW_DATA_DIR))
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--production_dir", type=str, default=str(PRODUCTION_DIR))
    parser.add_argument("--archive_dir",    type=str, default=str(ARCHIVE_DIR))
    parser.add_argument("--history_csv",    type=str, default=str(HISTORY_CSV))

    # Retraining hyperparameters
    parser.add_argument("--retrain_epochs", type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=5e-5,
                        help="Fine-tuning learning rate (lower than initial 1e-4)")
    parser.add_argument("--val_fraction",   type=float, default=0.1)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--num_workers",    type=int,   default=2)
    parser.add_argument("--channels",       type=int,   default=112)
    parser.add_argument("--num_res_blocks", type=int,   default=16)
    parser.add_argument("--embed_dim",      type=int,   default=32)

    # Daemon settings
    parser.add_argument("--poll_interval",  type=int,   default=60)
    parser.add_argument("--min_new_files",  type=int,   default=1)
    parser.add_argument("--once",           action="store_true")
    parser.add_argument("--dry_run",        action="store_true")
    args = parser.parse_args()

    new_data_noisy = Path(args.new_data_dir) / "NoisyLR"
    new_data_gt    = Path(args.new_data_dir) / "GT"
    new_data_noisy.mkdir(parents=True, exist_ok=True)
    new_data_gt.mkdir(parents=True, exist_ok=True)

    init_history_csv(Path(args.history_csv))

    print("=" * 60)
    print("  IRIS Auto-Retrain Pipeline")
    print("=" * 60)
    print(f"  Model          : IRISConditioned (Experiment 5)")
    print(f"  Data root      : {args.data_root}")
    print(f"  New data watch : {args.new_data_dir}")
    print(f"  Checkpoint dir : {args.checkpoint_dir}")
    print(f"  Production dir : {args.production_dir}")
    print(f"  Retrain epochs : {args.retrain_epochs}")
    print(f"  Fine-tune LR   : {args.lr}")
    print(f"  Poll interval  : {args.poll_interval}s")
    print(f"  Min new files  : {args.min_new_files}")
    print(f"  Mode           : {'one-shot' if args.once else 'daemon'}")
    print(f"  Dry run        : {args.dry_run}")
    print("=" * 60)
    print()

    if args.once:
        triggered = retrain_cycle(args)
        if not triggered:
            print("No new data found. Nothing to do.")
    else:
        print(f"Watching {args.new_data_dir} for new data "
              f"(checking every {args.poll_interval}s) ...")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                triggered = retrain_cycle(args)
                if not triggered:
                    time.sleep(args.poll_interval)
                else:
                    print("\nRetrain cycle complete. Resuming watch ...\n")
                    time.sleep(5)
        except KeyboardInterrupt:
            print("\n\nStopped by user. Exiting.")


if __name__ == "__main__":
    main()
