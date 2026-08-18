"""
compute_clean_metrics.py

Recomputes validation PSNR/SSIM for all five trained checkpoints
(Experiments 1–5), reporting BOTH:
    - "official": full val split, matching training-time logged numbers
    - "clean": same val split with the known corrupted/noise-only
      samples excluded (file IDs 002637 and 002973 -- confirmed via
      inspect_outliers.py to have GT std=0.289, the signature of pure
      uniform random noise; no model can achieve meaningful PSNR
      against an unpredictable target)

Experiment 5 (IRISConditioned + FiLM conditioning) is the final
submitted model.  Exp 5 with 8-way TTA is also evaluated for reference.

Usage:
    python compute_clean_metrics.py --data_root "C:\\path\\to\\dataset\\train\\train"
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import make_train_val_split
from model import IRISBaseline, IRISStronger
from model_conditioned import IRISConditioned


KNOWN_CORRUPTED_IDS = {"002637", "002973"}  # confirmed pure-noise GT, no learnable structure


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

    mu_x = F.conv2d(pred_clipped, window_2d, padding=pad)
    mu_y = F.conv2d(target, window_2d, padding=pad)
    mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x_sq = F.conv2d(pred_clipped * pred_clipped, window_2d, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window_2d, padding=pad) - mu_y_sq
    sigma_xy = F.conv2d(pred_clipped * target, window_2d, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return ssim_map.mean().item()


def _augment(x, mode):
    if mode == 0: return x
    if mode == 1: return torch.flip(x, dims=[3])
    if mode == 2: return torch.flip(x, dims=[2])
    if mode == 3: return torch.rot90(x, k=1, dims=[2, 3])
    if mode == 4: return torch.rot90(x, k=2, dims=[2, 3])
    if mode == 5: return torch.rot90(x, k=3, dims=[2, 3])
    if mode == 6: return torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3])
    if mode == 7: return torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
    return x

def _inverse_augment(x, mode):
    if mode == 0: return x
    if mode == 1: return torch.flip(x, dims=[3])
    if mode == 2: return torch.flip(x, dims=[2])
    if mode == 3: return torch.rot90(x, k=3, dims=[2, 3])
    if mode == 4: return torch.rot90(x, k=2, dims=[2, 3])
    if mode == 5: return torch.rot90(x, k=1, dims=[2, 3])
    if mode == 6: return torch.flip(x, dims=[3])
    if mode == 7: return torch.flip(x, dims=[2])
    return x

def evaluate_checkpoint(model, val_set, device, use_tta=False):
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    official_psnr, official_ssim = [], []
    clean_psnr, clean_ssim = [], []

    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)
            file_id = batch["file_id"][0]

            if use_tta:
                preds = []
                for m in range(8):
                    aug_in = _augment(noisy, m)
                    aug_pred = model(aug_in).clamp(0.0, 1.0)
                    preds.append(_inverse_augment(aug_pred, m))
                pred = torch.stack(preds, dim=0).mean(dim=0)
            else:
                pred = model(noisy)

            psnr = compute_psnr(pred, gt)
            ssim = compute_ssim(pred, gt)

            official_psnr.append(psnr)
            official_ssim.append(ssim)

            if file_id not in KNOWN_CORRUPTED_IDS:
                clean_psnr.append(psnr)
                clean_ssim.append(ssim)

    return {
        "official_psnr": np.mean(official_psnr),
        "official_ssim": np.mean(official_ssim),
        "clean_psnr": np.mean(clean_psnr),
        "clean_ssim": np.mean(clean_ssim),
        "n_official": len(official_psnr),
        "n_clean": len(clean_psnr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    _, val_set = make_train_val_split(args.data_root, val_fraction=args.val_fraction, seed=args.seed)

    experiments = [
        ("Experiment 1 (baseline)",                     "checkpoints/best.pt",      "baseline",   False),
        ("Experiment 2 (+ struct/edge loss)",            "checkpoints_exp2/best.pt", "baseline",   False),
        ("Experiment 3 (+ capacity)",                   "checkpoints_exp3/best.pt", "stronger",   False),
        ("Experiment 4 (+ synthetic augmentation)",      "checkpoints_exp4/best.pt", "stronger",   False),
        ("Experiment 5 (+ degradation conditioning)",    "checkpoints_exp5/best.pt", "conditioned", False),
        ("Experiment 5 + 8-way TTA (final submission)",  "checkpoints_exp5/best.pt", "conditioned", True),
    ]

    results = []

    for name, ckpt_path, model_type, use_tta in experiments:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except FileNotFoundError:
            print(f"SKIP {name}: checkpoint not found at {ckpt_path}")
            continue

        if model_type == "stronger":
            model = IRISStronger(channels=112, num_res_blocks=16).to(device)
        elif model_type == "conditioned":
            model = IRISConditioned(channels=112, num_res_blocks=16, embed_dim=32).to(device)
        else:
            model = IRISBaseline(channels=64, num_res_blocks=8).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        metrics = evaluate_checkpoint(model, val_set, device, use_tta=use_tta)
        metrics["name"] = name
        results.append(metrics)

        print(f"{name}")
        print(f"  Official (n={metrics['n_official']}): PSNR={metrics['official_psnr']:.2f} dB, SSIM={metrics['official_ssim']:.4f}")
        print(f"  Clean    (n={metrics['n_clean']}): PSNR={metrics['clean_psnr']:.2f} dB, SSIM={metrics['clean_ssim']:.4f}")
        print()

    print("=" * 70)
    print("Summary table")
    print("=" * 70)
    print(f"{'Experiment':<35} {'Official PSNR':>14} {'Clean PSNR':>12}")
    for r in results:
        print(f"{r['name']:<35} {r['official_psnr']:>11.2f} dB {r['clean_psnr']:>9.2f} dB")


if __name__ == "__main__":
    main()