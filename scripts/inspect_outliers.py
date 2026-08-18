"""
inspect_outliers.py

Loads specific file_ids from the raw dataset (NoisyLR + GT) and the
model's restored output, and saves side-by-side comparisons -- used to
investigate the cluster of anomalously low PSNR scores (~11 dB) found
in metrics.csv from the full-dataset evaluate.py run, e.g.:

    000352, 000625, 000626, 000627, 000957, 000958, 000959,
    002637, 002638, 002639, 002973, 002974, 002975,
    002981, 002982, 002983

Everything else in the dataset scores 17-46 dB, so an ~11 dB cluster is
either a data-quality issue (corrupted/degenerate sample) or a genuine,
specific model failure mode -- this script lets us see which.

Usage:
    python inspect_outliers.py --data_root "C:\\Users\\sekuh\\Desktop\\semicon\\train" --checkpoint checkpoints_exp3/best.pt --model stronger --file_ids 000352 000625 000957 002637 002973 002981
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import IRISBaseline, IRISStronger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints_exp3/best.pt")
    parser.add_argument("--model", type=str, default="stronger", choices=["baseline", "stronger"])
    parser.add_argument("--file_ids", type=str, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="results/outliers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.model == "stronger":
        model = IRISStronger(channels=112, num_res_blocks=16).to(device)
    else:
        model = IRISBaseline(channels=64, num_res_blocks=8).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data_root = Path(args.data_root)
    noisy_dir = data_root / "NoisyLR"
    gt_dir = data_root / "GT"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_id in args.file_ids:
        noisy_path = noisy_dir / f"{file_id}.npy"
        gt_path = gt_dir / f"{file_id}.npy"

        if not noisy_path.exists() or not gt_path.exists():
            print(f"SKIP {file_id}: file not found")
            continue

        noisy = np.load(noisy_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)

        print(f"\n--- {file_id} ---")
        print(f"NoisyLR: min={noisy.min():.4f} max={noisy.max():.4f} mean={noisy.mean():.4f} std={noisy.std():.4f}")
        print(f"GT     : min={gt.min():.4f} max={gt.max():.4f} mean={gt.mean():.4f} std={gt.std():.4f}")

        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_tensor = model(noisy_tensor).clamp(0.0, 1.0)
        pred = pred_tensor[0, 0].cpu().numpy()

        mse = np.mean((pred - gt) ** 2)
        psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else 100.0
        print(f"Prediction: min={pred.min():.4f} max={pred.max():.4f} mean={pred.mean():.4f} | PSNR={psnr:.2f} dB")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(noisy.clip(0, 1), cmap="gray")
        axes[0].set_title(f"NoisyLR\n{file_id}")
        axes[1].imshow(pred, cmap="gray")
        axes[1].set_title(f"Prediction\nPSNR={psnr:.2f} dB")
        axes[2].imshow(gt, cmap="gray")
        axes[2].set_title("Ground Truth")
        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        out_path = out_dir / f"outlier_{file_id}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"\nDone. Comparisons saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()