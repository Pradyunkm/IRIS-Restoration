"""
visualize_predictions.py

Loads the best checkpoint and generates NoisyLR / Prediction / GT triptychs
for a handful of validation samples, so we can visually judge whether the
25.76 dB / 0.767 SSIM baseline is actually preserving fine detail or just
producing a smoothed/blurry average -- PSNR/SSIM alone can't tell us that.

Usage:
    python scripts/visualize_predictions.py --data_root "C:\\Users\\sekuh\\Desktop\\semicon\\train" --checkpoint checkpoints/best.pt --num_samples 6
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import make_train_val_split
from model import IRISBaseline, IRISStronger
from model_conditioned import IRISConditioned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="results/predictions")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "stronger", "conditioned"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_set = make_train_val_split(args.data_root, val_fraction=args.val_fraction, seed=args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.model == "stronger":
        model = IRISStronger(channels=112, num_res_blocks=16).to(device)
    elif args.model == "conditioned":
        model = IRISConditioned(channels=112, num_res_blocks=16, embed_dim=32).to(device)
    else:
        model = IRISBaseline(channels=64, num_res_blocks=8).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, val_psnr={checkpoint['val_psnr']:.2f} dB")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(args.num_samples, len(val_set))

    with torch.no_grad():
        for i in range(num_samples):
            sample = val_set[i]
            noisy = sample["noisy"].unsqueeze(0).to(device)  # (1,1,128,128)
            gt = sample["gt"].unsqueeze(0).to(device)         # (1,1,256,256)
            file_id = sample["file_id"]

            pred = model(noisy).clamp(0.0, 1.0)

            noisy_np = noisy[0, 0].cpu().numpy()
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = gt[0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
            axes[0].imshow(noisy_np.clip(0, 1), cmap="gray")
            axes[0].set_title(f"NoisyLR\n{file_id} (128x128)")
            axes[1].imshow(pred_np, cmap="gray")
            axes[1].set_title("Prediction (256x256)")
            axes[2].imshow(gt_np, cmap="gray")
            axes[2].set_title("Ground Truth (256x256)")
            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            out_path = out_dir / f"pred_{file_id}.png"
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

    print()
    print(f"Done. {num_samples} comparison images saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()