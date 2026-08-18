"""
evaluate.py

Standalone evaluation/inference script, per report Section 18:
"The evaluator should accept an input directory and output directory,
load the trained weights, process all images, and save restored outputs."

This is deliberately independent of train.py/train_exp2.py/train_exp3.py --
it does not need the dataset.py train/val split logic at all, since a
real evaluator will point this at a held-out test folder we've never
seen, not our own validation split. It only needs: a folder of NoisyLR
.npy files, a trained checkpoint, and the matching model architecture.

Defaults to the current best model (IRISConditioned + checkpoints_exp5/best.pt,
degradation-aware conditioning on top of the Experiment 3 backbone), but
--model / --checkpoint let you point it at any
of the three trained checkpoints for comparison.

Usage:
    python evaluate.py --input_dir "C:\\path\\to\\test\\NoisyLR" --output_dir results/test_predictions

    # to evaluate a different checkpoint:
    python evaluate.py --input_dir ... --output_dir ... --checkpoint checkpoints\\best.pt --model baseline

Outputs (per input file NNNNNN.npy):
    output_dir/npy/NNNNNN.npy   -- restored image, float32, clipped to [0,1] (for quantitative scoring)
    output_dir/png/NNNNNN.png   -- restored image, grayscale PNG (for quick visual review)

If GT files are available alongside NoisyLR (i.e. this is run against a
labeled val/test split rather than a truly blind test set), pass
--gt_dir to also get PSNR/SSIM computed and written to metrics.csv --
this is optional and the script works fine without it (a real blind
test set won't have GT available).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model import IRISBaseline, IRISStronger
from model_conditioned import IRISConditioned

def load_model(checkpoint_path: str, model_type: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == "stronger":
        model = IRISStronger(channels=112, num_res_blocks=16).to(device)
    elif model_type == "conditioned":
        model = IRISConditioned(channels=112, num_res_blocks=16, embed_dim=32).to(device)
    else:
        model = IRISBaseline(channels=64, num_res_blocks=8).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    trained_epoch = checkpoint.get("epoch", "unknown")
    trained_psnr = checkpoint.get("val_psnr", None)
    psnr_str = f"{trained_psnr:.2f} dB" if trained_psnr is not None else "unknown"
    print(f"Loaded {model_type} checkpoint: epoch={trained_epoch}, val_psnr={psnr_str}")
    print(f"Model parameters: {model.count_parameters():,}")

    return model


@torch.no_grad()
def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def save_png(array_01: np.ndarray, path: Path):
    """array_01 is float32 in [0,1] -- convert to 8-bit grayscale PNG."""
    img_uint8 = (array_01 * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser(description="IRIS restoration -- batch inference")
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory containing NoisyLR .npy files (128x128, float32)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints_exp5/best.pt")
    parser.add_argument("--model", type=str, default="conditioned", choices=["baseline", "stronger", "conditioned"])
    parser.add_argument("--gt_dir", type=str, default=None,
                         help="Optional: directory of matching GT .npy files, for computing metrics")
    parser.add_argument("--save_png", action="store_true", default=True,
                         help="Also save 8-bit PNG previews alongside .npy outputs (default: on)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.model, device)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    input_files = sorted(
        p for p in input_dir.glob("*.npy") if not p.name.startswith("._")
    )
    if len(input_files) == 0:
        raise ValueError(f"No .npy files found in {input_dir}")
    print(f"Found {len(input_files)} input files")

    output_dir = Path(args.output_dir)
    npy_out_dir = output_dir / "npy"
    npy_out_dir.mkdir(parents=True, exist_ok=True)
    png_out_dir = None
    if args.save_png:
        png_out_dir = output_dir / "png"
        png_out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    metrics_rows = []

    for file_path in tqdm(input_files, desc="Restoring"):
        noisy = np.load(file_path).astype(np.float32)
        if noisy.shape != (128, 128):
            print(f"WARNING: skipping {file_path.name}, unexpected shape {noisy.shape}")
            continue

        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,128,128)

        with torch.no_grad():
            pred_tensor = model(noisy_tensor).clamp(0.0, 1.0)

        pred = pred_tensor[0, 0].cpu().numpy()  # (256,256), float32, in [0,1]

        np.save(npy_out_dir / file_path.name, pred)
        if png_out_dir is not None:
            save_png(pred, png_out_dir / (file_path.stem + ".png"))

        if gt_dir is not None:
            gt_path = gt_dir / file_path.name
            if gt_path.exists():
                gt = np.load(gt_path).astype(np.float32)
                psnr = compute_psnr(pred, gt)
                metrics_rows.append({"file_id": file_path.stem, "psnr": psnr})

    print()
    print(f"Restored {len(input_files)} images.")
    print(f"NPY outputs saved to: {npy_out_dir.resolve()}")
    if png_out_dir is not None:
        print(f"PNG previews saved to: {png_out_dir.resolve()}")

    if gt_dir is not None and metrics_rows:
        metrics_path = output_dir / "metrics.csv"
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file_id", "psnr"])
            writer.writeheader()
            writer.writerows(metrics_rows)

        mean_psnr = sum(r["psnr"] for r in metrics_rows) / len(metrics_rows)
        print()
        print(f"Mean PSNR over {len(metrics_rows)} GT-matched files: {mean_psnr:.2f} dB")
        print(f"Per-file metrics saved to: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()