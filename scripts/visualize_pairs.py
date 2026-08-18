import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def normalize_image(img):
    """
    Normalize an image independently for visualization.
    """
    img = img.astype(np.float32)

    min_val = img.min()
    max_val = img.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(img)

    return (img - min_val) / (max_val - min_val)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        required=True,
        help="Path containing NoisyLR and GT folders"
    )

    parser.add_argument(
        "--sample",
        default="000040",
        help="Sample ID without .npy"
    )

    parser.add_argument(
        "--output",
        default="results/pair_000040.png",
        help="Output visualization path"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)

    noisy_path = data_root / "NoisyLR" / f"{args.sample}.npy"
    gt_path = data_root / "GT" / f"{args.sample}.npy"

    if not noisy_path.exists():
        raise FileNotFoundError(f"NoisyLR file not found: {noisy_path}")

    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    noisy = np.load(noisy_path)
    gt = np.load(gt_path)

    print("\n========== SAMPLE INSPECTION ==========")

    print(f"Sample       : {args.sample}")
    print(f"NoisyLR shape: {noisy.shape}")
    print(f"GT shape     : {gt.shape}")

    print("\nNoisyLR statistics:")
    print(f"  dtype : {noisy.dtype}")
    print(f"  min   : {noisy.min()}")
    print(f"  max   : {noisy.max()}")
    print(f"  mean  : {noisy.mean()}")
    print(f"  std   : {noisy.std()}")

    print("\nGT statistics:")
    print(f"  dtype : {gt.dtype}")
    print(f"  min   : {gt.min()}")
    print(f"  max   : {gt.max()}")
    print(f"  mean  : {gt.mean()}")
    print(f"  std   : {gt.std()}")

    noisy_display = normalize_image(noisy)
    gt_display = normalize_image(gt)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(noisy_display, cmap="gray")
    axes[0].set_title(
        f"NoisyLR\n{noisy.shape[0]}×{noisy.shape[1]}"
    )
    axes[0].axis("off")

    axes[1].imshow(gt_display, cmap="gray")
    axes[1].set_title(
        f"Ground Truth\n{gt.shape[0]}×{gt.shape[1]}"
    )
    axes[1].axis("off")

    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"\nVisualization saved to:")
    print(output_path)


if __name__ == "__main__":
    main()