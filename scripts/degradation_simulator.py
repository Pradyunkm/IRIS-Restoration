"""
degradation_simulator.py

Synthetic degradation pipeline that mirrors the challenge's stated
degradation components (speckle noise, downsampling, additive Gaussian
noise) applied in RANDOMIZED order per sample -- directly targeting the
KLA problem statement's explicit note that degradation order should not
be assumed fixed.

Given a clean 256x256 GT image, produces a synthetic 128x128 "NoisyLR"
by applying the three operations in one of six random orderings, with
randomized parameters per sample so a single GT image can generate many
different synthetic degraded versions across training.

This does NOT replace the real paired dataset -- it supplements it.
Real pairs teach the true (fixed, unknown) degradation distribution;
synthetic pairs teach robustness to the STATED variable-order property,
which the real dataset alone cannot fully cover (its order, whatever it
is, is fixed and unknown to us).

Usage (self-test):
    python degradation_simulator.py
"""

import random

import numpy as np
import torch
import torch.nn.functional as F


def apply_speckle(x: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(x) * sigma
    return x * (1.0 + noise)


def apply_gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def apply_downsample(x: torch.Tensor) -> torch.Tensor:
    """x: (1, H, W) or (B, 1, H, W). Area-average pool 256 -> 128."""
    needs_batch_dim = x.dim() == 3
    if needs_batch_dim:
        x = x.unsqueeze(0)
    out = F.avg_pool2d(x, kernel_size=2, stride=2)
    return out.squeeze(0) if needs_batch_dim else out


def synthesize_degraded(
    gt_256: torch.Tensor,
    speckle_sigma_range=(0.03, 0.12),
    gauss_sigma_range=(0.03, 0.12),
    seed: int = None,
) -> torch.Tensor:
    """
    gt_256: (1, 256, 256) float32 tensor, values in [0, 1]
    Returns: (1, 128, 128) float32 tensor, synthetic NoisyLR
             (intentionally left unclipped, matching real data behavior)
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    speckle_sigma = rng.uniform(*speckle_sigma_range)
    gauss_sigma = rng.uniform(*gauss_sigma_range)

    orderings = [
        ["speckle", "gaussian", "downsample"],
        ["speckle", "downsample", "gaussian"],
        ["gaussian", "speckle", "downsample"],
        ["gaussian", "downsample", "speckle"],
        ["downsample", "speckle", "gaussian"],
        ["downsample", "gaussian", "speckle"],
    ]
    order = rng.choice(orderings)

    x = gt_256.clone()
    for op in order:
        if op == "speckle":
            x = apply_speckle(x, speckle_sigma)
        elif op == "gaussian":
            x = apply_gaussian(x, gauss_sigma)
        elif op == "downsample":
            x = apply_downsample(x)

    return x


def _self_test():
    print("=" * 60)
    print("degradation_simulator.py self-test")
    print("=" * 60)

    torch.manual_seed(0)
    gt = torch.rand(1, 256, 256)

    shapes_seen = set()
    for i in range(20):
        degraded = synthesize_degraded(gt, seed=i)
        shapes_seen.add(tuple(degraded.shape))

    print(f"Shapes produced across 20 calls: {shapes_seen}")
    assert shapes_seen == {(1, 128, 128)}, "All outputs must be (1,128,128) regardless of operation order!"

    sample = synthesize_degraded(gt, seed=42)
    print(f"Sample stats: min={sample.min():.4f} max={sample.max():.4f} "
          f"mean={sample.mean():.4f} std={sample.std():.4f}")
    print(f"GT stats:     min={gt.min():.4f} max={gt.max():.4f} "
          f"mean={gt.mean():.4f} std={gt.std():.4f}")

    print()
    print("Self-test passed: all orderings produce correctly-shaped output.")


if __name__ == "__main__":
    _self_test()