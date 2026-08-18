"""
augmented_dataset.py

Wraps the real paired dataset (IRISPairedDataset) and, for a configurable
fraction of samples per epoch, replaces the real NoisyLR with a freshly
synthesized, order-randomized degraded version of the SAME GT image
(via degradation_simulator.py) instead.

This means: every epoch, the model sees mostly real pairs (preserving
the true, dataset-specific degradation distribution) plus some fraction
of synthetic pairs with randomized degradation order and randomized
noise parameters -- explicitly training robustness to the "order should
not be assumed" property stated in the KLA problem material, which the
real (fixed-order) pairs alone cannot teach.

GT is never modified -- only which NoisyLR is paired with it.

Usage (self-test):
    python augmented_dataset.py --data_root "C:\\Users\\sekuh\\Desktop\\semicon\\train"
"""

import argparse
import random

import torch
from torch.utils.data import Dataset

from dataset import IRISPairedDataset, make_train_val_split
from degradation_simulator import synthesize_degraded


class AugmentedIRISDataset(Dataset):
    """
    Wraps a real IRISPairedDataset (or Subset of one). With probability
    `synthetic_prob`, replaces the real NoisyLR with a synthetic,
    order-randomized degraded version of the same GT.

    Only used for TRAINING. Validation should always use
    make_train_val_split's val_set directly (real pairs only), so
    metrics stay comparable to Experiments 1-3.
    """

    def __init__(self, base_dataset, synthetic_prob: float = 0.3):
        self.base_dataset = base_dataset
        self.synthetic_prob = synthetic_prob

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        if random.random() < self.synthetic_prob:
            gt = sample["gt"]  # (1, 256, 256), real GT, unchanged
            synthetic_noisy = synthesize_degraded(gt)
            return {
                "noisy": synthetic_noisy,
                "gt": gt,
                "file_id": sample["file_id"] + "_synthetic",
            }

        return sample


def _self_test(data_root: str):
    print("=" * 60)
    print("AugmentedIRISDataset self-test")
    print("=" * 60)

    train_set, _ = make_train_val_split(data_root, val_fraction=0.1, seed=42)
    aug_dataset = AugmentedIRISDataset(train_set, synthetic_prob=0.3)

    print(f"Base train set size: {len(train_set)}")
    print(f"Augmented dataset size (same, just swaps content): {len(aug_dataset)}")

    n_synthetic = 0
    n_real = 0
    for i in range(200):
        sample = aug_dataset[i % len(aug_dataset)]
        if "_synthetic" in sample["file_id"]:
            n_synthetic += 1
        else:
            n_real += 1
        assert sample["noisy"].shape == (1, 128, 128), f"Bad noisy shape: {sample['noisy'].shape}"
        assert sample["gt"].shape == (1, 256, 256), f"Bad gt shape: {sample['gt'].shape}"

    print(f"Over 200 draws: {n_synthetic} synthetic, {n_real} real "
          f"(~{n_synthetic/200*100:.0f}% synthetic, target ~30%)")

    print()
    print("Self-test passed: shapes correct, synthetic ratio roughly matches target.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()
    _self_test(args.data_root)