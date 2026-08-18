"""
dataset.py

PyTorch Dataset for the IRIS paired restoration task.

Pairs:
    NoisyLR : 128x128 float32, unclipped (can be < 0 or > 1)
    GT      : 256x256 float32, clipped to [0, 1]

Design decisions (based on dataset_audit.py + visualize_pairs.py findings
on samples 000040, 000052, 000064):

- Degradation is additive, ~zero-mean noise on top of a downsampled GT.
  Mean(NoisyLR) matches Mean(GT) almost exactly across samples; NoisyLR
  std is consistently higher than GT std; NoisyLR max always exceeds 1.0
  and min sits near (but not exactly) 0. This is NOT sensor noise with
  fixed-pattern structure -- it looks like synthetic unclipped Gaussian
  noise added after downsampling.

- Therefore: DO NOT clip NoisyLR to [0, 1] before feeding it to the model.
  Clipping would destroy real information in the noise tails and would
  also change the effective noise distribution the model has to learn
  (making it asymmetric near 0 and 1). Clipping is only applied to
  predictions at evaluation/visualization time, never to network input.

- GT is already well-behaved ([0, 1], float32) and needs no normalization
  changes -- only dtype/tensor conversion.

Usage (quick self-test):
    python dataset.py --data_root "D:\\VSProjects\\IRIS---Restoration-main\\dataset\\train\\train"
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class IRISPairedDataset(Dataset):
    """
    Loads paired (NoisyLR, GT) .npy samples from:

        data_root/
            NoisyLR/000000.npy ...
            GT/000000.npy ...

    Pairing is established by filename, not by directory listing order,
    so we never risk NoisyLR[i] getting matched to the wrong GT[i] if the
    two directories were ever to list files in a different order.
    """

    def __init__(self, data_root: str, file_ids=None):
        self.data_root = Path(data_root)
        self.noisy_dir = self.data_root / "NoisyLR"
        self.gt_dir = self.data_root / "GT"

        if not self.noisy_dir.is_dir():
            raise FileNotFoundError(f"NoisyLR directory not found: {self.noisy_dir}")
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

        if file_ids is None:
            noisy_stems = {
                p.stem for p in self.noisy_dir.glob("*.npy") if not p.name.startswith("._")
            }
            gt_stems = {
                p.stem for p in self.gt_dir.glob("*.npy") if not p.name.startswith("._")
            }
            file_ids = sorted(noisy_stems & gt_stems)

            missing_noisy = gt_stems - noisy_stems
            missing_gt = noisy_stems - gt_stems
            if missing_noisy or missing_gt:
                raise ValueError(
                    f"Unmatched files detected (only_in_GT={len(missing_noisy)}, "
                    f"only_in_NoisyLR={len(missing_gt)}). Re-run dataset_audit.py "
                    f"before training -- pairing must PASS."
                )

        self.file_ids = list(file_ids)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]

        noisy = np.load(self.noisy_dir / f"{file_id}.npy").astype(np.float32)
        gt = np.load(self.gt_dir / f"{file_id}.npy").astype(np.float32)

        if noisy.shape != (128, 128):
            raise ValueError(f"{file_id}: expected NoisyLR shape (128,128), got {noisy.shape}")
        if gt.shape != (256, 256):
            raise ValueError(f"{file_id}: expected GT shape (256,256), got {gt.shape}")

        noisy_t = torch.from_numpy(noisy).unsqueeze(0)  # (1, 128, 128)
        gt_t = torch.from_numpy(gt).unsqueeze(0)         # (1, 256, 256)

        return {
            "noisy": noisy_t,
            "gt": gt_t,
            "file_id": file_id,
        }


def make_train_val_split(data_root: str, val_fraction: float = 0.1, seed: int = 42):
    """
    Builds train/val Subsets from a single IRISPairedDataset, using a fixed
    seed so the split is reproducible across runs (important: re-running
    this must always yield the same val set, or metrics aren't comparable
    across experiments).
    """
    full_dataset = IRISPairedDataset(data_root)
    n = len(full_dataset)

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = max(1, int(n * val_fraction))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)

    return train_set, val_set


def _self_test(data_root: str):
    print("=" * 60)
    print("IRISPairedDataset self-test")
    print("=" * 60)

    train_set, val_set = make_train_val_split(data_root, val_fraction=0.1, seed=42)
    print(f"Total pairs found : {len(train_set) + len(val_set)}")
    print(f"Train pairs       : {len(train_set)}")
    print(f"Val pairs         : {len(val_set)}")

    loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print()
    print("Sample batch:")
    print(f"  noisy shape : {tuple(batch['noisy'].shape)}, dtype={batch['noisy'].dtype}")
    print(f"  gt shape    : {tuple(batch['gt'].shape)}, dtype={batch['gt'].dtype}")
    print(f"  noisy range : [{batch['noisy'].min():.4f}, {batch['noisy'].max():.4f}]")
    print(f"  gt range    : [{batch['gt'].min():.4f}, {batch['gt'].max():.4f}]")
    print(f"  file_ids    : {batch['file_id']}")

    _, val_set_2 = make_train_val_split(data_root, val_fraction=0.1, seed=42)
    val_ids_1 = sorted(val_set.dataset.file_ids[i] for i in val_set.indices)
    val_ids_2 = sorted(val_set_2.dataset.file_ids[i] for i in val_set_2.indices)
    assert val_ids_1 == val_ids_2, "Val split is NOT reproducible -- check the seed logic!"
    print()
    print("Reproducibility check passed: val split is identical across runs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRIS paired dataset self-test")
    parser.add_argument("--data_root", type=str,
                        default=r"D:\VSProjects\IRIS---Restoration-main\dataset\train\train",
                        help=r'Path to train folder, e.g. D:\VSProjects\IRIS---Restoration-main\dataset\train\train')
    args = parser.parse_args()

    _self_test(args.data_root)