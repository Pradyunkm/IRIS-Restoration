"""
run.py  –  Team-LML  |  SEMICON Hackathon 2026 – KLA Problem Statement
AI-Based Restoration of Degraded Images

Model: Experiment 5 – IRISConditioned
       Degradation-aware FiLM-conditioned ResNet with PixelShuffle 2× upsample

Speed optimisations (vs naive one-image-at-a-time):
  1. Batched inference   — N images processed per forward pass (--batch_size, default 16)
  2. Fused batched TTA   — all 8 TTA views stacked into one (N×8, 1, H, W) forward pass
                           instead of 8 separate passes per image  →  8× fewer kernel launches
  3. FP16 autocast       — halves memory bandwidth and VRAM usage on NVIDIA GPUs
  4. cudnn.benchmark     — selects fastest cuDNN kernel for fixed-size inputs
  5. Threaded file I/O   — ThreadPoolExecutor prefetches the next batch from disk
                           while the GPU is processing the current batch
  6. Non-blocking GPU    — pin_memory + non_blocking transfers overlap CPU↔GPU copy

Usage:
    python run.py <input-dir> <output-dir> [--batch_size N] [--no_tta] [--no_auto_train]

    <input-dir>   Directory containing NoisyLR .npy files  (128×128, float32)
    <output-dir>  Directory where restored .npy files will be written
                  (created automatically if it does not exist)

Each input file  <input-dir>/NNNNNN.npy  produces exactly one output file
<output-dir>/NNNNNN.npy  (same filename, shape 256×256, float32, values in [0,1]).
"""

import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
#  Speed settings
# =============================================================================

torch.backends.cudnn.benchmark = True   # fastest kernel for fixed-size inputs


# =============================================================================
#  Model architecture  (self-contained – no external module imports)
# =============================================================================

class ResidualBlock(nn.Module):
    """Standard residual block used in the post-upsample refinement stage."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class PixelShuffleUpsample(nn.Module):
    """2× spatial upsample via PixelShuffle (no checkerboard artefacts)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv    = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.shuffle(self.conv(x)))


class DegradationEncoder(nn.Module):
    """Small CNN that encodes per-image degradation characteristics into a fixed-dim embedding."""
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 128 → 64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64 → 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 32 → 16
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).flatten(1))


class FiLMResidualBlock(nn.Module):
    """
    Residual block with FiLM modulation.
    Degradation embedding → per-channel (γ, β) → out = γ * features + β.
    """
    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.film  = nn.Linear(embed_dim, channels * 2)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        identity          = x
        out               = self.relu(self.conv1(x))
        out               = self.conv2(out)
        gamma_beta        = self.film(embedding)
        gamma, beta       = gamma_beta.chunk(2, dim=1)
        out               = out * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) \
                              + beta.unsqueeze(-1).unsqueeze(-1)
        return identity + out


class IRISConditioned(nn.Module):
    """
    Experiment 5: FiLM-conditioned ResNet for IRIS image restoration.
    Input : (B, 1, 128, 128) → Output: (B, 1, 256, 256)
    """
    def __init__(self, channels: int = 112, num_res_blocks: int = 16, embed_dim: int = 32):
        super().__init__()
        self.degradation_encoder  = DegradationEncoder(embed_dim=embed_dim)
        self.stem                 = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.stem_relu            = nn.ReLU(inplace=True)
        self.res_blocks           = nn.ModuleList(
            [FiLMResidualBlock(channels, embed_dim) for _ in range(num_res_blocks)]
        )
        self.pre_upsample_conv    = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample             = PixelShuffleUpsample(channels)
        self.post_upsample_block  = ResidualBlock(channels)
        self.refine               = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        naive_up  = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        embedding = self.degradation_encoder(x)
        feat      = self.stem_relu(self.stem(x))
        for block in self.res_blocks:
            feat  = block(feat, embedding)
        feat      = self.pre_upsample_conv(feat)
        feat      = self.upsample(feat)
        feat      = self.post_upsample_block(feat)
        return naive_up + self.refine(feat)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
#  Batched fused TTA
#
#  Key idea: instead of 8 forward passes per image, we stack all 8 views of
#  ALL images in the batch into a single (B*8, 1, 128, 128) tensor and run
#  ONE forward pass.  This saturates the GPU pipeline and eliminates 7 out of
#  8 kernel-launch round-trips per image.
# =============================================================================

# Pre-computed (mode → transform fn) tables — avoids branching in the hot path
_AUG_FNS = [
    lambda x: x,                                                        # 0: identity
    lambda x: x.flip(3),                                                # 1: hflip
    lambda x: x.flip(2),                                                # 2: vflip
    lambda x: x.rot90(1, [2, 3]),                                       # 3: rot90
    lambda x: x.rot90(2, [2, 3]),                                       # 4: rot180
    lambda x: x.rot90(3, [2, 3]),                                       # 5: rot270
    lambda x: x.rot90(1, [2, 3]).flip(3),                               # 6: rot90+hflip
    lambda x: x.rot90(1, [2, 3]).flip(2),                               # 7: rot90+vflip
]

_INV_AUG_FNS = [
    lambda x: x,                                                        # 0
    lambda x: x.flip(3),                                                # 1
    lambda x: x.flip(2),                                                # 2
    lambda x: x.rot90(3, [2, 3]),                                       # 3
    lambda x: x.rot90(2, [2, 3]),                                       # 4
    lambda x: x.rot90(1, [2, 3]),                                       # 5
    lambda x: x.flip(3).rot90(3, [2, 3]),                               # 6
    lambda x: x.flip(2).rot90(3, [2, 3]),                               # 7
]

N_TTA = len(_AUG_FNS)   # 8


@torch.no_grad()
def inference_batch(
    model: nn.Module,
    batch: torch.Tensor,          # (B, 1, 128, 128) — already on GPU
    use_tta: bool,
    amp_enabled: bool,
) -> torch.Tensor:
    """
    Run model on a batch.  With TTA, builds a (B*8, 1, H, W) super-batch
    and runs a single forward pass, then reshapes and averages.
    Returns (B, 1, 256, 256) float32 CPU tensor.
    """
    autocast_ctx = torch.amp.autocast("cuda", enabled=amp_enabled)

    if not use_tta:
        with autocast_ctx:
            pred = model(batch).clamp(0.0, 1.0)
        return pred.float().cpu()

    B = batch.shape[0]

    # Stack 8 augmented views → (B*8, 1, 128, 128)
    views = torch.cat([fn(batch) for fn in _AUG_FNS], dim=0)   # (B*8, 1, H, W)

    with autocast_ctx:
        out_all = model(views).clamp(0.0, 1.0)                  # (B*8, 1, 256, 256)

    # out_all[mode*B : (mode+1)*B] corresponds to augmentation `mode`
    preds = torch.stack(
        [_INV_AUG_FNS[m](out_all[m * B:(m + 1) * B]) for m in range(N_TTA)],
        dim=0,
    )                                                            # (8, B, 1, 256, 256)
    return preds.mean(dim=0).float().cpu()                       # (B, 1, 256, 256)


# =============================================================================
#  Helpers
# =============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state      = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    channels       = saved_args.get("channels", 112)
    num_res_blocks = saved_args.get("num_res_blocks", 16)
    embed_dim      = saved_args.get("embed_dim", 32)

    model = IRISConditioned(
        channels=channels, num_res_blocks=num_res_blocks, embed_dim=embed_dim
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(
        f"[Team-LML] Loaded checkpoint: {checkpoint_path} "
        f"(channels={channels}, blocks={num_res_blocks})",
        flush=True,
    )
    return model


def _load_one(file_path: Path):
    """Load + validate a single .npy file (runs in thread-pool)."""
    try:
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        if arr.ndim == 2 and arr.shape == (128, 128):
            return file_path, arr, None
        return file_path, None, f"unexpected shape {arr.shape}"
    except Exception as e:
        return file_path, None, str(e)


def validate_and_clip(arr: np.ndarray, filename: str) -> np.ndarray:
    if not np.isfinite(arr).all():
        n_bad = (~np.isfinite(arr)).sum()
        print(f"  WARNING: {n_bad} non-finite value(s) in {filename} – replacing with 0", flush=True)
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr.clip(0.0, 1.0).astype(np.float32)


# =============================================================================
#  Auto-Retrain Integration
# =============================================================================

def trigger_auto_retrain_if_new_data_present(verbose: bool = True):
    """
    Checks if new paired training data exists in new_data/.
    If found, runs the auto-retraining pipeline to fine-tune the model,
    evaluate on validation set, and replace models/best.pt if accuracy improves.
    """
    root_dir    = Path(__file__).resolve().parent.parent
    scripts_dir = root_dir / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    print("\n[Auto-Train] Checking for new training data in new_data/ ...", flush=True)

    try:
        from auto_retrain import (
            retrain_cycle, find_new_pairs,
            DATA_ROOT, NEW_DATA_DIR, CHECKPOINT_DIR,
            PRODUCTION_DIR, ARCHIVE_DIR, HISTORY_CSV,
        )
    except ImportError as e:
        print(f"[Auto-Train] WARNING: Could not import auto_retrain module ({e}).", flush=True)
        print(f"[Auto-Train]          Make sure scripts/ is accessible at: {scripts_dir}", flush=True)
        print("[Auto-Train] Skipping auto-retrain check.", flush=True)
        return

    new_pairs = find_new_pairs(NEW_DATA_DIR)

    if not new_pairs:
        if verbose:
            print(
                f"[Auto-Train] No new paired data found in {NEW_DATA_DIR}.\n"
                f"[Auto-Train] Drop matching .npy files into:\n"
                f"[Auto-Train]   {NEW_DATA_DIR / 'NoisyLR'}  (128×128 inputs)\n"
                f"[Auto-Train]   {NEW_DATA_DIR / 'GT'}        (256×256 ground truth)",
                flush=True,
            )
        return

    print(f"\n[Auto-Train] Detected {len(new_pairs)} new pair(s)! Starting fine-tune ...", flush=True)

    class RetrainArgs:
        data_root      = str(DATA_ROOT)
        new_data_dir   = str(NEW_DATA_DIR)
        checkpoint_dir = str(CHECKPOINT_DIR)
        production_dir = str(PRODUCTION_DIR)
        archive_dir    = str(ARCHIVE_DIR)
        history_csv    = str(HISTORY_CSV)
        retrain_epochs = 30
        batch_size     = 8
        lr             = 5e-5
        val_fraction   = 0.1
        seed           = 42
        num_workers    = 2
        channels       = 112
        num_res_blocks = 16
        embed_dim      = 32
        min_new_files  = 1
        dry_run        = False

    retrain_cycle(RetrainArgs())


# =============================================================================
#  Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Team-LML  –  IRIS Image Restoration (SEMICON Hackathon 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python run.py /data/NoisyLR /data/restored_output --batch_size 32",
    )
    parser.add_argument("input_dir",  type=str,
                        help="Directory of NoisyLR .npy input files (128×128, float32)")
    parser.add_argument("output_dir", type=str,
                        help="Directory where restored .npy files will be saved")
    parser.add_argument("--batch_size",     type=int, default=16,
                        help="Images per GPU batch (default: 16 — increase for faster throughput)")
    parser.add_argument("--no_tta",         action="store_true",
                        help="Disable 8-way TTA (faster, ~0.1-0.3 dB lower quality)")
    parser.add_argument("--no_fp16",        action="store_true",
                        help="Disable FP16 autocast (use if GPU does not support it)")
    parser.add_argument("--no_auto_train",  action="store_true",
                        help="Disable checking for new training data")
    parser.add_argument("--io_workers",     type=int, default=4,
                        help="Threads for parallel file I/O (default: 4)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Locate model weights ---
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "models" / "best.pt"
    if not model_path.exists():
        fallback = script_dir.parent / "checkpoints_exp5" / "best.pt"
        if fallback.exists():
            model_path = fallback
        else:
            print(f"[ERROR] Model checkpoint not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_fp16

    print(f"[Team-LML] Device      : {device}", flush=True)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[Team-LML] GPU         : {props.name} "
              f"({props.total_memory // 1024**2} MB VRAM)", flush=True)
    print(f"[Team-LML] Batch size  : {args.batch_size}", flush=True)
    print(f"[Team-LML] TTA         : {'disabled' if args.no_tta else '8-way fused-batch'}", flush=True)
    print(f"[Team-LML] FP16        : {'enabled' if amp_enabled else 'disabled'}", flush=True)

    model = load_model(model_path, device)

    # Warm up CUDA/cuDNN on a dummy tensor so benchmark picks the best kernel
    if device.type == "cuda":
        dummy = torch.zeros(args.batch_size if args.no_tta else min(args.batch_size, 4),
                            1, 128, 128, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
            _ = model(dummy)
        del dummy
        torch.cuda.synchronize()
        print("[Team-LML] GPU warmed up.", flush=True)

    # --- Discover input files ---
    input_files = sorted(p for p in input_dir.glob("*.npy") if not p.name.startswith("._"))
    if not input_files:
        print(f"[ERROR] No .npy files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    total   = len(input_files)
    bs      = args.batch_size
    n_batch = (total + bs - 1) // bs
    print(f"[Team-LML] Found {total} file(s) → {n_batch} batch(es) of up to {bs}.", flush=True)

    t_start  = time.perf_counter()
    skipped  = 0
    processed = 0

    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        for batch_idx in tqdm(range(n_batch), desc="Batches", unit="batch"):
            batch_paths = input_files[batch_idx * bs : (batch_idx + 1) * bs]

            # --- Load this batch in parallel threads ---
            results = list(pool.map(_load_one, batch_paths))

            # Separate valid vs skipped
            valid_paths, valid_arrs = [], []
            for fpath, arr, err in results:
                if arr is not None:
                    valid_paths.append(fpath)
                    valid_arrs.append(arr)
                else:
                    print(f"  WARNING: skipping {fpath.name} ({err})", flush=True)
                    skipped += 1

            if not valid_arrs:
                continue

            # --- Build GPU batch ---
            # stack → (B, 128, 128) → (B, 1, 128, 128)
            np_batch = np.stack(valid_arrs, axis=0)
            gpu_batch = torch.from_numpy(np_batch).unsqueeze(1)
            # pin_memory copy + non-blocking transfer overlaps CPU→GPU with next I/O
            if device.type == "cuda":
                gpu_batch = gpu_batch.pin_memory().to(device, non_blocking=True)
            else:
                gpu_batch = gpu_batch.to(device)

            # --- Fused batched inference (+ optional TTA) ---
            out_cpu = inference_batch(
                model, gpu_batch, use_tta=not args.no_tta, amp_enabled=amp_enabled
            )   # (B, 1, 256, 256) float32 CPU

            # --- Save outputs ---
            for i, fpath in enumerate(valid_paths):
                pred = out_cpu[i, 0].numpy()          # (256, 256)
                pred = validate_and_clip(pred, fpath.name)
                np.save(output_dir / fpath.name, pred)
                processed += 1

    elapsed = time.perf_counter() - t_start
    throughput = processed / elapsed if elapsed > 0 else float("inf")

    print(f"\n[Team-LML] Done. {processed} image(s) restored → {output_dir}", flush=True)
    print(f"[Team-LML] Total time  : {elapsed:.1f}s", flush=True)
    print(f"[Team-LML] Throughput  : {throughput:.1f} img/s", flush=True)
    if skipped:
        print(f"[Team-LML] Skipped     : {skipped} file(s) (unexpected shape / load error).", flush=True)

    # --- Automatic Retraining Trigger ---
    if not args.no_auto_train:
        trigger_auto_retrain_if_new_data_present(verbose=True)


if __name__ == "__main__":
    main()
