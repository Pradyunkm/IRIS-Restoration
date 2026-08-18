"""
run.py  –  Team-LML  |  SEMICON Hackathon 2026 – KLA Problem Statement
AI-Based Restoration of Degraded Images

Model: Experiment 5 – IRISConditioned
       Degradation-aware FiLM-conditioned ResNet with PixelShuffle 2× upsample

Architecture:
  - DegradationEncoder: small CNN that embeds the input's degradation characteristics
  - FiLMResidualBlock: 16 residual blocks modulated by the degradation embedding
  - PixelShuffle 2× upsample head + global residual (bilinear + learned correction)
  - Parameters: ~4.66 M

Usage:
    python run.py <input-dir> <output-dir>

    <input-dir>   Directory containing NoisyLR .npy files  (128×128, float32)
    <output-dir>  Directory where restored .npy files will be written
                  (created automatically if it does not exist)

Each input file  <input-dir>/NNNNNN.npy  produces exactly one output file
<output-dir>/NNNNNN.npy  (same filename, shape 256×256, float32, values in [0,1]).
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
#  Model architecture  (self-contained – no external module imports)
# =============================================================================

class ResidualBlock(nn.Module):
    """Standard residual block used in the post-upsample refinement stage."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class PixelShuffleUpsample(nn.Module):
    """2× spatial upsample via PixelShuffle (no checkerboard artefacts)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.shuffle(self.conv(x)))


class DegradationEncoder(nn.Module):
    """
    Small CNN that encodes the degradation characteristics of the input.
    Produces a compact embedding used by FiLM layers to adapt each block's
    processing to the specific noise / blur profile of the current image.
    """
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
        feat = self.net(x).flatten(1)
        return self.fc(feat)


class FiLMResidualBlock(nn.Module):
    """
    Residual block with FiLM (Feature-wise Linear Modulation).
    The degradation embedding is projected to per-channel scale (gamma)
    and shift (beta): out = gamma * features + beta.
    This lets the network adapt internally to each image's degradation.
    """
    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.film = nn.Linear(embed_dim, channels * 2)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        gamma_beta = self.film(embedding)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        out = out * (1.0 + gamma) + beta
        return identity + out


class IRISConditioned(nn.Module):
    """
    Experiment 5: degradation-aware FiLM-conditioned restoration network.

    Input : (B, 1, 128, 128) float32
    Output: (B, 1, 256, 256) float32

    Default config: channels=112, num_res_blocks=16, embed_dim=32 → ~4.66 M params
    """
    def __init__(self, channels: int = 112, num_res_blocks: int = 16, embed_dim: int = 32):
        super().__init__()
        self.degradation_encoder = DegradationEncoder(embed_dim=embed_dim)

        self.stem = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.stem_relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.ModuleList(
            [FiLMResidualBlock(channels, embed_dim) for _ in range(num_res_blocks)]
        )

        self.pre_upsample_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.upsample = PixelShuffleUpsample(channels)
        self.post_upsample_block = ResidualBlock(channels)

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        naive_upsample = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        embedding = self.degradation_encoder(x)

        feat = self.stem_relu(self.stem(x))
        for block in self.res_blocks:
            feat = block(feat, embedding)

        feat = self.pre_upsample_conv(feat)
        feat = self.upsample(feat)
        feat = self.post_upsample_block(feat)

        correction = self.refine(feat)
        return naive_upsample + correction

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
#  Test-Time Augmentation (TTA)
# =============================================================================

def _augment(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0: return x
    if mode == 1: return torch.flip(x, dims=[3])
    if mode == 2: return torch.flip(x, dims=[2])
    if mode == 3: return torch.rot90(x, k=1, dims=[2, 3])
    if mode == 4: return torch.rot90(x, k=2, dims=[2, 3])
    if mode == 5: return torch.rot90(x, k=3, dims=[2, 3])
    if mode == 6: return torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3])
    if mode == 7: return torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
    return x

def _inverse_augment(x: torch.Tensor, mode: int) -> torch.Tensor:
    if mode == 0: return x
    if mode == 1: return torch.flip(x, dims=[3])
    if mode == 2: return torch.flip(x, dims=[2])
    if mode == 3: return torch.rot90(x, k=3, dims=[2, 3])
    if mode == 4: return torch.rot90(x, k=2, dims=[2, 3])
    if mode == 5: return torch.rot90(x, k=1, dims=[2, 3])
    if mode == 6: return torch.flip(x, dims=[3])
    if mode == 7: return torch.flip(x, dims=[2])
    return x

@torch.no_grad()
def inference_tta(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    preds = []
    for mode in range(8):
        aug_in = _augment(t, mode)
        aug_pred = model(aug_in).clamp(0.0, 1.0)
        preds.append(_inverse_augment(aug_pred, mode))
    return torch.stack(preds, dim=0).mean(dim=0)


# =============================================================================
#  Helpers
# =============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt

    # Read architecture args saved in checkpoint, fall back to defaults
    saved_args    = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    channels      = saved_args.get("channels", 112)
    num_res_blocks = saved_args.get("num_res_blocks", 16)
    embed_dim     = saved_args.get("embed_dim", 32)

    model = IRISConditioned(
        channels=channels,
        num_res_blocks=num_res_blocks,
        embed_dim=embed_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"[Team-LML] Loaded checkpoint: {checkpoint_path} "
          f"(channels={channels}, blocks={num_res_blocks})", flush=True)
    return model


def validate_output(arr: np.ndarray, filename: str) -> np.ndarray:
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

    Drop new training pairs into:
      new_data/NoisyLR/NNNNNN.npy  (128×128 float32)
      new_data/GT/NNNNNN.npy       (256×256 float32)
    before running run.py, and the model will be automatically fine-tuned.
    """
    root_dir = Path(__file__).resolve().parent.parent
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

    new_data_dir = NEW_DATA_DIR
    new_pairs = find_new_pairs(new_data_dir)

    if not new_pairs:
        if verbose:
            print(
                f"[Auto-Train] No new paired data found in {new_data_dir}.\n"
                f"[Auto-Train] To trigger retraining, drop matching .npy files into:\n"
                f"[Auto-Train]   {new_data_dir / 'NoisyLR'}  (128×128 inputs)\n"
                f"[Auto-Train]   {new_data_dir / 'GT'}        (256×256 ground truth)\n"
                f"[Auto-Train] Current model is up-to-date.",
                flush=True,
            )
        return

    print(f"\n[Auto-Train] Detected {len(new_pairs)} new training pair(s) in {new_data_dir}!", flush=True)
    print("[Auto-Train] Initiating fine-tuning & accuracy evaluation...", flush=True)

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
        epilog="Example:\n  python run.py /data/NoisyLR /data/restored_output",
    )
    parser.add_argument("input_dir",  type=str,
                        help="Directory containing NoisyLR .npy input files (128×128, float32)")
    parser.add_argument("output_dir", type=str,
                        help="Directory where restored .npy files will be saved")
    parser.add_argument("--no_tta", action="store_true", help="Disable 8-way TTA")
    parser.add_argument("--no_auto_train", action="store_true", help="Disable checking for new training data")

    args = parser.parse_args()

    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "models" / "best.pt"

    if not model_path.exists():
        # Fallback to checkpoints_exp5 if not yet copied to models/
        fallback = script_dir.parent / "checkpoints_exp5" / "best.pt"
        if fallback.exists():
            model_path = fallback
        else:
            print(f"[ERROR] Model checkpoint not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Team-LML] Device: {device}", flush=True)
    print(f"[Team-LML] TTA: {'disabled' if args.no_tta else '8-way ensemble enabled'}", flush=True)

    model = load_model(model_path, device)

    input_files = sorted(p for p in input_dir.glob("*.npy") if not p.name.startswith("._"))
    if not input_files:
        print(f"[ERROR] No .npy files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[Team-LML] Found {len(input_files)} input file(s). Restoring …", flush=True)

    skipped = 0
    for file_path in tqdm(input_files, desc="Restoring", unit="img"):
        noisy = np.load(file_path).astype(np.float32)

        if noisy.ndim == 3 and noisy.shape[2] == 1:
            noisy = noisy[:, :, 0]

        if noisy.ndim != 2 or noisy.shape[0] != 128 or noisy.shape[1] != 128:
            print(f"  WARNING: skipping {file_path.name} (unexpected shape {noisy.shape})", flush=True)
            skipped += 1
            continue

        t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            if args.no_tta:
                pred_t = model(t).clamp(0.0, 1.0)
            else:
                pred_t = inference_tta(model, t)

        pred = pred_t[0, 0].cpu().numpy()
        pred = validate_output(pred, file_path.name)

        out_path = output_dir / file_path.name
        np.save(out_path, pred)

    processed = len(input_files) - skipped
    print(f"\n[Team-LML] Done. {processed} image(s) restored -> {output_dir}", flush=True)
    if skipped:
        print(f"           {skipped} file(s) skipped (unexpected shape).", flush=True)

    # --- Automatic Retraining Trigger ---
    if not args.no_auto_train:
        trigger_auto_retrain_if_new_data_present(verbose=True)


if __name__ == "__main__":
    main()
