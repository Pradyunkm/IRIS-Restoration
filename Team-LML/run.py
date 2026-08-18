"""
run.py  –  Team-LML  |  SEMICON Hackathon 2026 – KLA Problem Statement
AI-Based Restoration of Degraded Images

Model: Experiment 6 – NAFNetIRIS (UNet + SimpleGate + Channel Attention + FFT Loss + EMA + TTA)

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

class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, channels: int, ffn_expand: int = 2):
        super().__init__()
        dw_channels = channels * ffn_expand

        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, dw_channels * 2, 1)
        self.conv2 = nn.Conv2d(dw_channels * 2, dw_channels * 2,
                               kernel_size=3, padding=1, groups=dw_channels * 2)
        self.conv3 = nn.Conv2d(dw_channels, channels, 1)
        self.gate  = SimpleGate()
        self.ca    = ChannelAttention(dw_channels)

        self.norm2 = nn.GroupNorm(1, channels)
        self.conv4 = nn.Conv2d(channels, channels * 2, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)

        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1) + 1e-2)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1) + 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x   = self.norm1(x)
        x   = self.conv1(x)
        x   = self.conv2(x)
        x   = self.gate(x)
        x   = self.ca(x)
        x   = self.conv3(x)
        y   = inp + x * self.beta

        x      = self.norm2(y)
        x      = self.conv4(x)
        x1, x2 = x.chunk(2, dim=1)
        x      = x1 * torch.sigmoid(x2)
        x      = self.conv5(x)
        return y + x * self.gamma


class DownsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv    = nn.Conv2d(channels, channels // 2 * 4, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class NAFNetIRIS(nn.Module):
    def __init__(
        self,
        width: int = 32,
        enc_blks: list = None,
        middle_blks: int = 1,
        dec_blks: list = None,
    ):
        super().__init__()
        if enc_blks is None:
            enc_blks = [1, 1, 1, 28]
        if dec_blks is None:
            dec_blks = [1, 1, 1, 1]

        self.intro = nn.Conv2d(1, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        ch = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(DownsampleBlock(ch))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blks)])

        self.decoders = nn.ModuleList()
        self.ups      = nn.ModuleList()
        for n in dec_blks:
            self.ups.append(UpsampleBlock(ch))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        self.ending = nn.Conv2d(ch, width, kernel_size=3, padding=1)

        self.upsample = nn.Sequential(
            nn.Conv2d(width, width * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        feat = self.intro(x)

        enc_features = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            enc_features.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for decoder, up, enc_feat in zip(self.decoders, self.ups, reversed(enc_features)):
            feat = up(feat)
            feat = feat + enc_feat
            feat = decoder(feat)

        feat = self.ending(feat)
        feat = self.upsample(feat)
        return skip + self.refine(feat)


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

    model = NAFNetIRIS(width=32, enc_blks=[1, 1, 1, 28], middle_blks=1, dec_blks=[1, 1, 1, 1]).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"[Team-LML] Loaded checkpoint: {checkpoint_path}", flush=True)
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
      new_data/NoisyLR/NNNNNN.npy  (128x128 float32)
      new_data/GT/NNNNNN.npy       (256x256 float32)
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
                f"[Auto-Train]   {new_data_dir / 'NoisyLR'}  (128x128 inputs)\n"
                f"[Auto-Train]   {new_data_dir / 'GT'}        (256x256 ground truth)\n"
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
        weight_decay   = 1e-4
        ema_decay      = 0.999
        val_fraction   = 0.1
        seed           = 42
        num_workers    = 2
        width          = 32
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
        # Fallback to checkpoints_exp6 if not yet copied to models/
        fallback = script_dir.parent / "checkpoints_exp6" / "best.pt"
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
