"""
run.py  –  Team-LML  |  SEMICON Hackathon 2026 – KLA Problem Statement
AI-Based Restoration of Degraded Images

Model: Experiment 5 – IRISConditioned (FiLM Degradation-Aware Conditioning)
Val PSNR: 29.07 dB  |  Val SSIM: 0.7964

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

class DegradationEncoder(nn.Module):
    """Lightweight CNN that embeds each input's degradation characteristics."""
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # 128→64
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64→32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 32→16
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).flatten(1))


class FiLMResidualBlock(nn.Module):
    """Residual block whose features are modulated by a degradation embedding (FiLM)."""
    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.film  = nn.Linear(embed_dim, channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.relu(self.conv1(x)))
        gamma, beta = self.film(emb).chunk(2, dim=1)
        out = out * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return x + out


class ResidualBlock(nn.Module):
    """Standard residual block (used after upsampling)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class PixelShuffleUpsample(nn.Module):
    """2× spatial upsampling via sub-pixel convolution (PixelShuffle)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv    = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.shuffle(self.conv(x)))


class IRISConditioned(nn.Module):
    """
    Final submitted model – Experiment 5.

    Input : (B, 1, 128, 128) float32
    Output: (B, 1, 256, 256) float32  (not internally clamped; clamp at call site)
    """
    def __init__(self, channels: int = 112, num_res_blocks: int = 16, embed_dim: int = 32):
        super().__init__()
        self.degradation_encoder  = DegradationEncoder(embed_dim)
        self.stem                 = nn.Conv2d(1, channels, 3, padding=1)
        self.stem_relu            = nn.ReLU(inplace=True)
        self.res_blocks           = nn.ModuleList(
            [FiLMResidualBlock(channels, embed_dim) for _ in range(num_res_blocks)]
        )
        self.pre_upsample_conv    = nn.Conv2d(channels, channels, 3, padding=1)
        self.upsample             = PixelShuffleUpsample(channels)
        self.post_upsample_block  = ResidualBlock(channels)
        self.refine               = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        emb  = self.degradation_encoder(x)
        feat = self.stem_relu(self.stem(x))
        for blk in self.res_blocks:
            feat = blk(feat, emb)
        feat = self.upsample(self.pre_upsample_conv(feat))
        feat = self.post_upsample_block(feat)
        return skip + self.refine(feat)


# =============================================================================
#  Helpers
# =============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = IRISConditioned(channels=112, num_res_blocks=16, embed_dim=32).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[Team-LML] Loaded checkpoint: {checkpoint_path}", flush=True)
    return model


def validate_output(arr: np.ndarray, filename: str) -> np.ndarray:
    """
    Ensure output is a finite float32 array in [0, 1].
    Replaces any NaN / Inf with 0 (safe fallback) and clips to [0, 1].
    """
    if not np.isfinite(arr).all():
        n_bad = (~np.isfinite(arr)).sum()
        print(f"  WARNING: {n_bad} non-finite value(s) in {filename} – replacing with 0",
              flush=True)
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr.clip(0.0, 1.0).astype(np.float32)


# =============================================================================
#  Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Team-LML  –  IRIS Image Restoration (SEMICON Hackathon 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python run.py /data/NoisyLR /data/restored_output",
    )
    # Positional arguments matching the required interface:  python run.py <input-dir> <output-dir>
    parser.add_argument("input_dir",  type=str,
                        help="Directory containing NoisyLR .npy input files (128×128, float32)")
    parser.add_argument("output_dir", type=str,
                        help="Directory where restored .npy files will be saved")

    args = parser.parse_args()

    # --- resolve paths --------------------------------------------------------
    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # output directory is created automatically (requirement)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- locate model weights -------------------------------------------------
    script_dir    = Path(__file__).resolve().parent
    model_path    = script_dir / "models" / "best.pt"

    if not model_path.exists():
        print(f"[ERROR] Model checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # --- device ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Team-LML] Device: {device}", flush=True)

    # --- load model -----------------------------------------------------------
    model = load_model(model_path, device)

    # --- enumerate inputs -----------------------------------------------------
    input_files = sorted(
        p for p in input_dir.glob("*.npy") if not p.name.startswith("._")
    )
    if not input_files:
        print(f"[ERROR] No .npy files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[Team-LML] Found {len(input_files)} input file(s). Restoring …", flush=True)

    skipped = 0
    for file_path in tqdm(input_files, desc="Restoring", unit="img"):
        # ---- load input ------------------------------------------------------
        noisy = np.load(file_path).astype(np.float32)

        # Handle (H, W, 1) inputs gracefully
        if noisy.ndim == 3 and noisy.shape[2] == 1:
            noisy = noisy[:, :, 0]

        if noisy.ndim != 2 or noisy.shape[0] != 128 or noisy.shape[1] != 128:
            print(f"  WARNING: skipping {file_path.name} (unexpected shape {noisy.shape})",
                  flush=True)
            skipped += 1
            continue

        # ---- inference -------------------------------------------------------
        t = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,128,128)
        with torch.no_grad():
            pred_t = model(t).clamp(0.0, 1.0)

        pred = pred_t[0, 0].cpu().numpy()  # (256, 256), float32

        # ---- validate output: NaN / Inf guard, value range ------------------
        pred = validate_output(pred, file_path.name)

        # ---- save: same filename as input, directly inside output_dir --------
        # output shape: (256, 256)  →  requirement: (H, W) grayscale
        out_path = output_dir / file_path.name
        np.save(out_path, pred)

    # --- summary --------------------------------------------------------------
    processed = len(input_files) - skipped
    print(f"\n[Team-LML] Done. {processed} image(s) restored -> {output_dir}", flush=True)
    if skipped:
        print(f"           {skipped} file(s) skipped (unexpected shape).", flush=True)


if __name__ == "__main__":
    main()
