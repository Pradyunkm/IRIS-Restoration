"""
run_exp6.py  –  Team-LML  |  SEMICON Hackathon 2026 – KLA Problem Statement
AI-Based Restoration of Degraded Images

Model: Experiment 6 – NAFNetIRIS
       NAFNet backbone + FFT loss + AdamW + EMA + geometric augmentation
       + Test-Time Augmentation (TTA) at inference

Usage:
    python run_exp6.py <input-dir> <output-dir>

    <input-dir>   Directory containing NoisyLR .npy files  (128×128, float32)
    <output-dir>  Directory where restored .npy files will be written
                  (created automatically if it does not exist)

Test-Time Augmentation (TTA):
    At inference, each image is run through 8 geometric transforms
    (4 rotations × 2 flips). Predictions are inverse-transformed and
    averaged. This gives a consistent +0.1–0.3 dB PSNR improvement
    with zero extra training — pure inference-time ensemble.

    To disable TTA (faster but slightly lower quality):
        python run_exp6.py <input-dir> <output-dir> --no_tta
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import NAFNet architecture (must be in same directory as this script,
# or scripts/ must be on the Python path)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from model_nafnet import NAFNetIRIS


# =============================================================================
#  Test-Time Augmentation helpers
# =============================================================================

def _augment(x: torch.Tensor, mode: int) -> torch.Tensor:
    """Apply one of 8 geometric transforms to (B,1,H,W) tensor."""
    if mode == 0:   return x
    if mode == 1:   return torch.flip(x, dims=[3])
    if mode == 2:   return torch.flip(x, dims=[2])
    if mode == 3:   return torch.rot90(x, k=1, dims=[2, 3])
    if mode == 4:   return torch.rot90(x, k=2, dims=[2, 3])
    if mode == 5:   return torch.rot90(x, k=3, dims=[2, 3])
    if mode == 6:   return torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3])
    if mode == 7:   return torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
    raise ValueError(f"Invalid TTA mode: {mode}")


def _inverse_augment(x: torch.Tensor, mode: int) -> torch.Tensor:
    """Invert the geometric transform applied by _augment."""
    if mode == 0:   return x
    if mode == 1:   return torch.flip(x, dims=[3])
    if mode == 2:   return torch.flip(x, dims=[2])
    if mode == 3:   return torch.rot90(x, k=3, dims=[2, 3])
    if mode == 4:   return torch.rot90(x, k=2, dims=[2, 3])
    if mode == 5:   return torch.rot90(x, k=1, dims=[2, 3])
    if mode == 6:   return torch.flip(x, dims=[3])
    if mode == 7:   return torch.flip(x, dims=[2])
    raise ValueError(f"Invalid TTA mode: {mode}")


@torch.no_grad()
def inference_tta(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """Run TTA over 8 augmentations, average the predictions."""
    preds = []
    for mode in range(8):
        aug_input  = _augment(t, mode)
        aug_pred   = model(aug_input).clamp(0.0, 1.0)
        pred_orig  = _inverse_augment(aug_pred, mode)
        preds.append(pred_orig)
    return torch.stack(preds, dim=0).mean(dim=0)


# =============================================================================
#  Model loading
# =============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Read architecture args from checkpoint if available
    saved_args = ckpt.get("args", {})
    width       = saved_args.get("width",       32)
    middle_blks = saved_args.get("middle_blks", 1)

    model = NAFNetIRIS(
        width=width,
        enc_blks=[1, 1, 1, 28],
        middle_blks=middle_blks,
        dec_blks=[1, 1, 1, 1],
    ).to(device)

    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"[Team-LML] Loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"           Architecture: width={width}, middle_blks={middle_blks}", flush=True)
    return model


# =============================================================================
#  Output validation
# =============================================================================

def validate_output(arr: np.ndarray, filename: str) -> np.ndarray:
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
        description="Team-LML – IRIS Image Restoration Exp 6 (NAFNet + TTA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python run_exp6.py /data/NoisyLR /data/restored_output",
    )
    parser.add_argument("input_dir",  type=str,
                        help="Directory containing NoisyLR .npy input files (128×128, float32)")
    parser.add_argument("output_dir", type=str,
                        help="Directory where restored .npy files will be saved")
    parser.add_argument("--no_tta",  action="store_true",
                        help="Disable test-time augmentation (faster, slightly lower quality)")
    args = parser.parse_args()

    # --- resolve paths --------------------------------------------------------
    input_dir  = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- locate model weights -------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    # Look for checkpoint: submission layout → exp6 → exp5 (fallback)
    candidates = [
        script_dir / "models" / "best.pt",
        script_dir.parent / "checkpoints_exp6" / "best.pt",
        script_dir.parent / "checkpoint_exp6"  / "best.pt",
        script_dir.parent / "checkpoints_exp5" / "best.pt",
    ]
    model_path = next((p for p in candidates if p.exists()), None)

    if model_path is None:
        print(f"[ERROR] No checkpoint found. Tried:", file=sys.stderr)
        for p in candidates:
            print(f"         {p}", file=sys.stderr)
        sys.exit(1)

    # --- device ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Team-LML] Device  : {device}", flush=True)
    print(f"[Team-LML] TTA     : {'disabled' if args.no_tta else 'enabled (8 augmentations)'}", flush=True)

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
            if args.no_tta:
                pred_t = model(t).clamp(0.0, 1.0)
            else:
                pred_t = inference_tta(model, t)

        pred = pred_t[0, 0].cpu().numpy()  # (256, 256), float32

        # ---- validate & save -------------------------------------------------
        pred     = validate_output(pred, file_path.name)
        out_path = output_dir / file_path.name
        np.save(out_path, pred)

    # --- summary --------------------------------------------------------------
    processed = len(input_files) - skipped
    print(f"\n[Team-LML] Done. {processed} image(s) restored -> {output_dir}", flush=True)
    if skipped:
        print(f"           {skipped} file(s) skipped (unexpected shape).", flush=True)


if __name__ == "__main__":
    main()
