# Team-LML — IRIS Image Restoration & Super-Resolution
**SEMICON India Hackathon 2026 – KLA Problem Statement**

---

## Directory Structure

```
Team-LML/
├── run.py              # Single-file inference entry point
├── auto_retrain.py     # Auto-retraining pipeline (bonus feature)
├── requirements.txt    # Pinned Python dependencies
├── README.md           # This file
└── models/
    └── best.pt         # Trained model weights (Experiment 6 – NAFNet + EMA)
```

---

## Setup

```bash
pip install -r requirements.txt
```

No internet access, API keys, or additional model downloads are required at runtime.

---

## Running Inference

```bash
python run.py <input-dir> <output-dir>
```

| Argument | Description |
|---|---|
| `<input-dir>` | Folder containing NoisyLR `.npy` files (128×128, float32) |
| `<output-dir>` | Folder where restored `.npy` files will be written (auto-created if absent) |

**Example:**
```bash
python run.py /data/test/NoisyLR /data/test/restored
```

### Optional Flags

| Flag | Description |
|---|---|
| `--no_tta` | Disable 8-way test-time augmentation (faster, slightly lower quality) |
| `--no_auto_train` | Disable checking/triggering auto-retrain on new incoming data |

---

## Output Format

For every input file `<input-dir>/NNNNNN.npy` the script produces exactly one output file:

```
<output-dir>/NNNNNN.npy   # shape (256, 256), float32, values in [0, 1]
```

- Shape: `(H, W)` grayscale — `(256, 256)` for standard 128×128 inputs.
- Values: float32, guaranteed to be in `[0, 1]` with no NaN or Inf values.
- Filename: identical to the corresponding input file.

---

## Model & Benchmark Results

| Experiment | Architecture | Parameters | Val PSNR (dB) | Val SSIM |
|---|---|---|---|---|
| Exp 1: Baseline | Simple CNN + pixel loss | 813 K | 28.40 | 0.7703 |
| Exp 2: + Structural/Edge loss | Simple CNN | 813 K | 28.26 | 0.7753 |
| Exp 3: + High-capacity backbone | Deep ResNet | 4.52 M | 29.05 | 0.7946 |
| Exp 4: + Synthetic augmentation | Deep ResNet | 4.52 M | 29.06 | 0.7946 |
| Exp 5: + FiLM conditioning | Conditioned ResNet | 4.66 M | 29.07 | 0.7964 |
| **Exp 6: NAFNet + EMA + FFT** | **NAFNet UNet** | **~17 M** | **26.56** | **0.7957** |

`models/best.pt` contains the EMA-averaged weights for **Experiment 6** (NAFNet).

### Architecture Summary (Experiment 6)
- **NAFNet UNet** — encoder-decoder with SimpleGate activation and Channel Attention, based on "Simple Baselines for Image Restoration" (Chen et al., ECCV 2022).
- **SimpleGate** — splits channels in half and multiplies element-wise; strictly more expressive than ReLU with no dead neurons.
- **Channel Attention** — squeeze-and-excitation style per-channel recalibration.
- **EMA (Exponential Moving Average)** — ensemble of historical weights for consistent improvement.
- **FFT Loss** — frequency-domain L1 loss for sharper high-frequency detail.
- **Test-Time Augmentation** — 8-way geometric ensemble (4 rotations × 2 flips) at inference.
- **PixelShuffle ×2** — sub-pixel convolution upsampling from 128×128 → 256×256.
- **Global residual** — output = bilinear upsample of input + learned correction.

---

## Auto-Retraining Pipeline (Bonus Feature)

The `auto_retrain.py` script enables **continuous model improvement**. When new training data becomes available, it automatically fine-tunes the model and replaces the production weights only if accuracy improves.

### How It Works

1. Drop new paired `.npy` files into `new_data/NoisyLR/` and `new_data/GT/`
2. The script detects new data, validates shapes, and ingests it into the training set
3. Fine-tunes the NAFNet model from the current `best.pt` checkpoint
4. Evaluates on a held-out validation set using EMA weights
5. **Replaces** `models/best.pt` only if the new model has higher PSNR
6. Archives old models for rollback and logs every cycle

### Usage

```bash
# Start the auto-retrain daemon (watches continuously)
python auto_retrain.py

# One-shot mode (run once, then exit)
python auto_retrain.py --once

# Dry run (simulate without training)
python auto_retrain.py --dry_run
```

> **Note:** Auto-retraining requires the full training dataset in `../dataset/train/train/` (NoisyLR/ + GT/ subfolders). This is separate from the core inference pipeline and is not required for `run.py` to work.

---

## Hardware

Runs on **NVIDIA GPU** (recommended) or CPU.
No internet connection, API keys, or user interaction required at runtime.
