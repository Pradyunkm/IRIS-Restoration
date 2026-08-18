# Team-LML — IRIS Image Restoration & Super-Resolution
**SEMICON India Hackathon 2026 – KLA Problem Statement**

---

## Directory Structure

```
Team-LML/
├── run.py            # Single-file inference entry point
├── requirements.txt  # Pinned Python dependencies
├── README.md         # This file
└── models/
    └── best.pt       # Trained model weights (Experiment 5, ~18 MB)
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

| Experiment | Parameters | Val PSNR (dB) | Val SSIM |
|---|---|---|---|
| Exp 1: Baseline (pixel loss) | 813 K | 28.40 | 0.7703 |
| Exp 2: + Structural/Edge loss | 813 K | 28.26 | 0.7753 |
| Exp 3: + High-capacity backbone | 4.52 M | 29.05 | 0.7946 |
| Exp 4: + Synthetic augmentation | 4.52 M | 29.06 | 0.7946 |
| **Exp 5: + FiLM degradation conditioning** | **4.66 M** | **29.07** | **0.7964** |

`models/best.pt` contains the weights for **Experiment 5** (the best-performing model).

### Architecture Summary
- **DegradationEncoder** — lightweight CNN that produces a per-image embedding capturing noise level, speckle severity, etc.
- **FiLM Modulation** — the embedding applies per-channel scale/shift to all 16 residual blocks, letting the network adapt its processing to each input's specific degradation.
- **PixelShuffle ×2** — sub-pixel convolution upsampling from 128×128 → 256×256.
- **Global residual** — output = bilinear upsample of input + learned correction.

---

## Hardware

Runs on **NVIDIA GPU** (recommended) or CPU.  
No internet connection, API keys, or user interaction required at runtime.
