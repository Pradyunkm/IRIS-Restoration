# Team-LML — IRIS Image Restoration & Super-Resolution
**SEMICON India Hackathon 2026 – KLA Problem Statement**

---

## Directory Structure

```
Team-LML/
├── run.py              # Single-file high-throughput inference entry point (Exp 5 IRISConditioned)
├── auto_retrain.py     # Continuous auto-retraining & model hot-swap runner
├── requirements.txt    # Pinned Python dependencies
├── README.md           # This file
└── models/
    └── best.pt         # Trained model weights (Experiment 5 – IRISConditioned)
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
python run.py dataset/Test_NoisyLR/ results/output/
```

### Performance & Configuration Flags

| Flag | Default | Description |
|---|---|---|
| `--batch_size N` | `16` | Number of images to process in parallel on GPU (increase for higher throughput) |
| `--no_tta` | `False` | Disable 8-way test-time augmentation (faster, ~0.1–0.3 dB quality tradeoff) |
| `--no_fp16` | `False` | Disable FP16 mixed precision autocast (forces standard FP32) |
| `--no_auto_train` | `False` | Disable checking/triggering auto-retrain on new incoming data |
| `--io_workers N` | `4` | Number of threads used for prefetching & background I/O |

---

## Output Format

For every input file `<input-dir>/NNNNNN.npy` the script produces exactly one output file:

```
<output-dir>/NNNNNN.npy   # shape (256, 256), float32, values in [0, 1]
```

- **Shape**: `(256, 256)` float32 array per input file.
- **Values**: Clamped to `[0.0, 1.0]` with zero NaN or Inf values.
- **Filename parity**: Exactly identical stem matching input file name.

---

## Model & Benchmark Results

| Experiment | Architecture | Parameters | Val PSNR (dB) | Val SSIM |
|---|---|---|---|---|
| Exp 1: Baseline | Simple CNN + Charbonnier pixel loss | 813 K | 28.40 | 0.7703 |
| Exp 2: + Structural/Edge loss | Simple CNN + SSIM & Sobel Edge loss | 813 K | 28.26 | 0.7753 |
| Exp 3: + High-capacity backbone | Deep ResNet (16 blocks, PixelShuffle 2×) | 4.52 M | 29.05 | 0.7946 |
| Exp 4: + Synthetic augmentation | Deep ResNet + randomized degradation sim | 4.52 M | 29.06 | 0.7946 |
| **Exp 5: + FiLM conditioning (Final)** | **IRISConditioned (FiLM-modulated ResNet)** | **4.66 M** | **29.07** | **0.7964** |

`models/best.pt` contains the best checkpoint weights for **Experiment 5** (`IRISConditioned`).

### Architecture Summary (Experiment 5 — IRISConditioned)
- **DegradationEncoder**: Lightweight CNN extracting a compact 32-dim latent embedding of noise & blur characteristics directly from the input.
- **FiLM Modulation**: Feature-wise Linear Modulation applied to 16 residual blocks, dynamically scaling ($\gamma$) and shifting ($\beta$) feature representations to adapt to the specific degradation profile.
- **PixelShuffle 2×**: Sub-pixel convolution upsampling from 128×128 → 256×256 with no checkerboard artifacts.
- **Global Residual Connection**: Bilinear 2× interpolation combined with learned high-frequency correction.
- **Fused Batched TTA**: 8-way geometric test-time augmentation executed as a single fused batch on GPU for maximum throughput.

---

## Auto-Retraining Pipeline (Bonus Feature)

The `auto_retrain.py` script enables **continuous model improvement**. When new paired training data is dropped into `new_data/`, it automatically fine-tunes the model and hot-swaps `models/best.pt` only if validation PSNR improves.

### How It Works

1. Drop new paired `.npy` files into `new_data/NoisyLR/` and `new_data/GT/`.
2. The pipeline detects matching pairs, validates shapes, and ingests them into the training set.
3. Fine-tunes the `IRISConditioned` model from the current `best.pt` checkpoint.
4. Evaluates on the held-out validation set.
5. **Replaces** `models/best.pt` only if the new checkpoint achieves a higher PSNR score.
6. Archives replaced checkpoints for full rollback capability and logs metrics to `retrain_history.csv`.

### Usage

```bash
# Start the auto-retrain daemon (watches continuously)
python auto_retrain.py

# One-shot mode (check once, retrain if data present, then exit)
python auto_retrain.py --once

# Dry run (simulate pipeline without training)
python auto_retrain.py --dry_run
```

---

## Hardware & Deployment

- Runs on **NVIDIA GPU** (recommended, FP16 enabled by default) or **CPU**.
- Optimized for high throughput via fused batching and multi-threaded I/O prefetching.
- Fully offline: zero internet access, external API calls, or manual interaction needed.
