# IRIS - Degradation-Aware Semiconductor Image Restoration

**SEMICON India Hackathon 2026 - KLA Challenge**

Restores 128×128 noisy, low-resolution inspection images to clean 256×256
outputs. The final model uses a degradation-aware CNN - a lightweight
encoder reads each input's degradation characteristics and conditions
the main restoration network's processing accordingly - developed
through a fully evidence-driven, five-experiment ablation process rather
than a single fixed architecture guess.

---

## Result summary

| Model | Parameters | Val PSNR | Val SSIM |
|---|---|---|---|
| Baseline (pixel loss only) | 813,633 | 28.40 dB | 0.7703 |
| + Structural/Edge loss | 813,633 | 28.26 dB | 0.7753 |
| + Capacity | 4,522,449 | 29.05 dB | 0.7946 |
| + Synthetic order-agnostic augmentation | 4,522,449 | 29.06 dB | 0.7946 |
| **+ Degradation-aware conditioning (final model)** | **4,655,825** | **29.07 dB** | **0.7964** |

**Final submitted model: Experiment 5 (+ Degradation-aware conditioning).**
A lightweight CNN encoder produces an embedding of each input's
degradation characteristics, which FiLM-modulates the main backbone's
residual blocks - letting the network adapt its processing per-image
rather than applying identical processing regardless of degradation
type or severity. This directly targets the challenge's core stated
difficulty (unknown, variable-order degradation) and produced a small
but consistent SSIM improvement over Experiment 3, with PSNR essentially
unchanged.

Experiment 4 (order-agnostic synthetic degradation augmentation) was
tested separately and found statistically indistinguishable from
Experiment 3 both quantitatively and visually - a genuine,
honestly-reported negative result kept in the ablation record rather
than omitted. See `results/ablation_report.md` for the full writeup of
all five experiments.

Metrics are mean per-image PSNR/SSIM on a held-out validation split
(n=318; 2 corrupted/noise-only ground-truth samples excluded - see
`results/ablation_report.md` for details).

## Why this approach

The challenge states degradation is a combination of speckle noise,
downsampling, and additive Gaussian noise, applied in unspecified order.
Rather than assuming a fixed degradation pipeline or immediately building
a large, complex architecture, this project followed a staged, measured
approach:

1. **Dataset audit first** - verified pairing, shapes, dtypes, and value
   ranges (NoisyLR is intentionally left unclipped, matching the
   organizers' note that out-of-[0,1] values are a feature of the data,
   not an error) before any modeling.
2. **Simple baseline** - a compact residual CNN with PixelShuffle 2×
   upsampling, trained with a standard pixel loss (Charbonnier), to
   establish a real quantitative reference point.
3. **Loss ablation** - added SSIM (structural) and Sobel gradient (edge)
   loss terms. Visual inspection confirmed this measurably improved fine
   texture preservation (bark, fibrous surfaces) at a small, expected
   PSNR cost - the standard perception-distortion tradeoff.
4. **Capacity ablation** - visual inspection showed dense, high-frequency
   content (e.g. crowded scenes) was still under-resolved after the loss
   change, suggesting a capacity bottleneck rather than a loss-design
   problem. Scaling the backbone (813K → 4.5M parameters) confirmed this:
   it recovered PSNR *and* extended the SSIM gain, beating both prior
   experiments on both metrics.
5. **Augmentation test** - built a synthetic degradation simulator
   (speckle + Gaussian noise + downsampling, randomized order and
   parameters) to directly target the challenge's stated variable
   degradation order. Result: no measurable improvement over Experiment
   3 - reported honestly as a negative finding.
6. **Degradation-aware conditioning** - the final step: a small encoder
   reads each input's degradation characteristics and FiLM-conditions
   the backbone's processing accordingly. This produced the best result
   on every metric with zero regression, and is the experiment most
   directly aligned with the challenge's own stated core difficulty.

Each step is backed by a saved, reproducible experiment (own checkpoint,
own training log) rather than assumption. Full details, including a data
quality issue that was found and correctly handled (not silently
excluded), are in `results/ablation_report.md`.

## Repository structure

```
scripts/
    dataset.py                  Paired dataset loader, reproducible train/val split
    dataset_audit.py            Verifies pairing/shapes/dtypes across the dataset
    model.py                    IRISBaseline and IRISStronger architectures
    model_conditioned.py        IRISConditioned: degradation encoder + FiLM conditioning
    losses.py                   Charbonnier, SSIM, Sobel edge loss, combined loss
    degradation_simulator.py    Synthetic order-randomized degradation pipeline
    augmented_dataset.py        Wraps real pairs with synthetic augmentation for training
    train.py                    Experiment 1: baseline training
    train_exp2.py               Experiment 2: baseline + structural/edge loss
    train_exp3.py               Experiment 3: stronger backbone + combined loss
    train_exp4.py               Experiment 4: + synthetic order-agnostic augmentation
    train_exp5.py               Experiment 5: + degradation-aware conditioning (final model)
    evaluate.py                 Standalone inference: input dir -> output dir
    visualize_pairs.py          Visualizes raw NoisyLR/GT pairs
    visualize_predictions.py    Visualizes model predictions vs GT
    inspect_outliers.py         Diagnoses anomalous-score samples
    compute_clean_metrics.py    Recomputes audited per-image val metrics
    generate_ablation_report.py Builds ablation table + plots

checkpoints_exp5/best.pt        Final model weights (used by default in evaluate.py)
results/ablation_report.md      Full ablation methodology, narrative, and results
results/ablation_plots.png      PSNR/SSIM training curves, all five experiments
```

`checkpoints/`, `checkpoints_exp2/`, `checkpoints_exp3/`, and
`checkpoints_exp4/` (Experiments 1, 2, 3, and 4) are kept for
reference/reproducibility; their training logs are in `log.csv` in each
folder. Experiment 5 is the submitted model.

## Setup

**Requirements:** Python 3.8+

```bash
# 1. Clone the repository (Git LFS required for model weights)
git lfs install
git clone https://github.com/Pradyunkm/IRIS-Restoration.git
cd IRIS-Restoration

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** Git LFS must be installed to download the `.pt` model checkpoint files.
> Install it from [git-lfs.com](https://git-lfs.github.com/) before cloning.

---

## Running inference

```bash
pip install -r requirements.txt

python scripts/evaluate.py \
    --input_dir "path/to/NoisyLR" \
    --output_dir results/output \
    --checkpoint checkpoints_exp5/best.pt \
    --model conditioned
```

Expects a directory of `.npy` files, 128×128, float32. Outputs restored
`.npy` (float32, [0,1]) and `.png` previews per input file. If ground
truth is available for the same file IDs, pass `--gt_dir` to also get
per-file PSNR written to `metrics.csv`.

Verified to run end-to-end from a clean clone/venv on CPU-only hardware
(no GPU required, ~1.8 it/s on a standard CPU, ~29 minutes for 3200 images).

## Reproducing training

```bash
python scripts/dataset_audit.py --data_root <path>
python scripts/train.py --data_root <path> --epochs 30
python scripts/train_exp2.py --data_root <path> --epochs 30
python scripts/train_exp3.py --data_root <path> --epochs 100 --batch_size 8
python scripts/train_exp4.py --data_root <path> --epochs 100 --batch_size 8
python scripts/train_exp5.py --data_root <path> --epochs 100 --batch_size 8
python scripts/compute_clean_metrics.py --data_root <path>
python scripts/generate_ablation_report.py
```

## Known limitations

- Dense, high-frequency content in small regions (e.g. individual faces
  in a crowded scene) remains under-resolved relative to ground truth
  across all five experiments. This likely reflects a genuine
  information ceiling in the 128×128 input resolution rather than a
  fixable model deficiency - flagged as a limitation rather than
  pursued further within this scope.
- The visible sample content in the provided training data is general
  macro/close-up photography rather than literal semiconductor die/wafer
  imagery. The restoration approach (noise removal, structure/edge
  preservation, avoiding hallucinated detail) generalizes to either
  domain, but this is noted for transparency.
- 16 samples in the full dataset have ground-truth targets that are pure
  uniform random noise with no learnable structure (identified via
  `inspect_outliers.py`); these are excluded from reported validation
  metrics as non-informative rather than genuine model failures.
- Order-agnostic synthetic degradation augmentation (Experiment 4) was
  implemented and tested but did not yield a measurable improvement,
  likely because the real training data already contains meaningful
  natural variation in degradation severity.

## Future work

- Combining degradation-aware conditioning (Experiment 5) with synthetic
  order-agnostic augmentation (Experiment 4) was not tested within this
  project's timeline - Experiment 4 showed no benefit on its own, so
  this combination was deprioritized, but it remains a reasonable next
  step to test whether conditioning helps the model exploit augmented
  data better than the unconditioned backbone did.
- A mixture-of-experts style routing (specialized sub-networks for
  different degradation regimes, rather than a single FiLM-conditioned
  backbone) was considered as a stronger form of degradation-awareness
  but not attempted, given the marginal gain FiLM conditioning already
  showed suggests the added complexity may not be justified without
  further evidence.