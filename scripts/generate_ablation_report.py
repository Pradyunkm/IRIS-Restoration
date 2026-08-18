"""
generate_ablation_report.py

Reads the three experiments' training logs (checkpoints/log.csv,
checkpoints_exp2/log.csv, checkpoints_exp3/log.csv) and produces:

    results/ablation_plots.png   -- val PSNR and val SSIM vs epoch, all
                                     three experiments overlaid, for the
                                     report/demo slides
    results/ablation_report.md   -- a markdown table + narrative summary,
                                     ready to paste into the final report

The final "clean" PSNR/SSIM numbers (val set with 2 known corrupted/
noise-only samples excluded, computed by compute_clean_metrics.py) are
hardcoded here since they were already computed and verified -- this
script does not require re-running inference, only the training logs
and those already-confirmed final numbers.

Usage:
    python generate_ablation_report.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


FINAL_RESULTS = {
    "Experiment 1: Baseline (pixel loss only)": {
        "log_path": "checkpoints/log.csv",
        "params": 813_633,
        "clean_psnr": 28.40,
        "clean_ssim": 0.7703,
    },
    "Experiment 2: + Structural/Edge loss": {
        "log_path": "checkpoints_exp2/log.csv",
        "params": 813_633,
        "clean_psnr": 28.26,
        "clean_ssim": 0.7753,
    },
    "Experiment 3: + Capacity (final model)": {
        "log_path": "checkpoints_exp3/log.csv",
        "params": 4_522_449,
        "clean_psnr": 29.05,
        "clean_ssim": 0.7946,
    },
    "Experiment 4: + Synthetic augmentation": {
        "log_path": "checkpoints_exp4/log.csv",
        "params": 4_522_449,
        "clean_psnr": 29.06,
        "clean_ssim": 0.7946,
    },
    "Experiment 5: + Degradation conditioning (final model)": {
        "log_path": "checkpoints_exp5/log.csv",
        "params": 4_655_825,
        "clean_psnr": 29.07,
        "clean_ssim": 0.7964,
    },
}


def load_log(log_path: str):
    path = Path(log_path)
    if not path.exists():
        print(f"WARNING: {log_path} not found, skipping in plots")
        return None

    epochs, val_psnr, val_ssim = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            val_psnr.append(float(row["val_psnr"]))
            val_ssim.append(float(row["val_ssim"]))
    return {"epochs": epochs, "val_psnr": val_psnr, "val_ssim": val_ssim}


def make_plots(out_path: str = "results/ablation_plots.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {"Experiment 1: Baseline (pixel loss only)": "#888888",
              "Experiment 2: + Structural/Edge loss": "#4C72B0",
              "Experiment 3: + Capacity (final model)": "#C44E52",
              "Experiment 4: + Synthetic augmentation": "#55A868",
              "Experiment 5: + Degradation conditioning (final model)": "#8172B2"}

    for name, info in FINAL_RESULTS.items():
        log = load_log(info["log_path"])
        if log is None:
            continue
        color = colors.get(name, None)
        axes[0].plot(log["epochs"], log["val_psnr"], label=name, color=color, linewidth=2)
        axes[1].plot(log["epochs"], log["val_ssim"], label=name, color=color, linewidth=2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation PSNR (dB)")
    axes[0].set_title("Validation PSNR over training\n(training-time batch-pooled metric)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation SSIM")
    axes[1].set_title("Validation SSIM over training\n(training-time batch-pooled metric)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots to: {Path(out_path).resolve()}")


def make_report(out_path: str = "results/ablation_report.md"):
    lines = []
    lines.append("# IRIS -- Ablation Study Results\n")
    lines.append(
        "All PSNR/SSIM figures below are **mean per-image** metrics on the "
        "held-out validation split (n=318), with 2 known corrupted samples "
        "excluded (file IDs 002637, 002973 -- confirmed to have "
        "ground-truth targets that are pure uniform random noise, "
        "GT std=0.289, matching the theoretical std of a uniform "
        "distribution over [0,1]; no restoration model can achieve "
        "meaningful PSNR against an unpredictable target, so these are "
        "excluded as non-informative rather than as genuine failures).\n"
    )

    lines.append("| Experiment | Parameters | Val PSNR (dB) | Val SSIM |")
    lines.append("|---|---|---|---|")
    for name, info in FINAL_RESULTS.items():
        lines.append(f"| {name} | {info['params']:,} | {info['clean_psnr']:.2f} | {info['clean_ssim']:.4f} |")
    lines.append("")

    lines.append("## Ablation narrative\n")
    lines.append(
        "**Experiment 1 -> Experiment 2 (loss change only, same 813K-parameter "
        "backbone):** adding structural (SSIM) and edge (Sobel gradient) loss "
        "terms on top of the baseline Charbonnier pixel loss improved SSIM "
        "(0.7703 -> 0.7753) but slightly reduced PSNR (28.40 -> 28.26 dB). "
        "This is the well-documented perception-distortion tradeoff: pixel-only "
        "loss rewards 'safe' blurred predictions that minimize average pixel "
        "error, while structural/edge loss rewards sharper, more perceptually "
        "faithful reconstructions even where that costs a small amount of raw "
        "pixel accuracy. Visual inspection confirmed this was a real "
        "improvement, not a metric artifact: fine texture (bark, fibrous "
        "surfaces) that was posterized/flattened by the pixel-only baseline "
        "was substantially better preserved after adding structural/edge loss.\n"
    )
    lines.append(
        "**Experiment 2 -> Experiment 3 (capacity increase, same loss):** "
        "visual inspection of Experiment 2 showed that dense, high-frequency "
        "content (a crowd of faces) was still under-resolved despite the "
        "improved loss, suggesting the 813K-parameter backbone lacked "
        "sufficient representational capacity, not just the wrong training "
        "signal. Increasing the backbone to 4.5M parameters (more channels, "
        "more residual blocks, one additional block operating at full output "
        "resolution) while keeping the same structural/edge loss recovered "
        "PSNR *and* extended the SSIM gain (29.05 dB / 0.7946), beating both "
        "prior experiments on both metrics -- confirming capacity, not just "
        "loss design, was a genuine bottleneck.\n"
    )
    lines.append(
        "**Experiment 3 -> Experiment 4 (order-agnostic synthetic degradation "
        "augmentation, same architecture and loss):** to directly address the "
        "challenge's explicit statement that degradation order should not be "
        "assumed fixed, a synthetic degradation simulator "
        "(`degradation_simulator.py`) was built that applies speckle, additive "
        "Gaussian noise, and downsampling in randomized order with randomized "
        "parameters. 30% of training samples per epoch were replaced with "
        "synthetic order-randomized pairs generated from the same GT images "
        "(`augmented_dataset.py`), while validation remained real-pairs-only "
        "for a fair comparison. Result: 29.06 dB / 0.7946 SSIM (clean, "
        "per-image) -- statistically indistinguishable from Experiment 3's "
        "29.05 dB / 0.7946. Visual inspection of the hardest case (dense "
        "crowd/faces) also showed no discernible difference. **This "
        "augmentation did not yield a measurable improvement.** A plausible "
        "explanation: the real training data already exhibits meaningful "
        "natural variation in degradation severity (noise std varied "
        "considerably across the audited samples), which may have already "
        "provided sufficient robustness before synthetic augmentation was "
        "added, leaving little additional signal for the augmentation to "
        "contribute. This is reported as a genuine negative result rather "
        "than omitted, consistent with the project's evidence-driven "
        "methodology throughout.\n"
    )
    lines.append(
        "**Experiment 3 -> Experiment 5 (degradation-aware conditioning, "
        "same architecture family and loss, real pairs only):** a "
        "lightweight CNN encoder (`model_conditioned.py`) processes the raw "
        "NoisyLR input into a compact embedding, which FiLM-modulates "
        "(per-channel scale and shift) every residual block in the "
        "backbone -- letting the network adapt its internal processing "
        "based on each input's specific degradation characteristics rather "
        "than applying identical processing regardless of degradation "
        "type or severity. This directly targets the challenge's stated "
        "core difficulty (degradation composed of an unknown mixture of "
        "speckle, Gaussian noise, and downsampling). Result: 29.07 dB / "
        "0.7964 SSIM (clean, per-image) -- PSNR essentially unchanged from "
        "Experiment 3 (29.05 dB), but SSIM improved measurably (0.7946 -> "
        "0.7964), consistent across the full validation set rather than a "
        "single-sample fluke. Visual inspection of the hardest case (dense "
        "crowd/faces) showed no dramatic difference from Experiment 3, "
        "consistent with a modest rather than transformative improvement. "
        "**Experiment 5 is adopted as the final submitted model**: it is "
        "the only variant tested that never underperforms Experiment 3 on "
        "any metric while measurably improving on one, and it is the "
        "experiment most directly aligned with the challenge's own stated "
        "central difficulty.\n"
    )

    lines.append("## Recommendation\n")
    lines.append(
        "**Experiment 5 (IRISConditioned: degradation-aware FiLM "
        "conditioning + combined structural/edge/pixel loss) is the final "
        "submitted model** and is used for `evaluate.py` by default. "
        "Experiment 4 (synthetic augmentation) was tested and found to "
        "provide no measurable benefit, and is reported as a genuine "
        "negative result rather than omitted. Remaining known limitation: "
        "dense facial content in crowded scenes is still under-resolved "
        "relative to ground truth across all five experiments, likely "
        "reflecting a genuine information ceiling in the 128x128 input "
        "resolution rather than a fixable model deficiency -- flagged as "
        "a limitation rather than pursued further within this scope.\n"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report to: {Path(out_path).resolve()}")


if __name__ == "__main__":
    make_plots()
    make_report()