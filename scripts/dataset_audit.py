import argparse
import json
from pathlib import Path
import numpy as np

def stats(path):
    a = np.load(path, allow_pickle=False)
    floating = np.issubdtype(a.dtype, np.floating)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "nan_count": int(np.isnan(a).sum()) if floating else 0,
        "inf_count": int(np.isinf(a).sum()) if floating else 0,
    }

def main():
    ap = argparse.ArgumentParser(description="Audit paired NoisyLR/GT .npy dataset")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sample_count", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    noisy = root / "NoisyLR"
    gt = root / "GT"
    if not noisy.is_dir() or not gt.is_dir():
        raise FileNotFoundError("data_root must contain NoisyLR and GT folders")

    nfiles = {p.name for p in noisy.glob("*.npy")}
    gfiles = {p.name for p in gt.glob("*.npy")}
    matched = sorted(nfiles & gfiles)
    only_n = sorted(nfiles - gfiles)
    only_g = sorted(gfiles - nfiles)
    if not matched:
        raise RuntimeError("No matching filenames found.")

    names = matched[:min(args.sample_count, len(matched))]
    pairs = []
    for name in names:
        pairs.append({
            "file": name,
            "noisy": stats(noisy / name),
            "gt": stats(gt / name),
        })

    report = {
        "data_root": str(root),
        "noisy_count": len(nfiles),
        "gt_count": len(gfiles),
        "matched_pairs": len(matched),
        "only_in_noisy": only_n[:20],
        "only_in_gt": only_g[:20],
        "pairing_status": "PASS" if not only_n and not only_g else "CHECK",
        "sample_pairs": pairs,
    }
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== IRIS DATASET AUDIT ==========")
    print("Dataset root :", root)
    print("NoisyLR      :", len(nfiles))
    print("GT           :", len(gfiles))
    print("Matched      :", len(matched))
    print("Only NoisyLR :", len(only_n))
    print("Only GT      :", len(only_g))
    print("Pairing      :", report["pairing_status"])
    for p in pairs:
        print(f'{p["file"]}: {p["noisy"]["shape"]} -> {p["gt"]["shape"]} | '
              f'{p["noisy"]["dtype"]} -> {p["gt"]["dtype"]}')
    print("\nReport:", (out / "dataset_audit.json").resolve())

if __name__ == "__main__":
    main()
