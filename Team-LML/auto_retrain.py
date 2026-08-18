"""
auto_retrain.py  –  Team-LML  |  SEMICON Hackathon 2026 – KLA Problem Statement
Automated Retraining Pipeline with Automatic Model Hot-Swap

This script:
  1. Watches a staging directory (`new_data/`) for new paired NoisyLR + GT files.
  2. Moves matched pairs into the training dataset.
  3. Fine-tunes the IRISConditioned model starting from the current best checkpoint.
  4. Evaluates the new model on the validation set.
  5. Automatically hot-swaps `models/best.pt` IF AND ONLY IF the new model achieves higher PSNR.
  6. Archives replaced models and appends detailed metrics to `retrain_history.csv`.

Usage:
    # Daemon mode — continuously watches for new incoming data
    python auto_retrain.py

    # One-shot mode — checks once and exits
    python auto_retrain.py --once

    # Dry run mode — simulates pipeline without training
    python auto_retrain.py --dry_run

    # Custom options
    python auto_retrain.py --retrain_epochs 30 --poll_interval 60 --lr 5e-5
"""

import sys
from pathlib import Path

# Add scripts directory to path to reuse shared dataset, loss, and model modules
_root = Path(__file__).resolve().parent.parent
_scripts_dir = _root / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

# Import the core implementation from scripts/auto_retrain.py
from auto_retrain import main

if __name__ == "__main__":
    main()
