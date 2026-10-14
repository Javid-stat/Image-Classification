"""
Central configuration for the project.

Keeping every path and hyperparameter in one place means you only have to
change a value here (not hunt through several files) when you experiment.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# ROOT_DIR is the repository root (one level above src/), so the scripts work
# no matter which folder you run them from.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"          # Fashion-MNIST is downloaded here
MODELS_DIR = ROOT_DIR / "models"      # trained weights (.pth) are saved here
RESULTS_DIR = ROOT_DIR / "results"    # plots and metrics end up here

MODEL_PATH = MODELS_DIR / "fashion_cnn.pth"
HISTORY_PATH = RESULTS_DIR / "history.json"
METRICS_PATH = RESULTS_DIR / "results.json"

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
# The 10 Fashion-MNIST classes, in label order (label 0 = "T-shirt/top", ...).
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = len(CLASSES)

# Mean and standard deviation of the Fashion-MNIST training pixels (scaled to
# [0, 1]). Normalising with these gives inputs with roughly zero mean and unit
# variance, which makes optimisation faster and more stable.
MEAN = (0.2860,)
STD = (0.3530,)

# The official training set has 60,000 images. We hold out 10% of it as a
# validation set to pick the best epoch, and keep the official 10,000-image
# test set untouched until the very end.
VAL_FRACTION = 0.1

# --------------------------------------------------------------------------- #
# Training hyperparameters
# --------------------------------------------------------------------------- #
SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4   # light L2 regularisation through the optimiser

# Worker processes for loading data. On Windows extra worker processes are slow
# to start (and can hang inside notebooks), so we load in the main process there.
NUM_WORKERS = 0 if os.name == "nt" else 2
