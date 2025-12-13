import os
import torch
from pathlib import Path

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
CONFIG = {
    "seed": 1526,
    "img_size": 300,
    "batch_size": 32,
    "num_classes": 7,
    "folds": 5,
    "epochs": 15,
    "lr": 3e-4,
    "device": get_device(),
    "num_workers": min(os.cpu_count(), 8) if os.cpu_count() else 2
}

# Path Management (Local vs Kaggle)
BASE_DIR = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("./data")

PATHS = {
    "train_dir": BASE_DIR / "skin-cancer-mnist-ham10000",
    "metadata": BASE_DIR / "skin-cancer-mnist-ham10000/HAM10000_metadata.csv",
    "ext_test": BASE_DIR / "unified-dataset-for-skin-cancer-classification/Unified_dataset/val",
    "weights_dir": Path("./weights")
}

# Ensure weights directory exists
PATHS["weights_dir"].mkdir(exist_ok=True, parents=True)