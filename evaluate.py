import glob
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset

from src.config import CONFIG, PATHS
from src.dataset import ExternalDataset, get_transforms
from src.model import build_model
from src.engine import run_tta
from src.utils import plot_evaluation_triplet

def main():
    # Load Class Map logic
    df = pd.read_csv(PATHS["metadata"])
    df_unique = df.drop_duplicates(subset=['lesion_id'], keep='first')
    CLASSES_MAP = {label: idx for idx, label in enumerate(sorted(df_unique['dx'].unique()))}
    class_names = sorted(CLASSES_MAP.keys())
    
    _, val_tf = get_transforms(CONFIG["img_size"])
    
    if not PATHS["ext_test"].exists():
        print("External dataset path not found.")
        return

    print("\n>>> EXTERNAL TEST EVALUATION")
    ext_ds = ExternalDataset(PATHS["ext_test"], CLASSES_MAP, transform=val_tf)
    
    # Subsampling Logic (2000 images)
    if len(ext_ds) > 2000:
        print(f"Subsampling External Test from {len(ext_ds)} to 2000 images...")
        indices = np.random.choice(len(ext_ds), 2000, replace=False)
        ext_ds = Subset(ext_ds, indices)
    
    ext_loader = DataLoader(ext_ds, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"])
    
    # Ensemble Prediction
    model = build_model(CONFIG["num_classes"], CONFIG["device"])
    ens_preds = []
    
    print("Running Ensemble Inference...")
    for fold in range(CONFIG["folds"]):
        w_path = PATHS["weights_dir"] / f"effnetb3_fold{fold}.pth"
        if w_path.exists():
            model.load_state_dict(torch.load(w_path, map_location=CONFIG["device"]))
            ens_preds.append(run_tta(model, ext_loader, CONFIG["device"]))
    
    if not ens_preds:
        print("No weights found. Run train.py first.")
        return

    avg_preds = np.mean(ens_preds, axis=0)
    final_preds = np.argmax(avg_preds, axis=1)
    
    # Retrieve Ground Truth
    y_true_ext = []
    for i in range(len(ext_ds)):
        if isinstance(ext_ds, Subset):
            _, lbl = ext_ds.dataset[ext_ds.indices[i]]
        else:
            _, lbl = ext_ds[i]
        y_true_ext.append(lbl)
    
    y_true_ext = np.array(y_true_ext)
    
    # Filter valid labels
    valid_mask = y_true_ext != -1
    y_true_clean = y_true_ext[valid_mask]
    y_pred_clean = final_preds[valid_mask]

    print("\n=== CLASSIFICATION REPORT (EXTERNAL) ===")
    print(classification_report(y_true_clean, y_pred_clean, target_names=class_names))
    
    plot_evaluation_triplet(y_true_clean, y_pred_clean, CLASSES_MAP)

if __name__ == "__main__":
    main()