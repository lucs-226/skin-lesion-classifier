import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from src.config import CONFIG, PATHS
from src.utils import set_seed, plot_training_curves, visualize_tsne
from src.dataset import SkinDataset, get_transforms
from src.model import build_model
from src.engine import Trainer, FocalLoss, run_tta

def main():
    set_seed(CONFIG["seed"])
    
    # 1. Data Preparation
    img_pattern = str(PATHS["train_dir"] / "**" / "*.jpg")
    image_map = {Path(x).stem: x for x in glob.glob(img_pattern, recursive=True)}
    
    df = pd.read_csv(PATHS["metadata"])
    df['path'] = df['image_id'].map(image_map)
    df_unique = df.drop_duplicates(subset=['lesion_id'], keep='first').reset_index(drop=True)
    
    CLASSES_MAP = {label: idx for idx, label in enumerate(sorted(df_unique['dx'].unique()))}
    df_unique['label_idx'] = df_unique['dx'].map(CLASSES_MAP)
    
    targets = df_unique['label_idx'].values
    train_tf, val_tf = get_transforms(CONFIG["img_size"])
    
    # 2. Training Loop
    skf = StratifiedKFold(n_splits=CONFIG["folds"], shuffle=True, random_state=CONFIG["seed"])
    feature_storage = [] 
    history_log = {'train_loss': [], 'val_loss': []}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n>>> FOLD {fold+1}/{CONFIG['folds']}")
        
        train_sub = Subset(SkinDataset(df_unique, train_tf), train_idx)
        val_sub = Subset(SkinDataset(df_unique, val_tf), val_idx)
        
        y_train = targets[train_idx]
        class_weights = 1. / np.bincount(y_train)
        sample_weights = [class_weights[t] for t in y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_sub, batch_size=CONFIG["batch_size"], sampler=sampler, num_workers=CONFIG["num_workers"], pin_memory=True)
        val_loader = DataLoader(val_sub, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        
        model = build_model(CONFIG["num_classes"], CONFIG["device"])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
        criterion = FocalLoss(gamma=2.0)
        
        trainer = Trainer(model, optimizer, scheduler, criterion, CONFIG["device"])
        
        best_loss = float('inf')
        fold_train_hist, fold_val_hist = [], []
        
        for epoch in range(CONFIG["epochs"]):
            t_loss = trainer.train_epoch(train_loader)
            v_loss = trainer.validate(val_loader)
            scheduler.step()
            
            fold_train_hist.append(t_loss)
            fold_val_hist.append(v_loss)
            
            if v_loss < best_loss:
                best_loss = v_loss
                torch.save(model.state_dict(), PATHS["weights_dir"] / f"effnetb3_fold{fold}.pth")
            
            print(f"Ep {epoch+1} | T: {t_loss:.4f} | V: {v_loss:.4f}")

        history_log['train_loss'] = fold_train_hist
        history_log['val_loss'] = fold_val_hist
        
        # Save Validation Embeddings for t-SNE
        model.load_state_dict(torch.load(PATHS["weights_dir"] / f"effnetb3_fold{fold}.pth"))
        model.classifier = nn.Identity()
        model.eval()
        with torch.no_grad():
            for i, (imgs, lbls) in enumerate(val_loader):
                if i > 5: break 
                emb = model(imgs.to(CONFIG["device"])).cpu().numpy()
                feature_storage.append((emb, lbls.numpy()))

    print("\nTraining Complete.")
    plot_training_curves(history_log)
    
    if feature_storage:
        all_embs = np.concatenate([x[0] for x in feature_storage])
        all_lbls = np.concatenate([x[1] for x in feature_storage])
        visualize_tsne(all_embs, all_lbls, CLASSES_MAP)

if __name__ == "__main__":
    main()