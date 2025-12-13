import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_training_curves(history, smooth_factor=0.85):
    def smooth(scalars, weight):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history['train_loss'], alpha=0.3, label='Train Raw')
    plt.plot(epochs, history['val_loss'], alpha=0.3, label='Val Raw')
    plt.plot(epochs, smooth(history['train_loss'], smooth_factor), label='Train Smooth')
    plt.plot(epochs, smooth(history['val_loss'], smooth_factor), label='Val Smooth')
    plt.title("Loss Curves")
    plt.legend()
    plt.show()

def plot_evaluation_triplet(y_true, y_pred, class_map):
    class_names = sorted(class_map.keys())
    cm = confusion_matrix(y_true, y_pred)
    
    # Binary Setup
    MALIGNANT = ['mel', 'bcc', 'akiec']
    mal_idxs = [class_map[k] for k in MALIGNANT if k in class_map]
    
    bin_true = [1 if x in mal_idxs else 0 for x in y_true]
    bin_pred = [1 if x in mal_idxs else 0 for x in y_pred]
    cm_bin = confusion_matrix(bin_true, bin_pred)
    # Normalize Binary
    cm_bin_norm = cm_bin.astype('float') / (cm_bin.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # 1. Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Confusion Matrix (Counts)")
    
    # 2. Recall (Normalized)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Reds', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Normalized Matrix (Recall)")
    
    # 3. Binary (%)
    sns.heatmap(cm_bin_norm, annot=True, fmt='.1%', cmap='Greens', ax=axes[2],
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    axes[2].set_title("Binary Evaluation (Sensitivity & Specificity)")

    plt.tight_layout()
    plt.show()

def visualize_tsne(embeddings, labels, class_map, limit=1500):
    if len(embeddings) > limit:
        idx = np.random.choice(len(embeddings), limit, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    x_embedded = tsne.fit_transform(embeddings)
    
    inv_map = {v: k for k, v in class_map.items()}
    lbl_names = [inv_map[i] for i in labels]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x_embedded[:,0], y=x_embedded[:,1], hue=lbl_names, palette='tab10', alpha=0.8)
    plt.title("t-SNE Projection")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()