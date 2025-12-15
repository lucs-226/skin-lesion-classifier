# 🔬 Skin Lesion Classification (EfficientNet-B3)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-Demo-pink)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Deep Learning pipeline for multi-class skin cancer classification using the **HAM10000** dataset. This repository implements a robust training strategy focusing on reproducibility, data leakage prevention, class imbalance handling, and **Explainable AI (XAI)**.

### Project Objective
The primary goal is to classify dermoscopic images into one of 7 diagnostic categories (e.g., Melanoma, Nevus, Basal Cell Carcinoma).
Beyond high accuracy, the engineering objective was to build a **modular, production-ready codebase** capable of iterating quickly over different architectures and hyperparameters, bridging the gap between experimental notebooks and professional engineering.

### Model Architecture: Why EfficientNet?
We selected **EfficientNet-B3** as the backbone for this task.
* **Compound Scaling:** Unlike ResNets, which scale depth/width arbitrarily, EfficientNet uniformly scales depth, width, and resolution. This results in better feature extraction for the specific input size ($300 \times 300$).
* **Parameter Efficiency:** B3 offers a superior accuracy-to-parameter ratio compared to VGG or ResNet50, reducing training time and inference latency.
* **Transfer Learning:** Pre-trained weights (ImageNet) allow the model to leverage low-level feature detectors immediately, crucial for medical datasets with limited samples.

### Strategy & Methodology

### 1. Data Integrity & Leakage Prevention
The HAM10000 dataset contains multiple images of the *same* lesion. A naive random split would place the same lesion in both Train and Validation sets, leading to **Data Leakage**.
* **Solution:** Deduplication based on `lesion_id`, ensuring all images of a specific lesion reside strictly in one fold.

### 2. Handling Class Imbalance
The dataset is heavily skewed towards Nevi ($nv$).
* **Weighted Random Sampling:** A custom sampler oversamples minority classes during batch generation.
* **Focal Loss:** Replaced standard Cross Entropy with **Focal Loss** ($\gamma=2.0$). This dynamically down-weights easy examples and forces the model to focus on hard, misclassified examples.

### 3. Reliability & XAI
* **TTA (Test Time Augmentation):** Averaging predictions over augmented views (Original + HFlip + VFlip) during inference.
* **Grad-CAM Integration:** We implemented Gradient-weighted Class Activation Mapping to visualize *where* the model is looking, helping to detect artifacts (e.g., rulers, gel bubbles) that cause **Domain Shift**.

---

## 📂 Repository Structure

The project is structured to separate configuration, logic, and execution scripts.

```text
skin-lesion-classifier/
├── data/               # Dataset storage (Ignored by Git)
├── weights/            # Trained model weights (.pth)
├── src/                # Source modules
│   ├── config.py       # Hyperparameters & Paths
│   ├── dataset.py      # Data loading & Transforms
│   ├── model.py        # EfficientNet Architecture
│   ├── engine.py       # Trainer, Loss, TTA
│   ├── utils.py        # Metrics & Plotting
│   └── xai.py          # GradCAM implementation
├── train.py            # Main training script
├── evaluate.py         # Evaluation on external test set
├── app.py              # Gradio Inference Demo
├── skinlesion.ipynb    # Notebook for repository functions         
└── requirements.txt
```

### Usage info
Due to size constraints, datasets are not included in the repo. Create a data folder and download the datasets (via Kaggle API or manually):
* **Option 1:**
```
mkdir -p data/skin-cancer-mnist-ham10000
mkdir -p data/unified-dataset-for-skin-cancer-classification

# Training Data
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 --unzip -p data/skin-cancer-mnist-ham10000

# External Test Data
kaggle datasets download -d cbames/unified-dataset-for-skin-cancer-classification --unzip -p data/unified-dataset-for-skin-cancer-classification
```
