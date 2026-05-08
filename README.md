# Stain Normalization for Cross-Scanner Generalization in Computational Pathology
**BMCS4575 — Final Project | Columbia University, Spring 2026**

---

## Project Overview

This project benchmarks five stain normalization methods — Reinhard, Macenko, Vahadane, Optimal Transport (OT), and StainNet — against an unnormalized baseline for cross-scanner tumor detection on the CAMELYON17 dataset. A ResNet18 classifier is trained from scratch on Center 0 patches with each normalization method applied consistently during both training and evaluation. Models are then evaluated on four held-out test scanners (Centers 1–4) using AUC.

---

## System Requirements

- **OS:** macOS (tested on Apple M-series, MPS backend) or Linux (CUDA)
- **Python:** 3.10+
- **RAM:** ≥ 16 GB recommended
- **Storage:** ~10 GB for dataset + ~300 MB for model checkpoints
- **GPU:** Apple MPS or NVIDIA CUDA (CPU fallback supported but slow)

---

## Installation & Dependencies

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install torch torchvision
pip install scikit-learn pandas numpy matplotlib Pillow
pip install POT                   # Python Optimal Transport
pip install spams                 # Required for Vahadane (dictionary learning)
```

> **Note:** The `stainnet_weights/` directory containing pretrained StainNet weights
> (`camelyon16_dataset/StainNet-Public-centerUni_layer3_ch32.pth`) must be present.
> These weights are included in the submission.

---

## Dataset

The CAMELYON17 dataset is **not included** in this submission due to its size (~10 GB).

**Download:** https://camelyon17.grand-challenge.org/Data/

**Reference:**  
Bandi et al. (2018). *From detection of individual metastases to classification of lymph node status at the patient level: The CAMELYON17 challenge.* IEEE Transactions on Medical Imaging, 38(2), 550–560.



---

## File Descriptions

### Source Code (`src/`)

| File | Description |
|---|---|
| `model.py` | ResNet18 classifier (random init, sigmoid output, binary classification) |
| `dataset.py` | `Camelyon17Dataset` — loads patches from `metadata.csv`, applies transforms |
| `train.py` | Training loop: Adam optimizer, CosineAnnealingLR, best-model checkpoint saving |
| `evaluate.py` | Evaluation loop: loads model checkpoint, runs inference, saves prediction CSVs |
| `baselines.py` | Stain normalizer implementations: Reinhard, Macenko, Vahadane, StainNet |
| `ot_norm.py` | Optimal Transport normalizer using EMD transport in optical density space |
| `utils.py` | Shared utility: `build_normalizer()` factory, `NormalizationTransform` wrapper |
| `visualize.py` | Visual comparison of normalizer outputs on a source/target patch pair |
| `plot_roc_final.py` | Generates ROC curves and per-center AUC bar chart from prediction CSVs |
| `plot_auc_line.py` | Generates line-plot version of per-center AUC (cross-scanner generalization) |
| `generate_figures.py` | Combined figure generation script (comparison + bar chart + ROC) |
| `run_train_and_eval.sh` | Master shell script: trains all 6 models sequentially, then evaluates |

### Output Files (`preds/`)

Prediction CSVs — one per method per test center. Each has two columns: `label` (ground truth) and `pred` (sigmoid probability).

| Pattern | Description |
|---|---|
| `preds_{method}_center{c}.csv` | Predictions for `method` ∈ {none, ot, macenko, reinhard, vahadane, stainnet}, center ∈ {1,2,3,4} |

### Model Checkpoints (`models/`)

| File | Description |
|---|---|
| `best_model_center_0_none.pth` | ResNet18 trained on raw patches (no normalization) |
| `best_model_center_0_ot.pth` | ResNet18 trained with OT normalization |
| `best_model_center_0_macenko.pth` | ResNet18 trained with Macenko normalization |
| `best_model_center_0_reinhard.pth` | ResNet18 trained with Reinhard normalization |
| `best_model_center_0_vahadane.pth` | ResNet18 trained with Vahadane normalization |
| `best_model_center_0_stainnet.pth` | ResNet18 trained with StainNet normalization |

### Figures (`figures/`)

| File | Description |
|---|---|
| `old auc.jpeg` | ROC curves for all 6 methods trained on pre-trained ResNet18 |
| `new auc.jpeg` | ROC curves for all 6 methods trained on ResNet18 (trained from scratch) |
| `old center.jpeg` | Per-center AUC line plot (cross-scanner generalization view) on pre-trained ResNet18 |
| `new center.jpeg` | Per-center AUC line plot (cross-scanner generalization view) on ResNet18 (trained from scratch) |
| `vis comparison.jpeg` | Visual stain normalization comparison on a Center 4 patch |
| `OT architecture.jpeg` | OT normalization algorithm architecture diagram |

### StainNet Weights (`stainnet_weights/`)

Pretrained StainNet normalizer weights trained on CAMELYON16.  
Source: https://github.com/khtao/StainNet

---

## Reproducing Results

### Step 1 — Prepare the dataset

Download and extract CAMELYON17. Place the pre-extracted `patches/` directory and `metadata.csv` in the project root (same level as `src/`).

### Step 2 — Train all models (full pipeline)

```bash
source venv/bin/activate
bash src/run_train_and_eval.sh
```

This will:
- Train one ResNet18 per normalization method on Center 0 (5,000 patches, 10 epochs)
- Evaluate each model on Centers 1–4 (500 patches/center for speed)
- Save prediction CSVs to `results/`
- Save model checkpoints as `best_model_center_0_{method}.pth`

**Estimated runtime:** ~3–4 hours on Apple M-series (MPS), ~1–2 hours on NVIDIA GPU.

### Step 3 — Evaluate only (using included checkpoints)

To skip training and evaluate using the included `.pth` files:

```bash
source venv/bin/activate

for METHOD in none ot macenko reinhard vahadane stainnet; do
    for CENTER in 1 2 3 4; do
        python src/evaluate.py \
            --data_dir patches/ \
            --metadata metadata.csv \
            --test_center $CENTER \
            --model_path models/best_model_center_0_${METHOD}.pth \
            --norm $METHOD \
            --target_img patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png \
            --limit 500
    done
done
```

### Step 4 — Generate plots

```bash
source venv/bin/activate
python src/plot_roc_final.py       # ROC curves + bar chart
python src/plot_auc_line.py        # Line plot
python src/generate_diagrams.py    # Pipeline + OT architecture diagrams
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--train_center` | `0` | Center used for training |
| `--test_center` | `1–4` | Center used for evaluation |
| `--norm` | `none` | Normalization method: `none`, `ot`, `macenko`, `reinhard`, `vahadane`, `stainnet` |
| `--epochs` | `10` | Number of training epochs |
| `--limit` | `5000` (train) / `500` (eval) | Max patches to use (for speed) |
| `--lr` | `1e-3` | Initial learning rate |
| `--batch_size` | `64` | Mini-batch size |
| `--target_img` | — | Path to reference patch for normalizer fitting |

---

## Results Summary

| Method | Center 1 | Center 2 | Center 3 | Center 4 | **Mean AUC** |
|---|---|---|---|---|---|
| No Normalization | 0.940 | 0.966 | 0.973 | 0.503 | 0.845 |
| Optimal Transport | 0.934 | 0.750 | 0.977 | 0.768 | 0.857 |
| Macenko | 0.925 | 0.964 | 0.961 | 0.673 | 0.881 |
| **Reinhard** | **0.943** | 0.954 | **0.974** | **0.919** | **0.947** |
| Vahadane | 0.877 | 0.872 | 0.946 | 0.676 | 0.843 |
| StainNet | 0.910 | 0.940 | 0.907 | 0.718 | 0.869 |

**Key finding:** Reinhard normalization achieves the best cross-scanner generalization, particularly at Center 4 (0.919 vs 0.503 baseline — from near-random to strong performance).

---

## References

1. Tellez et al. (2019). Quantifying the effects of data augmentation and stain color normalization. *Medical Image Analysis.*
2. Bandi et al. (2018). CAMELYON17 challenge. *IEEE Transactions on Medical Imaging.*
3. He et al. (2016). Deep residual learning for image recognition. *CVPR.*
4. Reinhard et al. (2001). Color transfer between images. *IEEE CG&A.*
5. Macenko et al. (2009). A method for normalizing histology slides. *ISBI.*
6. Vahadane et al. (2016). Structure-preserving color normalization. *IEEE TMI.*
7. Flamary et al. (2021). POT: Python Optimal Transport. *JMLR.*
8. Kang et al. (2021). StainNet. *Frontiers in Medicine.*
