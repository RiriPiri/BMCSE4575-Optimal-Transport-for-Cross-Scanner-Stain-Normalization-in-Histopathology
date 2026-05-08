# Stain Normalization for Cross-Scanner Generalization in Computational Pathology
**BMCS4575 — Final Project | Columbia University, Spring 2026**

---

## Project Overview

Deep learning algorithms trained on data captured from one scanner do not generalize well to data from another scanner, largely owing to the difference in tissue staining procedure during preparation, which is an example of a larger phenomenon referred to as the domain shift problem in deep learning applications \cite{stacke2021}. This research evaluates the effectiveness of five stain normalization techniques, Reinhard color transform \cite{reinhard2001}, Macenko stain matrix decomposition \cite{macenko2009}, Vahadane sparse stain separation \cite{vahadane2016}, Optimal Transport in optical density \cite{flamary2021}, and StainNet \cite{kang2021}, in detecting tumor using the CAMELYON17 dataset \cite{bandi2019} that spans multiple scanners. Pre-trained ResNet18 model and one ResNet18 trained from scratch, \cite{he2016} are trained with normalization applied throughout training and validation phases, according to the methodology used by Tellez et al.\ \cite{tellez2019} on data from Center 0, with performance evaluated in four out-of-domain test sets (Centers 1 to 4) using area under the ROC curve (AUC). Optimal Transport achieved a reasonable performance (AUC 0.857) and marginally performed better the non-normalized approach (AUC 0.845).

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

**Expected structure after extraction:**
```
BMCS4575/
├── patches/               # Pre-extracted 96×96 PNG patches
│   ├── patient_001_node_0/
│   │   └── patch_patient_001_node_0_x_XXXX_y_XXXX.png
│   └── ...
└── metadata.csv           # Patch-level metadata with labels and center 
```

`metadata.csv` **is included** in this submission.

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

### Output Files (`results/`)

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
| `roc_curves_final.png` | ROC curves for all 6 methods, Centers 1–4 combined |
| `auc_per_center_final.png` | Per-center AUC grouped bar chart |
| `auc_per_center_line.png` | Per-center AUC line plot (cross-scanner generalization view) |
| `comparison_final.png` | Visual stain normalization comparison on a Center 4 patch |
| `pipeline_flowchart.png` | End-to-end training and evaluation pipeline diagram |
| `ot_architecture.png` | OT normalization algorithm architecture diagram |

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
| **Optimal Transport** | **0.934** | **0.750** | **0.977** | **0.768** | **0.857** |
| Macenko | 0.925 | 0.964 | 0.961 | 0.673 | 0.881 |
| **Reinhard** | **0.943** | 0.954 | **0.974** | **0.919** | **0.947** |
| Vahadane | 0.877 | 0.872 | 0.946 | 0.676 | 0.843 |
| StainNet | 0.910 | 0.940 | 0.907 | 0.718 | 0.869 |

**Key finding:** Stain normalization can be an important step toward cross-scanner generalizability in tumor detection from H\&E-stained tissue. The simplest color matching approaches can achieve high accuracy and robustness at minimal computational cost. The more sophisticated approaches need to be properly tuned to exploit their potential. Further research is needed on the use of end-to-end learning-based normalization and domain adaptation methods.

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
