#!/bin/bash
# =============================================================================
#  CAMELYON17 — Correct Normalization Benchmark
#
#  Protocol (Tellez et al.):
#    For each normalization method:
#      1. Train ResNet18 (from scratch) on Center 0 WITH normalization applied
#      2. Evaluate that model on Centers 1-4 WITH the same normalization
#
#  This ensures the model learns features in the normalized domain and the
#  test distribution matches training — giving normalization its best chance
#  to outperform the raw (no normalization) baseline.
# =============================================================================

set -e
source venv/bin/activate

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_IMG="patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png"
RESULTS_FILE="final_results_scratch.txt"
EPOCHS=10
LR=1e-3
BATCH=64
LIMIT="--limit 5000"   # 5000 patches per method — fast but representative
# ──────────────────────────────────────────────────────────────────────────────

METHODS=("ot" "macenko" "reinhard" "vahadane" "stainnet")

echo "============================================================" | tee "$RESULTS_FILE"
echo " CAMELYON17 — Scratch-trained ResNet18 Normalization Benchmark" | tee -a "$RESULTS_FILE"
echo " Epochs: $EPOCHS | LR: $LR | Reference: $TARGET_IMG" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"

# ── Baseline: none — ALREADY TRAINED (20 epochs), eval only ──────────────────
echo "" | tee -a "$RESULTS_FILE"
echo "##############################################" | tee -a "$RESULTS_FILE"
echo "  METHOD: none (pre-trained, eval only)" | tee -a "$RESULTS_FILE"
echo "##############################################" | tee -a "$RESULTS_FILE"

for CENTER in 1 2 3 4; do
    echo "  --- Center $CENTER ---" | tee -a "$RESULTS_FILE"
    python evaluate.py \
        --data_dir patches/ \
        --metadata metadata.csv \
        --test_center $CENTER \
        --model_path best_model_center_0_none.pth \
        --norm none \
        $LIMIT 2>&1 | tee -a "$RESULTS_FILE"
done

# ── Remaining methods: train from scratch + eval ──────────────────────────────
for NORM in "${METHODS[@]}"; do
    MODEL_PATH="best_model_center_0_${NORM}.pth"

    echo "" | tee -a "$RESULTS_FILE"
    echo "##############################################" | tee -a "$RESULTS_FILE"
    echo "  METHOD: $NORM" | tee -a "$RESULTS_FILE"
    echo "##############################################" | tee -a "$RESULTS_FILE"

    echo "[TRAIN] norm=$NORM → $MODEL_PATH" | tee -a "$RESULTS_FILE"
    python train.py \
        --data_dir patches/ \
        --metadata metadata.csv \
        --train_center 0 \
        --batch_size $BATCH \
        --lr $LR \
        --epochs $EPOCHS \
        --norm "$NORM" \
        --target_img "$TARGET_IMG" \
        $LIMIT 2>&1 | tee -a "$RESULTS_FILE"

    echo "" | tee -a "$RESULTS_FILE"
    echo "[EVAL] model=$MODEL_PATH | norm=$NORM" | tee -a "$RESULTS_FILE"

    for CENTER in 1 2 3 4; do
        echo "  --- Center $CENTER ---" | tee -a "$RESULTS_FILE"
        python evaluate.py \
            --data_dir patches/ \
            --metadata metadata.csv \
            --test_center $CENTER \
            --model_path "$MODEL_PATH" \
            --norm "$NORM" \
            --target_img "$TARGET_IMG" \
            $LIMIT 2>&1 | tee -a "$RESULTS_FILE"
    done
done

echo "" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"
echo " All done! Results saved to $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"

