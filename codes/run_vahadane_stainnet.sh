#!/bin/bash
# Evaluate vahadane using existing scratch-trained .pth (fast mode)
# Train + evaluate stainnet from scratch

set -e
source venv/bin/activate

TARGET_IMG="patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png"
LIMIT="--limit 500"   # small limit so vahadane eval finishes quickly

echo "=== Evaluating Vahadane (using best_model_center_0_vahadane.pth) ==="
for CENTER in 1 2 3 4; do
    echo "  --- Center $CENTER ---"
    python evaluate.py \
        --data_dir patches/ \
        --metadata metadata.csv \
        --test_center $CENTER \
        --model_path best_model_center_0_vahadane.pth \
        --norm vahadane \
        --target_img "$TARGET_IMG" \
        $LIMIT
done

echo ""
echo "=== Training StainNet from scratch (10 epochs, 5000 patches) ==="
python train.py \
    --data_dir patches/ \
    --metadata metadata.csv \
    --train_center 0 \
    --batch_size 64 \
    --lr 1e-3 \
    --epochs 10 \
    --norm stainnet \
    --target_img "$TARGET_IMG" \
    --limit 5000

echo ""
echo "=== Evaluating StainNet ==="
for CENTER in 1 2 3 4; do
    echo "  --- Center $CENTER ---"
    python evaluate.py \
        --data_dir patches/ \
        --metadata metadata.csv \
        --test_center $CENTER \
        --model_path best_model_center_0_stainnet.pth \
        --norm stainnet \
        --target_img "$TARGET_IMG" \
        $LIMIT
done

echo ""
echo "All done! Regenerating plots..."
python plot_roc_final.py
