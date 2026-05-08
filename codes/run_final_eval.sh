#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

echo "Waiting for the training to finish and the best model to be saved..."
# Training is currently running in the background. This script assumes it's either done or running.
# In a real cluster environment, we'd wait for the PID, but here we just run it directly if it's a unified script.
# Since the training is already running in background process, we could just wait for it.
# Actually, I will just run the evaluations directly here, and tell the user to execute this script AFTER training completes.

echo "==========================================" > final_results.txt
echo " CAMELYON17 CROSS-SCANNER AUC RESULTS " >> final_results.txt
echo "==========================================" >> final_results.txt

echo "Evaluating Baseline (No Normalization)..."
python evaluate.py --data_dir patches/ --test_center 1 --model_path best_model_center_0.pth --norm none --limit 5000 >> final_results.txt 2>&1

echo "Evaluating Optimal Transport (Sinkhorn)..."
python evaluate.py --data_dir patches/ --test_center 1 --model_path best_model_center_0.pth --norm ot --target_img patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png --limit 5000 >> final_results.txt 2>&1

echo "Evaluating Macenko..."
python evaluate.py --data_dir patches/ --test_center 1 --model_path best_model_center_0.pth --norm macenko --target_img patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png --limit 5000 >> final_results.txt 2>&1

echo "Evaluating Reinhard..."
python evaluate.py --data_dir patches/ --test_center 1 --model_path best_model_center_0.pth --norm reinhard --target_img patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png --limit 5000 >> final_results.txt 2>&1

echo "Evaluating Vahadane..."
python evaluate.py --data_dir patches/ --test_center 1 --model_path best_model_center_0.pth --norm vahadane --target_img patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png --limit 5000 >> final_results.txt 2>&1

echo "Evaluating StainNet (WARNING: Uses untrained weights unless model_weights_path is provided in baselines.py)..."
python evaluate.py --data_dir patches/ --test_center 1 --model_path best_model_center_0.pth --norm stainnet --target_img patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png --limit 5000 >> final_results.txt 2>&1

echo "Evaluations complete! See final_results.txt"
