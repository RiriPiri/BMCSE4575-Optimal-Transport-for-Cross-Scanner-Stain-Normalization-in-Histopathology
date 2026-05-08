import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import random
from tqdm import tqdm

from ot_norm import OTSinkhornNormalizer
from baselines import get_baseline_normalizer

def evaluate_quality(data_dir, target_img_path, limit=50):
    # Get random source images from Center 1
    # Center 1 folder might be 'patient_015_node_4' etc. We'll just grab any from patches/ that match center 1
    # Actually, we can just grab random pngs from the data_dir.
    all_images = glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
    random.shuffle(all_images)
    source_images = all_images[:limit]
    
    if not source_images:
        print("No images found in data_dir")
        return
        
    target_img = Image.open(target_img_path).convert('RGB')
    target_np = np.array(target_img)
    
    methods = ['none', 'ot', 'macenko', 'reinhard', 'vahadane', 'stainnet']
    normalizers = {}
    
    print("Fitting normalizers...")
    for method in methods:
        if method == 'none': continue
        if method == 'ot':
            norm = OTSinkhornNormalizer()
        else:
            norm = get_baseline_normalizer(method)
        norm.fit(target_img)
        normalizers[method] = norm
        
    results = {m: {'psnr': [], 'ssim': []} for m in methods}
    
    print(f"Evaluating {len(source_images)} images...")
    for img_path in tqdm(source_images):
        source_img = Image.open(img_path).convert('RGB')
        source_np = np.array(source_img)
        
        # Baseline (No Norm) comparison is identity
        results['none']['psnr'].append(psnr(source_np, source_np))
        results['none']['ssim'].append(1.0)
        
        for method in methods:
            if method == 'none': continue
            try:
                norm_img_np = np.array(normalizers[method].transform(source_img))
                # Compute PSNR and SSIM. Win_size must be smaller than image size, channels=True
                val_psnr = psnr(source_np, norm_img_np, data_range=255)
                # Ensure image is large enough for default win_size=7, else fallback
                min_dim = min(source_np.shape[0], source_np.shape[1])
                win_size = min(7, min_dim)
                if win_size % 2 == 0: win_size -= 1
                val_ssim = ssim(source_np, norm_img_np, data_range=255, channel_axis=2, win_size=win_size)
                
                results[method]['psnr'].append(val_psnr)
                results[method]['ssim'].append(val_ssim)
            except Exception as e:
                # print(f"Error with {method} on {img_path}: {e}")
                pass
                
    # Plotting
    labels = []
    avg_psnr = []
    avg_ssim = []
    
    for method in methods:
        labels.append(method.capitalize() if method != 'ot' else 'Optimal Transport')
        # Filter out NaNs or infinite values if any
        psnrs = [v for v in results[method]['psnr'] if np.isfinite(v)]
        ssims = [v for v in results[method]['ssim'] if np.isfinite(v)]
        
        avg_psnr.append(np.mean(psnrs) if psnrs else 0)
        avg_ssim.append(np.mean(ssims) if ssims else 0)
        
    # Exclude 'None' for plotting since PSNR=inf, SSIM=1
    plot_labels = labels[1:]
    plot_psnr = avg_psnr[1:]
    plot_ssim = avg_ssim[1:]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(plot_labels))
    width = 0.35
    
    color1 = '#2ca02c'
    color2 = '#d62728'
    
    rects1 = ax1.bar(x - width/2, plot_psnr, width, label='PSNR (Higher is better)', color=color1)
    ax1.set_ylabel('PSNR (dB)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, plot_ssim, width, label='SSIM (Closer to 1 is better)', color=color2)
    ax2.set_ylabel('SSIM', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 1.1])
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_labels, rotation=45, ha='right')
    plt.title('Image Quality Preservation (PSNR & SSIM) vs Original Source')
    
    ax1.bar_label(rects1, padding=3, fmt='%.1f')
    ax2.bar_label(rects2, padding=3, fmt='%.3f')
    
    fig.tight_layout()
    plt.savefig('quality_metrics.png', dpi=300)
    print("Saved quality metrics plot to quality_metrics.png")

if __name__ == '__main__':
    # Using the same target image as run_final_eval.sh
    target_img = "patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png"
    evaluate_quality('patches/', target_img, limit=100)
