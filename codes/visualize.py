import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from ot_norm import OTSinkhornNormalizer
from baselines import get_baseline_normalizer

def visualize_normalizers(source_path, target_path, output_path='comparison.png'):
    print(f"Loading Source: {source_path}")
    print(f"Loading Target: {target_path}")
    
    source_img = Image.open(source_path).convert('RGB')
    target_img = Image.open(target_path).convert('RGB')
    
    results = {}
    
    # 1. Optimal Transport
    print("Applying Optimal Transport (Sinkhorn)...")
    ot_norm = OTSinkhornNormalizer(subsample_size=1000)
    ot_norm.fit(target_img)
    results['Optimal Transport'] = ot_norm.transform(source_img)
    
    # 2. Reinhard
    print("Applying Reinhard...")
    try:
        reinhard_norm = get_baseline_normalizer('reinhard')
        reinhard_norm.fit(target_img)
        results['Reinhard'] = reinhard_norm.transform(source_img)
    except Exception as e:
        print(f"Reinhard failed: {e}")
        
    # 3. Macenko
    print("Applying Macenko...")
    try:
        macenko_norm = get_baseline_normalizer('macenko')
        macenko_norm.fit(target_img)
        results['Macenko'] = macenko_norm.transform(source_img)
    except Exception as e:
        print(f"Macenko failed: {e}")

    # 4. Vahadane
    print("Applying Vahadane...")
    try:
        vahadane_norm = get_baseline_normalizer('vahadane')
        vahadane_norm.fit(target_img)
        results['Vahadane'] = vahadane_norm.transform(source_img)
    except Exception as e:
        print(f"Vahadane failed: {e}")
        
    # 5. StainNet
    print("Applying StainNet...")
    try:
        stainnet_norm = get_baseline_normalizer('stainnet')
        stainnet_norm.fit(target_img)
        results['StainNet'] = stainnet_norm.transform(source_img)
    except Exception as e:
        print(f"StainNet failed: {e}")

    # Plotting
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    axes[0].imshow(source_img)
    axes[0].set_title('Source (Scanner A)')
    axes[0].axis('off')
    
    axes[1].imshow(target_img)
    axes[1].set_title('Target / Reference (Scanner B)')
    axes[1].axis('off')
    
    axes[2].imshow(results.get('Optimal Transport', np.zeros_like(source_img)))
    axes[2].set_title('Optimal Transport')
    axes[2].axis('off')
    
    axes[3].imshow(results.get('Macenko', np.zeros_like(source_img)))
    axes[3].set_title('Macenko')
    axes[3].axis('off')
    
    axes[4].imshow(results.get('Reinhard', np.zeros_like(source_img)))
    axes[4].set_title('Reinhard')
    axes[4].axis('off')
    
    axes[5].imshow(results.get('Vahadane', np.zeros_like(source_img)))
    axes[5].set_title('Vahadane')
    axes[5].axis('off')
    
    axes[6].imshow(results.get('StainNet', np.zeros_like(source_img)))
    axes[6].set_title('StainNet')
    axes[6].axis('off')
    
    axes[7].axis('off') # Empty
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to source image')
    parser.add_argument('--target', type=str, required=True, help='Path to target/reference image')
    parser.add_argument('--output', type=str, default='comparison.png', help='Path to save the output plot')
    args = parser.parse_args()
    
    visualize_normalizers(args.source, args.target, args.output)
