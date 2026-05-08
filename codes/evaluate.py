import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from PIL import Image

from dataset import Camelyon17Dataset
from model import get_model
from utils import NormalizationTransform, build_normalizer

def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize normalizer if requested
    norm_transform = None
    if args.norm != 'none':
        try:
            normalizer = build_normalizer(args.norm, args.target_img)
            norm_transform = NormalizationTransform(normalizer)
        except Exception as e:
            print(f"Error fitting normalizer: {e}")
            return

    # Build eval transforms — same normalization as used during training
    transform_list = []
    if norm_transform is not None:
        transform_list.append(norm_transform)

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose(transform_list)

    print("Loading test dataset...")
    test_dataset = Camelyon17Dataset(
        root_dir=args.data_dir,
        metadata_file=args.metadata,
        center=args.test_center,
        split=1, # 1 for val (using val as test proxy since center 1 has no split 2)
        transform=eval_transform,
        limit=args.limit
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test size: {len(test_dataset)}")

    model = get_model(device)
    if args.model_path.lower() != 'random':
        try:
            state_dict = torch.load(args.model_path, map_location=device)
            # Ensure all keys have the 'model.' prefix for CamelyonClassifier
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('model.'):
                    new_state_dict['model.' + k] = v
                else:
                    new_state_dict[k] = v
            
            # Check if it's a 2-class model
            if 'model.fc.weight' in new_state_dict and new_state_dict['model.fc.weight'].shape[0] == 2:
                model.model.fc = nn.Linear(model.model.fc.in_features, 2).to(device)
                
            model.load_state_dict(new_state_dict)
            print("Loaded model weights.")
        except Exception as e:
            print(f"Could not load model weights from {args.model_path}: {e}")
            return
    else:
        print("Using randomly initialized model for preliminary testing.")
        
        
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating on Center {args.test_center}"):
            inputs, labels = inputs.to(device), labels.float()
            outputs = model(inputs).squeeze(1)
            
            if outputs.dim() > 1 and outputs.shape[1] == 2:
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    auc = roc_auc_score(all_labels, all_preds)
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_binary)
    
    print(f"--- Results for Center {args.test_center} | Norm: {args.norm} ---")
    print(f"AUC:      {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    import os
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame({'label': all_labels, 'pred': all_preds})
    csv_path = f'results/preds_{args.norm}_center{args.test_center}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='patches')
    parser.add_argument('--metadata', type=str, default='metadata.csv')
    parser.add_argument('--test_center', type=int, default=1, help='Center to evaluate on')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--norm', type=str, default='none', choices=['none', 'ot', 'macenko', 'reinhard', 'vahadane', 'stainnet'], help='Normalization method')
    parser.add_argument('--target_img', type=str, default=None, help='Path to reference image (required if norm != none)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size for fast testing')
    args = parser.parse_args()
    
    if args.norm != 'none' and args.target_img is None:
        parser.error("--target_img must be specified when using normalization")
        
    evaluate(args)
