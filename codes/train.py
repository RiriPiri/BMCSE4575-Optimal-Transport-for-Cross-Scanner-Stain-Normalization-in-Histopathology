import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from dataset import Camelyon17Dataset
from model import get_model
from utils import NormalizationTransform, build_normalizer


def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Build normalizer (applied to BOTH train and val images so the
    # model learns features in the normalized domain)
    # ------------------------------------------------------------------
    norm_transform = None
    if args.norm != 'none':
        if args.target_img is None:
            raise ValueError("--target_img must be provided when --norm != 'none'")
        normalizer = build_normalizer(args.norm, args.target_img)
        norm_transform = NormalizationTransform(normalizer)

    # ------------------------------------------------------------------
    # Transforms
    # NOTE: ColorJitter is intentionally skipped when normalization is
    # active — the normalizer already standardizes color distribution.
    # ------------------------------------------------------------------
    train_transform_list = []
    if norm_transform is not None:
        train_transform_list.append(norm_transform)

    train_transform_list += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    val_transform_list = []
    if norm_transform is not None:
        val_transform_list.append(norm_transform)
    val_transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    train_transform = transforms.Compose(train_transform_list)
    val_transform = transforms.Compose(val_transform_list)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("Loading datasets...")
    train_dataset = Camelyon17Dataset(
        root_dir=args.data_dir,
        metadata_file=args.metadata,
        center=args.train_center,
        split=0,
        transform=train_transform,
        limit=args.limit
    )

    val_dataset = Camelyon17Dataset(
        root_dir=args.data_dir,
        metadata_file=args.metadata,
        center=args.train_center,
        split=1,
        transform=val_transform,
        limit=args.limit
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    print(f"Normalization: {args.norm}")

    # ------------------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    model = get_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # CosineAnnealingLR: smoothly decays LR to near-zero over all epochs,
    # which helps training from scratch converge without stagnating.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = 0.0
    save_path = f"best_model_center_{args.train_center}_{args.norm}.pth"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs}  (lr={current_lr:.2e})")

        model.train()
        train_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_auc = roc_auc_score(all_labels, all_preds)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model → {save_path}  (Val AUC: {best_val_auc:.4f})")

    print(f"\nTraining complete. Best Val AUC: {best_val_auc:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 from scratch on CAMELYON17.")
    parser.add_argument('--data_dir', type=str, default='patches')
    parser.add_argument('--metadata', type=str, default='metadata.csv')
    parser.add_argument('--train_center', type=int, default=0, help='Center to use for training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size for fast testing')
    parser.add_argument('--norm', type=str, default='none',
                        choices=['none', 'ot', 'macenko', 'reinhard', 'vahadane', 'stainnet'],
                        help='Stain normalization method to apply during training AND validation')
    parser.add_argument('--target_img', type=str, default=None,
                        help='Path to reference image for fitting the normalizer (required if --norm != none)')
    args = parser.parse_args()

    if args.norm != 'none' and args.target_img is None:
        parser.error("--target_img must be specified when using normalization")

    train(args)
