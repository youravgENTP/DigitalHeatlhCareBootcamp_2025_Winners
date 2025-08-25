import os
import argparse
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from CustomDataset import CustomImageDataset
from Models import ResNet50, ResNet101, ResNet50_MCDropout


def build_transforms(aug=True):
    if not aug:
        return None
    return transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])


def build_model(model_name, num_classes=1, in_ch=3):
    name = model_name.lower()
    if name == "resnet50":
        return ResNet50(input_channel=in_ch, label_num=num_classes)
    if name == "resnet101":
        return ResNet101(input_channel=in_ch, label_num=num_classes)
    if name in ["resnet50_mcdo", "resnet50_mcdropout"]:
        return ResNet50_MCDropout(input_channel=in_ch, label_num=num_classes)
    raise ValueError(f"Unknown model_name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Binary classification training")
    parser.add_argument("--mode", default="Binary", help="Dataset mode: Binary | External | Merged | Binary_filtered")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--model", default="resnet50", help="resnet50 | resnet101 | resnet50_mcdo")
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--save-dir", default="./results")
    parser.add_argument("--no-aug", action="store_true", help="Disable train-time augmentation")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save-dir if hasattr(args, "save-dir") else args.save_dir, exist_ok=True)
    save_dir = args.save_dir  # for readability

    curr_result_direc = os.path.join(save_dir, str(args.attempt))
    os.makedirs(curr_result_direc, exist_ok=True)

    train_tf = build_transforms(aug=(not args.no_aug))
    train_dataset = CustomImageDataset(mode=args.mode, build_div="train", transform=train_tf)
    val_dataset   = CustomImageDataset(mode=args.mode, build_div="val")
    test_dataset  = CustomImageDataset(mode=args.mode, build_div="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args.model, num_classes=1, in_ch=3).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = os.path.join(curr_result_direc, f"checkpoint_v{args.attempt}.pkl")
    best_path = os.path.join(curr_result_direc, f"best_model_v{args.attempt}.pt")
    best_loss = float("inf")
    start_epoch = 0
    metrics_list = []

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        metrics_list = checkpoint["metrics"]
        print(f"Resuming from epoch {start_epoch}...")

    # (Optional) Prepare a few validation samples per class for later Grad-CAM, if needed
    samples_by_class = defaultdict(list)
    for img, label in val_dataset:
        lbl = int(label)
        if len(samples_by_class[lbl]) < 4:
            samples_by_class[lbl].append((img, label))
        if all(len(s) == 4 for s in samples_by_class.values()):
            break

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for _, (img, label) in loop:
            img = img.to(device)
            label = label.to(device).float().unsqueeze(1)  # shape [B,1] for BCEWithLogits

            pred = model(img)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred_class = (torch.sigmoid(pred) >= 0.5).int()
            correct_train += (pred_class == label.int()).sum().item()
            total_train += label.size(0)

            loop.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Train Acc": f"{(correct_train / max(total_train,1)):.4f}"
            })

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = correct_train / max(total_train, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(device)
                label = label.to(device).float().unsqueeze(1)
                pred = model(img)
                loss = criterion(pred, label)
                val_loss += loss.item()

                pred_class = (torch.sigmoid(pred) >= 0.5).int()
                correct_val += (pred_class == label.int()).sum().item()
                total_val += label.size(0)

        val_loss /= max(len(val_loader), 1)
        val_acc = correct_val / max(total_val, 1)

        metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
        }
        metrics_list.append(metrics)

        # Save checkpoint every epoch
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "epoch": epoch,
                "best_loss": best_loss,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics_list
            }, f)

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    print("Training complete.")
    print("Best model:", best_path)


if __name__ == "__main__":
    main()
