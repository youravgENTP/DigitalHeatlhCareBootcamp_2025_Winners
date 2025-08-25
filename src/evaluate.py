import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)

from CustomDataset import CustomImageDataset
from Models import ResNet50, ResNet101, ResNet50_MCDropout


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
    parser = argparse.ArgumentParser(description="Evaluate a trained binary classifier")
    parser.add_argument("--mode", default="Binary", help="Dataset mode: Binary | External | Merged | Binary_filtered")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="resnet50", help="resnet50 | resnet101 | resnet50_mcdo")
    parser.add_argument("--weights", required=True, help="Path to .pt weights (e.g., best_model_vX.pt)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = CustomImageDataset(mode=args.mode, build_div="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, num_classes=1, in_ch=3).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device).view(-1)

            logits = model(img).squeeze(1)
            probs = torch.sigmoid(logits)

            pred = (probs >= 0.5).int()

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy : {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUROC    : {auroc:.4f}")


if __name__ == "__main__":
    main()
