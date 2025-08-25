import os
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomImageDataset(Dataset):

    def __init__(self, mode: str, build_div: str, external_idx=None, transform=None) -> None:
        self.dataset_direc = "/content/drive/MyDrive/Project/Data"
        self.mode = mode
        self.transform = transform
        
        if self.mode == "MNIST":
            self.dataset_path = os.path.join(self.dataset_direc, "retinamnist_224.npz")
        elif self.mode == "External":
            self.dataset_path = os.path.join(self.dataset_direc, "MBRSET.npz")
        elif self.mode == "Binary":
            self.dataset_path = os.path.join(self.dataset_direc, "retinamnist_binary.npz")
        elif self.mode == "India":
            self.dataset_path = os.path.join(self.dataset_direc,"India.npz")
        elif self.mode == "Merged":
            self.dataset_path = os.path.join(self.dataset_direc,"merged_india_mnist.npz")
        elif self.mode == "Binary_filtered":
            self.dataset_path = os.path.join(self.dataset_direc, "retinamnist_binary_cleaned_shuffled.npz")
        
        self.data = np.load(self.dataset_path)

        # Case 1: MNIST with 'train' / 'val' split
        if build_div:
            self.x = self.data[f"{build_div}_images"]
            self.y = self.data[f"{build_div}_labels"]

        # Case 2: MBRSET with external index
        else:
            self.x = self.data["x"]
            self.y = self.data["y"]

            assert external_idx is not None, "external_idx must be provided for external validation"
            self.x = self.x[external_idx]
            self.y = self.y[external_idx]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        image = self.x[idx]
        label = self.y[idx]

        # Normalize and convert to torch.Tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(label, dtype=torch.long).squeeze()

        if self.transform:
            image = self.transform(image)

        return image, label
