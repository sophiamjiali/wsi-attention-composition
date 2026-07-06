# ==============================================================================
# Script:           patch_dataset.py
# Purpose:          Patch dataset for loading PNGs (patches) from disk
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/06/2026
# ==============================================================================

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path


class PatchDataset(Dataset):
    """PyTorch dataset for loading pre-extracted PNG patches from disk."""

    def __init__(self, patch_paths: list[Path]):
        self.patch_paths = patch_paths
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.patch_paths)
    
    def __getitem__(self, idx):

        # Load the image and force RGB, converting to tensor
        path = self.patch_paths[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        return {
            'image'     : image,
            'patch_name': path.stem
        }
    
# [END]