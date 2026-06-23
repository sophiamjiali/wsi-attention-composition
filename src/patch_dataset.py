# ==============================================================================
# Script:           patch_dataset.py
# Purpose:          Patch Streaming from TRIDENT coordinates for HoverNet 
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
#
# Notes:            Streams patches from WSIs using TRIDENT .h5 coordinate files
#                   and runs HoverNet inference without saving patches to disk.
# ==============================================================================

import h5py
import openslide

import numpy as np

from torch.utils.data import Dataset
from pathlib import Path


class PatchStreamDataset(Dataset):
    """
    Defines a WSI as a dataset consisting of all patch coordinates extracted
    by TRIDENT
    """

    def __init__(self,
                 wsi_path:   Path, # Path to the sample's WSI
                 h5_path:    Path, # Path to TRIDENT patch coordinates
                 patch_size: int):
        
        # Extract WSI, coordinates, and patch_size
        self.wsi = openslide.OpenSlide(wsi_path)

        with h5py.File(h5_path, 'r') as f: 
            self.coords = f["coords"][:]

        self.patch_size = patch_size

    
    def __len__(self):
        return len(self.coords)
    

    def __getitem__(self, idx):

        # Extract the patch using the coordinates
        x, y = self.coords[idx]
        patch = self.wsi.read_region(
            location = (int(x), int(y)), 
            level    = 0, 
            size     = (self.patch_size, self.patch_size)
        ).convert('RGB')

        return np.array(patch, dtype = np.uint8), int(x), int(y)

        
# [END]