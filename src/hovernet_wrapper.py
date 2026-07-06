# ==============================================================================
# Script:           hovernet_wrapper.py
# Purpose:          General utility functions for using HoverNet
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import logging
import torch

import numpy as np

from pathlib import Path
from src.patch_dataset import PatchDataset
from models.hovernet.net_desc import HoVerNet

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

model = None


def load_hovernet(weights_path: Path,
                  mode        : str,
                  nr_types    : int,
                  device      : torch.device): 
    """
    Initializes HoVer-Net via direct import and loads PanNuke pre-trained 
    weights. Note that PanNuke weights require the 'fast' HoVer-net mode.
    """

    # Instantiate the model architecture directly from the cloned repo
    model = HoVerNet(mode = mode, nr_types = nr_types)

    # Load the checkpoint into CPU memory first to prevent GPU memory spikes
    checkpoint = torch.load(
        f            = weights_path,
        map_location = "cpu",
        weights_only = False
    )

    # Extract the state dictionary and sanitize them (remove 'module' prefix)
    state_dict = checkpoint.get('desc', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load the sanitized weights into the initialized architecture
    model.load_state_dict(state_dict, strict = True)
    model.to(device)
    model.eval()

    return model



def apply_hovernet(group):
    global model
    
    patch_paths = [Path(p) for p in group['patch_path']]
    dataset = PatchDataset(patch_paths)
    loader = DataLoader(dataset, 32, num_workers = 0, pin_memory=False)

    results = []
    with torch.no_grad():
        for images, patch_names in loader:
            images = images.cuda()
            preds = model(images)

            np_out = preds['np'].cpu().numpy()
            hv_out = preds['hv'].cpu().numpy()
            tp_out = preds['tp'].cpu().numpy()

            for i in range(len(patch_names)):
                results.append({
                    'patch_name': patch_names[i],
                    'np': np_out[i],
                    'hv': hv_out[i],
                    'tp': tp_out[i]
                })

    return results


def init_worker(config, device):
    global model

    # Load the HoverNet with the provided weights
    model = load_hovernet(
        weights_path = config.models.hovernet.weights,
        mode         = config.models.hovernet.mode,
        nr_types     = config.models.hovernet.nr_types,
        device       = device
    )

# [END]