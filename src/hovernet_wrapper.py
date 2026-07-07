# ==============================================================================
# Script:           hovernet_wrapper.py
# Purpose:          General utility functions for using HoverNet
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import logging
import torch

from pathlib import Path
from src.patch_dataset import PatchDataset
from models.hovernet.net_desc import HoVerNet

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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


def apply_hovernet(model,
                   patch_paths: list[Path], 
                   batch_size: int,
                   n_workers: int,
                   device: torch.device) -> list:
    
    # Initialize the dataset and loader for GPU parallelization
    dataset = PatchDataset(patch_paths)
    loader = DataLoader(
        dataset            = dataset,
        batch_size         = batch_size,
        num_workers        = n_workers,
        pin_memory         = True,
        persistent_workers = True
    )

    n_batches = len(loader)
    logger.info(f"- | - Initialized Dataset and DataLoader with "
                f"{n_batches} batches")

    predictions = []
    with torch.inference_mode():
        for idx, batch in enumerate(loader):

            images      = batch['image'].to(device, non_blocking = True)
            patch_names = batch['patch_name']
            
            preds = model(images)

            np_out = preds['np'].cpu().numpy()
            hv_out = preds['hv'].cpu().numpy()
            tp_out = preds['tp'].cpu().numpy()

            for i in range(len(patch_names)):
                predictions.append({
                    'patch_name': patch_names[i],
                    'np': np_out[i],
                    'hv': hv_out[i],
                    'tp': tp_out[i]
                })

            logger.info(f"- |     - Processed batch {idx} / {n_batches} "
                        f"containing {len(patch_names)} total patches")

    logger.info("- | - Successfully generated predictions for all batches")

    return predictions

# [END]