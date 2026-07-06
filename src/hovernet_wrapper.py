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
from models.hovernet.net_desc import HoVerNet

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

    # Extract teh state dictionary and sanitize them (remove 'module' prefix)
    state_dict = checkpoint.get('desc', checkpoint)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load the sanitized weights into the initialized architecture
    model.load_state_dict(state_dict, strict = True)
    model.to(device)
    model.eval()

    return model

# [END]