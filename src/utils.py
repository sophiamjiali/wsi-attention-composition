# ==============================================================================
# Script:           utils.py
# Purpose:          General utility functions
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import logging
import h5py
import yaml

from box import Box
from pathlib import Path

logger = logging.getLogger(__name__)

WSI_EXTS = [".svs", ".tif", ".tiff", ".ndpi", ".mrxs"]


def load_config(path: Path) -> Box:
    """Loads the YAML configuration file as a Box."""
    return Box(yaml.safe_load(open(path)), frozen_box = True)


def save_sample_predictions(predictions: list,
                            out_path: Path) -> None:
    """Saves sample predictions as an HDF5."""

    # Automatically writes to the file, doesn't need to be explicitly saved
    with h5py.File(out_path, 'w') as h5_file:
        for pred in predictions:

            # Initialize each patch as an individual group
            grp = h5_file.create_group(pred['patch_name'])
            grp.create_dataset('np',data = pred['np'], compression = 'gzip')
            grp.create_dataset('hv',data = pred['hv'], compression = 'gzip')
            grp.create_dataset('tp',data = pred['tp'], compression = 'gzip')

    logger.info(f"- | - Saved predictions to {out_path.name}")
    return None

# [END]