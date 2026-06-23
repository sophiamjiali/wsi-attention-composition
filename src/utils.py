# ==============================================================================
# Script:           utils.py
# Purpose:          General utility functions
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import os
import glob
import yaml

from box import Box
from pathlib import Path

WSI_EXTS = [".svs", ".tif", ".tiff", ".ndpi", ".mrxs"]


def load_config(path: Path) -> Box:
    """Loads the YAML configuration file as a Box."""
    return Box(yaml.safe_load(open(path)), frozen_box = True)


# =====| Loading WSI and Coordinates |==========================================

def find_wsis(wsi_dir: Path) -> list[str]:
    """Returns a list of all WSI files in the provided directory."""

    return [
        p for ext in WSI_EXTS
        for p in glob.glob(os.path.join(wsi_dir, '**', f'*{ext}'), 
                           recursive = True)
    ]

def find_h5_for_wsi(wsi_path: Path, data_dir: Path) -> str | None:
    """Finds a WSI's corresponding .h5 file holding TRIDENT coordinates."""

    matches = glob.glob(os.path.join(data_dir, '**', f'{wsi_path.stem}.h5'),
                        recursive = True)
    return matches[0] if matches else None

# [END]