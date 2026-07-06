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


# [END]