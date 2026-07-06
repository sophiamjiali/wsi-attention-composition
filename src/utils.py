# ==============================================================================
# Script:           utils.py
# Purpose:          General utility functions
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import yaml

import pandas as pd

from box import Box
from pathlib import Path

WSI_EXTS = [".svs", ".tif", ".tiff", ".ndpi", ".mrxs"]


def load_config(path: Path) -> Box:
    """Loads the YAML configuration file as a Box."""
    return Box(yaml.safe_load(open(path)), frozen_box = True)


def build_patch_manifest(src_dir: Path, project: str):
    """Initialize a patch manifest with one row per patch."""

    manifest = []

    for sample_dir in src_dir.iterdir():
        for patch in sample_dir.glob("*.png"):
            manifest.append({
                'project': project,
                'sample_id': sample_dir.name,
                'patch_name': patch.name,
                'patch_path': patch
            })

    return pd.DataFrame(manifest)

# [END]