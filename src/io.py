# ==============================================================================
# Script:           io.py
# Purpose:          General input/output utility functions
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import h5py

import pandas as pd

from pathlib import Path

from src.utils import setup_logger

logger = setup_logger(__name__)


def save_sample_predictions(predictions: list,
                            dst_path: Path) -> None:
    """Saves sample predictions as an HDF5."""

    # Automatically writes to the file, doesn't need to be explicitly saved
    with h5py.File(dst_path, 'w') as h5_file:
        for pred in predictions:

            # Initialize each patch as an individual group
            grp = h5_file.create_group(pred['patch_name'])
            grp.create_dataset('np',data = pred['np'], compression = 'gzip')
            grp.create_dataset('hv',data = pred['hv'], compression = 'gzip')
            grp.create_dataset('tp',data = pred['tp'], compression = 'gzip')

    logger.info(f"- | - Saved predictions to {dst_path.name}")
    return None


def load_sample_predictions(src_path: Path) -> dict[str, dict]:
    """
    Loads per-sample raw HoVer-net output maps (np, hv, tp) from .h5 as a dictionary keyed by patch name.
    """

    predictions = {}
    with h5py.File(src_path, 'r') as f:
        for patch_name in f.keys():
            assert isinstance(f[patch_name], h5py.Group)
            predictions[patch_name] = {
                'np': f[patch_name]['np'][:],
                'hv': f[patch_name]['hv'][:],
                'tp': f[patch_name]['tp'][:]
            }
    return predictions


def save_sample_compositions(compositions: pd.DataFrame, dst_path: Path):
    compositions.to_parquet(dst_path, index = False)

# [END]