# ==============================================================================
# Script:           01_run_hovernet.py
# Purpose:          Generate patch compositions using HoverNet
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
#
# Notes:            Streams patches from WSIs using TRIDENT .h5 coordinate files
#                   and runs HoverNet inference without saving patches to disk.
# ==============================================================================

import os
import json
import yaml
import logging
import torch
import sys

import argparse as ap

from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from box import Box


from src.patch_dataset import PatchStreamDataset
from src.hovernet_wrapper import load_hovernet, infer_batch
from src.utils import load_config, find_wsis, find_h5_for_wsi

logger = logging.getLogger(__name__)



def main():
    args = parse_args()
    log_header(config_path = args.config)
    config = load_config(args.config)

    # Initialize directories for HoverNet outputs
    os.makedirs(config.paths.hovernet_outputs, exist_ok = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the HoverNet with the provided weights
    model = load_hovernet(
        model_path   = config.models.hovernet.model,
        weights_path = config.models.hovernet.weights,
        mode         = config.models.hovernet.mode,
        nr_types     = config.models.hovernet.nr_types,
        device       = device
    )

    # Process each WSI individually per dataset


    # change this to return dict[project_name: list[wsi_paths]]
    # process per project and convert this into a function later
    wsi_paths = find_wsis(config.paths.wsi_dir)


    # parallelize the WSIs, but compute per project serially
    # for project in ...keys():
        # extract all WSI 
        # load h5ad
        # process_slide(wsi_path, h5_path, ...)


# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Process raw patches and masks.")
    parser.add_argument("--config", type = str, default = "config.yaml")
    
    return parser.parse_args()


def log_header(config_path):
    logger.info("=" * 60)
    logger.info("Starting Pipeline Execution")
    logger.info("- Pipeline Stage: 01 - Run HoverNet")
    logger.info(f"- Configurations: {config_path}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)


def log_footer(cfg):
    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")
    logger.info(f"- Dataset WSI Directory: {cfg.paths.wsi_dir}")
    logger.info(f"- TRIDENT Coordinate Directory: {cfg.paths.trident_dir}")
    logger.info(f"- Output Directory: {cfg.paths.hovernet_outputs}")
    logger.info(f"- Model Path: {cfg.models.hovernet.model}")
    logger.info(f"- Model Weight Path: {cfg.models.hovernet.weights}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

# [END]