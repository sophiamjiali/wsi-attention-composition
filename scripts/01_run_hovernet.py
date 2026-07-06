# ==============================================================================
# Script:           01_run_hovernet.py
# Purpose:          Generate cell-type predictions using HoVer-net
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import h5py
import os
import logging
import torch

import argparse as ap
import torch.multiprocessing as mp

from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from src.utils import load_config, build_patch_manifest
from src.hovernet_wrapper import apply_hovernet, init_worker

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    log_header(config_path = args.config)
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize output directories
    dst_dir = config.paths.outputs.inference_dir / args.project
    dst_dir.mkdir(parents = True, exist_ok = True)
    
    # Point the input patch directory to the specified project
    src_dir = config.paths.inputs.patch_dir / args.project
    
    # Initialize a patch manifest to map training information
    manifest = build_patch_manifest(src_dir, args.project)
    sample_groups = [group for _, group in manifest.groupby('sample_id')]

    n_workers = config.models.hovernet.n_workers
    with ProcessPoolExecutor(max_workers = n_workers, 
                             initializer = init_worker,
                             initargs    = (config, device)) as pool:
        
        # Map the worker function to the sample groups
        for sample_id, sample_results in tqdm(
            zip(manifest['sample_id'].unique(), 
                pool.map(apply_hovernet, sample_groups, chunksize = 1)), 
                total = len(sample_groups)):
            
            out_path = dst_dir / f"{sample_id}.h5"
            
            # Save all patches together grouped by sample into HDF5
            with h5py.File(out_path, 'w') as h5_file:
                for pred in sample_results:

                    # Create a group per patch
                    grp = h5_file.create_group(pred['patch_name'])
                    grp.create_dataset('np',data=pred['np'], compression='gzip')
                    grp.create_dataset('hv',data=pred['hv'], compression='gzip')
                    grp.create_dataset('tp',data=pred['tp'], compression='gzip')
                
        log_footer(config)
        
    return






# =====| Helpers |==============================================================

def parse_args():
    parser = ap.ArgumentParser(description = "Process raw patches and masks.")
    parser.add_argument("--config", type = str, default = "config.yaml")
    parser.add_argument("--project", type = str)
    
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
    logger.info(f"- Patch Directory: {cfg.paths.inputs.patch_dir}")
    logger.info(f"- Embedding Directory: {cfg.paths.inputs.embedding_dir}")
    logger.info(f"- Output Directory: {cfg.paths.outputs.hovernet_outputs}")
    logger.info(f"- Model Path: {cfg.models.hovernet.model}")
    logger.info(f"- Model Weight Path: {cfg.models.hovernet.weights}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)

if __name__ == "__main__":
    mp.set_start_method('spawn', force = True)
    main()

# [END]