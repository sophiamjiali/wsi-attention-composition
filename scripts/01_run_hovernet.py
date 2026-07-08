# ==============================================================================
# Script:           01_run_hovernet.py
# Purpose:          Generate cell-type predictions using HoVer-net
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import torch

import argparse as ap

from pathlib import Path
from datetime import datetime

from src.utils import load_config, setup_logger
from src.hovernet_wrapper import load_hovernet, apply_hovernet
from src.io import save_sample_predictions

logger = setup_logger(__name__)


def main():
    args = parse_args()
    log_header(config_path = args.config)
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize output directories
    dst_dir = Path(config.paths.outputs.inference_dir) / Path(args.project)
    dst_dir.mkdir(parents = True, exist_ok = True)
    
    # Point the input patch directory to the specified project
    src_dir = Path(config.paths.inputs.patch_dir) / Path(args.project)

    # Initialize the model with pretrained weights
    model = load_hovernet(
        weights_path = config.models.hovernet.weights,
        mode         = config.models.hovernet.mode,
        nr_types     = config.models.hovernet.nr_types,
        device       = device
    )
    logger.info("Successfully loaded the HoVer-net model")

    # Process each sample sequentially; GPU will parallelize the tiles
    sample_dirs = [p for p in src_dir.iterdir() if p.is_dir()]
    logger.info(f"Beginning to process {len(sample_dirs)} samples "
                f"from {args.project}")

    for sample_dir in sample_dirs:
        logger.info(f"- | Beginning to process sample {sample_dir.name}")
        patch_paths = [Path(p) for p in sample_dir.glob("*.png")]
        logger.info(f"- | - Detected {len(patch_paths)} patches")

        # Generate predictions for all patches of the sample
        predictions = apply_hovernet(
            model       = model,
            patch_paths = patch_paths,
            batch_size  = config.models.hovernet.batch_size,
            n_workers   = config.models.hovernet.n_workers,
            device      = device
        )

        # Save all sample patches together into an HDF5
        dst_path = dst_dir / f"{sample_dir.name}_predictions.h5"
        save_sample_predictions(predictions, dst_path)

    logger.info("Completed generating predictions for all samples")
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
    main()

# [END]