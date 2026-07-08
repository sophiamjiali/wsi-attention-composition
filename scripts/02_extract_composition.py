# ==============================================================================
# Script:           02_extract_composition.py
# Purpose:          Extract per-patch cell-type compositions from HoVer-net .h5
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/08/2026
# ==============================================================================

import argparse as ap

from datetime import datetime
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

from src.composition import extract_sample_compositions
from src.utils import load_config, setup_logger


logger = setup_logger(__name__)


def main():
    args = parse_args()
    log_header(config_path = args.config)
    config = load_config(args.config)

    # Initialize output directories
    dst_dir = Path(config.paths.outputs.composition_dir) / Path(args.project)
    dst_dir.mkdir(parents = True, exist_ok = True)

    # Point the input predictions directory to the specified project
    src_dir = Path(config.paths.outputs.inference_dir) / Path(args.project)

    # Extract compositions from samples serially; tiles are parallelized
    sample_paths = sorted(src_dir.glob("*.h5"))
    logger.info(f"Beginning to process {len(sample_paths)} samples "
                f"from {args.project}")
    
    # Extract compositions for all samples in the project
    n_workers = config.composition.n_workers
    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        for sample_path in sample_paths:
            sample_id = sample_path.stem.removesuffix("_predictions")
            dst_path = dst_dir / f"{sample_id}.parquet"

            extract_sample_compositions(pool, sample_path, dst_path, n_workers)

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
    logger.info("- Pipeline Stage: 02 - Composition Extraction")
    logger.info(f"- Configurations: {config_path}")
    logger.info(f"- Working Directory: {Path.cwd()}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)


def log_footer(cfg):
    logger.info("=" * 60)
    logger.info("Successfully Completed Pipeline Execution")
    logger.info(f"- Prediction Directory: {cfg.paths.outputs.inference_dir}")
    logger.info(f"- Composition Directory: {cfg.paths.outputs.composition_dir}")
    logger.info(f"- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

# [END]