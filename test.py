# ==============================================================================
# Script:           01_run_hovernet.py
# Purpose:          Generate cell-type predictions using HoVer-net
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import os
import logging
import torch

import argparse as ap

from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime

from src.utils import load_config
from src.hovernet_wrapper import load_hovernet
from src.patch_dataset import PatchDataset
from src.composition import apply_composition_extraction

logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    log_header(config_path = args.config)
    config = load_config(args.config)

    # Initialize directories for HoverNet outputs
    os.makedirs(config.paths.outputs.hovernet_outputs, exist_ok = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the HoverNet with the provided weights
    model = load_hovernet(
        weights_path = config.models.hovernet.weights,
        mode         = config.models.hovernet.mode,
        nr_types     = config.models.hovernet.nr_types,
        device       = device
    )

    # Process per project (e.g. TCGA-BRCA, TCGA-SKCM, etc.)
    for project_dir in sorted(Path(config.paths.inputs.patch_dir).iterdir()):
        if not project_dir.is_dir(): continue

        # Process per sample, nested within each project
        for sample_dir in sorted(Path(project_dir).iterdir()):
            if not sample_dir.is_dir(): continue

            # Fetch all patches associated with the current sample
            patch_paths = sorted(sample_dir.glob("patch_*.png"))
            if not patch_paths: continue

            # Initialize its output directory for HoVerNet composition
            out_dir = (Path(config.paths.outputs.hovernet_outputs) / 
                       project_dir.name / sample_dir.name)
            out_path = out_dir / f"{sample_dir.name}.parquet"
            out_path.parent.mkdir(parents = True, exist_ok = True)
            
            dataset = PatchDataset(patch_paths)
            loader = DataLoader(
                dataset, 
                batch_size  = config.models.hovernet.batch_size,
                num_workers = config.models.hovernet.n_workers,
                shuffle     = False,
                pin_memory  = True
            )

            # Generate compositions for each batch of patches
            inference_list = []
            for batch in loader:

                # Generate cell-type predictions using the HoVer-net
                images      = batch['image'].to(device, non_blocking = True)
                patch_names = batch['patch_name']

                with torch.no_grad(): 
                    preds = model(images)

                    # Unpack the batch into the format expected for extraction
                    for i in range(len(patch_names)):
                        inference_list.append({
                            'sample_id' : sample_dir.name,
                            'patch_name': patch_names[i],
                            'np'        : preds['np'][i].detach().cpu().numpy(),
                            'hv'        : preds['hv'][i].detach().cpu().numpy(),
                            'tp'        : preds['tp'][i].detach().cpu().numpy()
                        })

            # Extract compositions from cell-type predictions
            compositions = apply_composition_extraction(
                predictions = inference_list, 
                nr_types    = config.models.hovernet.nr_types,
                n_workers   = config.models.hovernet.n_workers
            )

            compositions.to_parquet(out_path, index = False)
            logger.info(f"Finished {project_dir.name}/{sample_dir.name}: "
                        f"{len(patch_paths)} patches processed")
                
        log_footer(config)
        
    return

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