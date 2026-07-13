# ==============================================================================
# Script:           utils.py
# Purpose:          General utility functions
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             06/23/2026
# ==============================================================================

import logging
import yaml
import sys

from box import Box
from pathlib import Path

WSI_EXTS = [".svs", ".tif", ".tiff", ".ndpi", ".mrxs"]


def load_config(path: Path) -> Box:
    """Loads the YAML configuration file as a Box."""
    return Box(yaml.safe_load(open(path)), frozen_box = True)


def setup_logger(name: str | None = None) -> logging.Logger:
    """
    Create or retrieve a logger with safe default configuration. Enforces
    absolute determinism for logging and does not depend on upstream
    logging configuration.

    Ensures logging works in scripts and SLURM environments where
    no prior logging configuration exists.
    """

    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers = [logging.StreamHandler(sys.stdout)],
        force    = True,
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    return logger

# [END]