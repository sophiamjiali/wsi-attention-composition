# ==============================================================================
# Script:           composition.py
# Purpose:          Extracts patch compositions from HoVer-net output
# Author:           Sophia Mengjia Li
# Affiliation:      CCG Lab, Princess Margaret Cancer Center, UHN, UofT
# Date:             07/02/2026
# ==============================================================================

import logging
from functools import partial

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from typing import Any
from tqdm import tqdm

logger = logging.getLogger(__name__)

PANNUKE_CLASSES = {
    0: "background",
    1: "neoplastic",
    2: "inflammatory",
    3: "connective",
    4: "dead",
    5: "epithelial"
}

def apply_composition_extraction(predictions: list,
                                 nr_types   : int,
                                 n_workers  : int = 4): 
    """
    Extracts discrete cell counts using HoVer-Net's official watershed 
    post-processing. Returns a list of dictionaries containing exact instance 
    counts per cell type.
    """

    # Parallelize the CPU-bound watershed task
    worker_fn = partial(extract_composition, nr_types = nr_types)

    results = []
    with ProcessPoolExecutor(max_workers = n_workers) as pool:
        for r in tqdm(pool.map(worker_fn, predictions),
                      total = len(predictions),
                      desc = "Extracting"):
            results.append(r)

    return pd.DataFrame.from_records(results)


def extract_composition(data_bundle: dict, nr_types: int) -> dict:
    """Processes a single dictionary of inference arrays."""

    # Initialize the base structure of the extracted compositions
    sample_id, patch_name = data_bundle['sample_id'], data_bundle['patch_name']
    base_result = {'sample_id': sample_id, 'patch_name': patch_name}
    empty_comp = {name: 0 for name in PANNUKE_CLASSES.values() 
                  if name != "background"}
    
    try:
        patch_dict = {
            'np': np.transpose(data_bundle['np'], (1, 2, 0)),
            'hv': np.transpose(data_bundle['hv'], (1, 2, 0)),
            'tp': np.transpose(data_bundle['tp'], (1, 2, 0))
        }

        # Apply the provided watershed algorithm
        _, inst_info_dict = process(patch_dict, nr_types = nr_types, 
                                    return_centroids = False)
        
        # Aggregate cell-type composition
        counts = empty_comp.copy()
        for inst_info in inst_info_dict.values():
            type_idx = inst_info['type']
            type_name = PANNUKE_CLASSES.get(type_idx, f"type_{type_idx}")

            if type_name in counts: counts[type_name] += 1

        return {**base_result, **counts, 'status': 'success'}
    
    except Exception as e:
        return {**base_result, **empty_comp, "status": f"error: {str(e)}"}

# [END]