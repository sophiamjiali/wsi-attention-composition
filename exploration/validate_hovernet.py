"""
Validate HoVer-Net h5 output structure/fidelity.
Usage (interactive session):
    %run validate_hovernet_h5.py --h5 path/to/sample.h5
or:
    python validate_hovernet_h5.py --h5 path/to/sample.h5 [--nr-types 6] [--expected-count N]

Example: 
    python -c 'from validate_hovernet import validate; validate("/cluster/projects/kumargroup/sophia/wsi-attention-composition/predictions/TCGA-BLCA/TCGA-FD-A5BS-01A-02-TSB.21A9159B-7DD1-4F26-951E-C99C004C1956_predictions.h5", nr_types = 6)'
"""
import argparse
import h5py
import numpy as np
from pathlib import Path


def validate(h5_path, nr_types=None, expected_count=None):
    h5_path = Path(h5_path)
    expected_keys = {'np', 'hv', 'tp'}

    with h5py.File(h5_path, 'r') as f:
        patch_names = list(f.keys())
        n_patches = len(patch_names)
        print(f"[{h5_path.name}] Total patches: {n_patches}")

        if expected_count is not None and n_patches != expected_count:
            print(f"  WARNING: expected {expected_count} patches, found {n_patches}")

        missing_keys, shape_mismatches, nan_inf, empty_tp, tp_range_flags = [], [], [], [], []
        dtype_info = {}

        for name in patch_names:
            grp = f[name]
            keys = set(grp.keys())
            if keys != expected_keys:
                missing_keys.append((name, keys))
                continue

            np_arr, hv_arr, tp_arr = grp['np'][:], grp['hv'][:], grp['tp'][:]

            if np_arr.shape != tp_arr.shape:
                shape_mismatches.append((name, 'np/tp', np_arr.shape, tp_arr.shape))
            if hv_arr.shape[:2] != np_arr.shape[:2]:
                shape_mismatches.append((name, 'hv/np', hv_arr.shape, np_arr.shape))

            for k, arr in [('np', np_arr), ('hv', hv_arr), ('tp', tp_arr)]:
                dtype_info.setdefault(k, set()).add(arr.dtype)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    nan_inf.append((name, k))

            if tp_arr.max() == 0:
                empty_tp.append(name)

            if nr_types is not None:
                uniq = np.unique(tp_arr)
                if uniq.min() < 0 or uniq.max() > nr_types - 1:
                    tp_range_flags.append((name, uniq.tolist()))

        print(f"  Missing/unexpected keys: {len(missing_keys)}", missing_keys[:5] if missing_keys else "")
        print(f"  Shape mismatches: {len(shape_mismatches)}", shape_mismatches[:5] if shape_mismatches else "")
        print(f"  Dtypes observed: {dtype_info}")
        print(f"  Patches with NaN/Inf: {len(nan_inf)}")
        print(f"  Empty (all-background) tp maps: {len(empty_tp)} ({len(empty_tp)/max(n_patches,1):.1%})")
        if nr_types is not None:
            print(f"  tp values outside [0,{nr_types-1}]: {len(tp_range_flags)}", tp_range_flags[:5] if tp_range_flags else "")

        if patch_names:
            g = f[patch_names[0]]
            print(f"\n  Sample patch: {patch_names[0]}")
            print(f"    np: {g['np'].shape} {g['np'].dtype}, range=[{g['np'][:].min()},{g['np'][:].max()}]")
            print(f"    hv: {g['hv'].shape} {g['hv'].dtype}")
            print(f"    tp: {g['tp'].shape} {g['tp'].dtype}, unique={np.unique(g['tp'][:])}")

    return dict(n_patches=n_patches, missing_keys=missing_keys, shape_mismatches=shape_mismatches,
                nan_inf=nan_inf, empty_tp=empty_tp, tp_range_flags=tp_range_flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="Path to sample .h5 file")
    parser.add_argument("--nr-types", type=int, default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    validate(args.h5, nr_types=args.nr_types, expected_count=args.expected_count)