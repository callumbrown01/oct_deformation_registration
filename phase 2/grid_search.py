#!/usr/bin/env python3
"""
Grid-search Farneback optical-flow parameters on a synthetic OCT dataset.

Directory layout expected:
dataset_root/
 ├─ set_000/
 │   ├─ frame_000.png
 │   ├─ frame_001.png
 │   ├─ ...
 │   ├─ dx_000_001.npy
 │   ├─ dy_000_001.npy
 │   └─ ...
 ├─ set_001/
 │   └─ ...
 └─ ...

Usage:
    python farneback_grid.py --dataset ./oct_dataset --out farneback_grid_results.csv
"""

import os, glob, itertools, csv, argparse, cv2
import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import tqdm  # progress bar (pip install tqdm if needed)


# --- Helper -----------------------------------------------------------
def warp(img, dx, dy):
    h, w = img.shape
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = [Y + dy, X + dx]
    return map_coordinates(img, coords, order=1, mode='reflect')


def average_epe_for_set(path, fb_kwargs):
    """Return average EPE over all frame pairs in one dataset folder."""
    frames = sorted(glob.glob(os.path.join(path, "frame_*.png")))
    imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frames]
    epe_values = []

    for i in range(len(imgs) - 1):
        img1, img2 = imgs[i], imgs[i + 1]
        dx_gt = np.load(os.path.join(path, f"dx_{i:03d}_{i+1:03d}.npy"))
        dy_gt = np.load(os.path.join(path, f"dy_{i:03d}_{i+1:03d}.npy"))

        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, **fb_kwargs)
        dx_pred, dy_pred = flow[..., 0], flow[..., 1]
        epe = np.sqrt((dx_gt - dx_pred) ** 2 + (dy_gt - dy_pred) ** 2).mean()
        epe_values.append(epe)

    return float(np.mean(epe_values)) if epe_values else np.nan


# --- Main grid search -------------------------------------------------
def main(dataset_root, out_csv):

    # Parameter grid (edit as desired)
    grid = {
        'pyr_scale':  [0.3, 0.5, 0.7],
        'levels':     [1, 2, 3, 4],
        'winsize':    [9, 15, 21],
        'iterations': [1, 3, 5],
        'poly_n':     [5, 7],
        'poly_sigma': [1.1, 1.5, 1.9],
    }

    combos = list(itertools.product(*grid.values()))
    param_names = list(grid.keys())

    # Collect dataset sub-folders
    sets = sorted(
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    )

    results = []

    for combo in tqdm(combos, desc="Grid combos"):
        fb_kwargs = dict(zip(param_names, combo), flags=0)
        epe_all_sets = []

        for subset in sets:
            subset_path = os.path.join(dataset_root, subset)
            epe = average_epe_for_set(subset_path, fb_kwargs)
            epe_all_sets.append(epe)

        avg_epe = float(np.mean(epe_all_sets))
        results.append(list(combo) + [avg_epe])

    # Write CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(param_names + ['avg_EPE'])
        writer.writerows(results)

    print(f"Finished! Wrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Root folder of synthetic OCT datasets")
    parser.add_argument("--out", default="farneback_grid_results.csv", help="CSV output file")
    args = parser.parse_args()

    main(args.dataset, args.out)
