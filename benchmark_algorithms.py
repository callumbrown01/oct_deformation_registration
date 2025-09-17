import os
import glob
import numpy as np
import cv2
import pandas as pd
from tracking_algorithms import TrackingAlgorithms

def compute_epe(flow_pred, flow_gt):
    # Endpoint error per pixel
    epe_map = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=2))
    return np.mean(epe_map)

def compute_mse(img1, img2):
    # MSE between two images
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.mean(diff ** 2)

def warp_image(img, flow):
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x + flow[..., 0]
    map_y = y + flow[..., 1]
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped

def compute_pixel_correlation(img1, img2):
    """Compute pixel correlation between two images"""
    try:
        img1_flat = img1.flatten().astype(np.float32)
        img2_flat = img2.flatten().astype(np.float32)
        
        # Remove any NaN or inf values
        valid_mask = np.isfinite(img1_flat) & np.isfinite(img2_flat)
        if np.sum(valid_mask) > 1:
            correlation = np.corrcoef(img1_flat[valid_mask], img2_flat[valid_mask])[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        else:
            return 0.0
    except Exception:
        return 0.0

def main(samples_root):
    tracker = TrackingAlgorithms()
    algorithms = tracker.get_algorithm_names()
    results = []

    sample_dirs = sorted(glob.glob(os.path.join(samples_root, "sample_*")))
    for sample_dir in sample_dirs:
        print("Processing sample %s", sample_dir)
        frame_files = sorted(glob.glob(os.path.join(sample_dir, "frame_*.png")))
        n_frames = len(frame_files)
        if n_frames < 2:
            continue
        # Load all frames
        frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frame_files]
        # Load ground truth flows
        gt_flows = []
        for i in range(n_frames - 1):
            dx_path = os.path.join(sample_dir, f"dx_{i:03d}_{i+1:03d}.npy")
            dy_path = os.path.join(sample_dir, f"dy_{i:03d}_{i+1:03d}.npy")
            if not (os.path.exists(dx_path) and os.path.exists(dy_path)):
                gt_flows.append(None)
                continue
            dx = np.load(dx_path)
            dy = np.load(dy_path)
            gt_flows.append(np.stack((dx, dy), axis=-1))

        for alg in algorithms:
            epe_list = []
            mse_list = []
            pixel_corr_list = []
            for i in range(n_frames - 1):
                img1 = frames[i]
                img2 = frames[i + 1]
                flow_gt = gt_flows[i]
                if flow_gt is None:
                    continue
                try:
                    flow_pred = tracker.run(alg, img1, img2)
                    # EPE
                    epe = compute_epe(flow_pred, flow_gt)
                    # Warp img2 using predicted flow
                    warped = warp_image(img2, flow_pred)
                    # MSE (subtraction)
                    mse = compute_mse(img1, warped)
                    # Pixel correlation
                    pixel_corr = compute_pixel_correlation(img1, warped)
                    
                    epe_list.append(epe)
                    mse_list.append(mse)
                    pixel_corr_list.append(pixel_corr)
                except Exception as ex:
                    print(f"{sample_dir} | {alg} | frame {i}: Error - {ex}")
            
            if epe_list and mse_list and pixel_corr_list:
                results.append({
                    "sample": os.path.basename(sample_dir),
                    "algorithm": alg,
                    "avg_epe": np.mean(epe_list),
                    "avg_mse": np.mean(mse_list),
                    "avg_pixel_corr": np.mean(pixel_corr_list)
                })

    # Save results to CSV with a unique filename to avoid permission errors
    output_filename = "algorithm_performance_4.csv"
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(samples_root, output_filename), index=False)
    print("Saved results to", os.path.join(samples_root, output_filename))

if __name__ == "__main__":
    main("synthetic oct data 4")

