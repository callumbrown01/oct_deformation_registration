import os
import glob
import numpy as np
import cv2
import pandas as pd
from tracking_algorithms import TrackingAlgorithms
import itertools
import time

def compute_epe(flow_pred, flow_gt):
    """Endpoint error per pixel"""
    epe_map = np.sqrt(np.sum((flow_pred - flow_gt) ** 2, axis=2))
    return np.mean(epe_map)

def compute_mse(img1, img2):
    """MSE between two images"""
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.mean(diff ** 2)

def warp_image(img, flow):
    """Warp image using optical flow"""
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x + flow[..., 0]
    map_y = y + flow[..., 1]
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped

class FarnebackOptimizer:
    def __init__(self, synthetic_root):
        self.synthetic_root = synthetic_root
        self.tracker = TrackingAlgorithms()
        
        # Load synthetic data
        self.synthetic_samples = self._load_synthetic_data()
        
        # Farneback parameter grid
        self.param_grid = {
            'pyr_scale': [0.2, 0.3, 0.5, 0.7],
            'levels': [3, 4, 5],
            'winsize': [15, 21, 27, 35],
            'iterations': [1, 2, 3],
            'poly_n': [5, 7],
            'poly_sigma': [1.1, 1.3, 1.5, 2.0],
            'flags': [0]
        }
    
    def _load_synthetic_data(self):
        """Load synthetic dataset samples"""
        samples = []
        sample_dirs = sorted(glob.glob(os.path.join(self.synthetic_root, "sample_*")))[:10]  # First 10 samples
        
        for sample_dir in sample_dirs:
            frame_files = sorted(glob.glob(os.path.join(sample_dir, "frame_*.png")))[:3]  # First 3 frames
            if len(frame_files) < 2:
                continue
                
            frames = []
            for f in frame_files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    frames.append(img)
            
            gt_flows = []
            for i in range(len(frames) - 1):
                dx_path = os.path.join(sample_dir, f"dx_{i:03d}_{i+1:03d}.npy")
                dy_path = os.path.join(sample_dir, f"dy_{i:03d}_{i+1:03d}.npy")
                
                if os.path.exists(dx_path) and os.path.exists(dy_path):
                    dx = np.load(dx_path)
                    dy = np.load(dy_path)
                    gt_flows.append(np.stack((dx, dy), axis=-1))
                else:
                    gt_flows.append(None)
            
            if frames and gt_flows:
                samples.append({
                    'name': os.path.basename(sample_dir),
                    'frames': frames,
                    'gt_flows': gt_flows
                })
        
        return samples
    
    def evaluate_params(self, params):
        """Evaluate Farneback performance for given parameters"""
        s_epe_scores = []
        s_mse_scores = []
        
        try:
            # Evaluate on synthetic data
            for sample in self.synthetic_samples:
                frames = sample['frames']
                gt_flows = sample['gt_flows']
                
                for i in range(len(frames) - 1):
                    if gt_flows[i] is None:
                        continue
                        
                    try:
                        flow_pred = self.tracker.run('Farneback', frames[i], frames[i+1], params)
                        
                        # Compute metrics
                        epe = compute_epe(flow_pred, gt_flows[i])
                        warped = warp_image(frames[i+1], flow_pred)
                        mse = compute_mse(frames[i], warped)
                        
                        s_epe_scores.append(epe)
                        s_mse_scores.append(mse)
                        
                    except Exception as e:
                        print(f"Error with params {params}: {e}")
                        s_epe_scores.append(100.0)
                        s_mse_scores.append(10000.0)
            
        except Exception as e:
            print(f"Complete failure with params {params}: {e}")
            return {
                's_epe': 100.0,
                's_mse': 10000.0,
                'combined_score': 1000.0
            }
        
        # Calculate averages
        s_epe_avg = np.mean(s_epe_scores) if s_epe_scores else 100.0
        s_mse_avg = np.mean(s_mse_scores) if s_mse_scores else 10000.0
        
        # Combined score (lower is better)
        combined_score = 0.7 * (s_epe_avg / 50.0) + 0.3 * (s_mse_avg / 5000.0)
        
        return {
            's_epe': s_epe_avg,
            's_mse': s_mse_avg,
            'combined_score': combined_score
        }
    
    def optimize_farneback(self):
        """Optimize Farneback parameters using grid search"""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(param_names, combo)) 
                             for combo in itertools.product(*param_values)]
        
        print(f"Optimizing Farneback with {len(param_combinations)} parameter combinations...")
        start_time = time.time()
        
        best_score = float('inf')
        best_params = None
        results = []
        
        for i, params in enumerate(param_combinations):
            evaluation = self.evaluate_params(params)
            
            results.append({
                'algorithm_type': 'Farneback',
                'params': "; ".join([f"{k}={v}" for k, v in params.items()]),
                's_epe': evaluation['s_epe'],
                's_mse': evaluation['s_mse'],
                'a_mse': evaluation['s_mse'],  # Using synthetic MSE as proxy for actual MSE
                'combined_score': evaluation['combined_score']
            })
            
            if evaluation['combined_score'] < best_score:
                best_score = evaluation['combined_score']
                best_params = params
            
            # Progress updates
            if (i + 1) % 20 == 0 or (i + 1) == len(param_combinations):
                elapsed = time.time() - start_time
                progress = (i + 1) / len(param_combinations)
                eta = elapsed / progress - elapsed if progress > 0 else 0
                
                print(f"Evaluated {i+1}/{len(param_combinations)} combinations. "
                      f"Best score: {best_score:.4f} | "
                      f"Progress: {progress*100:.1f}% | "
                      f"ETA: {eta:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\nBest parameters for Farneback: {best_params}")
        print(f"Best score: {best_score:.4f}")
        print(f"Total optimization time: {total_time:.1f} seconds")
        
        # Save results to CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.synthetic_root, "farneback_optimization_results.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to: {csv_path}")
        
        return best_params, best_score, results

def main():
    print("Starting Farneback optimization...")
    main_start_time = time.time()
    
    # Initialize optimizer
    optimizer = FarnebackOptimizer("synthetic oct data 3")
    
    # Optimize Farneback
    best_params, best_score, results = optimizer.optimize_farneback()
    
    main_total_time = time.time() - main_start_time
    
    print(f"\n{'='*60}")
    print("FARNEBACK OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score:.4f}")
    print(f"Total combinations tested: {len(results)}")
    print(f"Total runtime: {main_total_time:.1f} seconds ({main_total_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()
