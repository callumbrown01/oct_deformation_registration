import os
import glob
import numpy as np
import cv2
import pandas as pd
from tracking_algorithms import TrackingAlgorithms
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

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

class AlgorithmOptimizer:
    def __init__(self, synthetic_root, real_root=None, weight_s_epe=0.4, weight_s_mse=0.3, weight_a_mse=0.3):
        self.synthetic_root = synthetic_root
        self.real_root = real_root
        self.weight_s_epe = weight_s_epe
        self.weight_s_mse = weight_s_mse
        self.weight_a_mse = weight_a_mse
        self.tracker = TrackingAlgorithms()
        
        # Load synthetic data (optimized - only load what we need)
        self.synthetic_samples = self._load_synthetic_data()
        
        # Load real data if provided
        self.real_samples = self._load_real_data() if real_root else []
        
        # Reduced parameter grids for faster optimization
        self.param_grids = {
            'TVL1': {
                'tau': [0.1, 0.25, 0.5],
                'lambda_': [0.05, 0.15, 0.3],
                'theta': [0.3, 0.5],
                'nscales': [3, 4],
                'warps': [3, 5],
                'epsilon': [0.01, 0.02]
            },
            'DIS': {
                'finest_scale': [0, 1],
                'patch_size': [4, 8],
                'patch_stride': [2, 4],
                'grad_descent_iter': [8, 12],
                'var_refine_iter': [3, 5],
                'var_refine_alpha': [20.0, 40.0],
                'var_refine_delta': [2.0, 5.0],
                'var_refine_gamma': [5.0, 10.0],
                'use_mean_normalization': [True],
                'use_spatial_propagation': [False]
            },
            'Farneback': {
                'pyr_scale': [0.3, 0.5, 0.7],
                'levels': [3, 4],
                'winsize': [15, 21],
                'iterations': [1, 2],
                'poly_n': [5, 7],
                'poly_sigma': [1.1, 1.5]
            }
        }
    
    def _load_synthetic_data(self):
        """Load synthetic dataset samples - optimized loading"""
        samples = []
        sample_dirs = sorted(glob.glob(os.path.join(self.synthetic_root, "sample_*")))[:5]  # Only first 5 for speed
        
        for sample_dir in sample_dirs:
            frame_files = sorted(glob.glob(os.path.join(sample_dir, "frame_*.png")))[:3]  # Only first 3 frames
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
    
    def _load_real_data(self):
        """Load real dataset samples - placeholder for now"""
        return []
    
    def evaluate_params_single(self, algorithm, params):
        """Evaluate algorithm performance for a single parameter set - optimized"""
        s_epe_scores = []
        s_mse_scores = []
        a_mse_scores = []
        
        try:
            # Evaluate on synthetic data
            for sample in self.synthetic_samples:
                frames = sample['frames']
                gt_flows = sample['gt_flows']
                
                for i in range(len(frames) - 1):
                    if gt_flows[i] is None:
                        continue
                        
                    try:
                        flow_pred = self.tracker.run(algorithm, frames[i], frames[i+1], params)
                        
                        # Compute metrics efficiently
                        epe = compute_epe(flow_pred, gt_flows[i])
                        warped = warp_image(frames[i+1], flow_pred)
                        mse = compute_mse(frames[i], warped)
                        
                        s_epe_scores.append(epe)
                        s_mse_scores.append(mse)
                        
                    except Exception:
                        # Silent failure with penalty
                        s_epe_scores.append(100.0)
                        s_mse_scores.append(10000.0)
            
            # Real data evaluation (placeholder - implement based on your real data)
            # For now, just use synthetic MSE as proxy
            a_mse_scores = s_mse_scores.copy() if not self.real_samples else [0.0]
            
        except Exception:
            # Complete failure
            return {
                's_epe': 100.0,
                's_mse': 10000.0,
                'a_mse': 10000.0,
                'combined_score': 1000.0
            }
        
        # Calculate averages
        s_epe_avg = np.mean(s_epe_scores) if s_epe_scores else 100.0
        s_mse_avg = np.mean(s_mse_scores) if s_mse_scores else 10000.0
        a_mse_avg = np.mean(a_mse_scores) if a_mse_scores else 10000.0
        
        # Three-factor combined score (normalized)
        s_epe_norm = s_epe_avg / 50.0  # Normalize EPE (typical range 0-50)
        s_mse_norm = s_mse_avg / 5000.0  # Normalize synthetic MSE
        a_mse_norm = a_mse_avg / 5000.0  # Normalize actual MSE
        
        combined_score = (self.weight_s_epe * s_epe_norm + 
                         self.weight_s_mse * s_mse_norm + 
                         self.weight_a_mse * a_mse_norm)
        
        return {
            's_epe': s_epe_avg,
            's_mse': s_mse_avg,
            'a_mse': a_mse_avg,
            'combined_score': combined_score
        }
    
    def evaluate_params_batch(self, algorithm, param_combinations_batch):
        """Evaluate multiple parameter combinations in batch"""
        results = []
        for params_dict in param_combinations_batch:
            evaluation = self.evaluate_params_single(algorithm, params_dict)
            results.append((params_dict, evaluation))
        return results
    
    def optimize_algorithm(self, algorithm):
        """Optimize parameters for a specific algorithm using parallel grid search"""
        if algorithm not in self.param_grids:
            print(f"No parameter grid defined for {algorithm}")
            return None
        
        param_grid = self.param_grids[algorithm]
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(param_names, combo)) 
                             for combo in itertools.product(*param_values)]
        
        print(f"Optimizing {algorithm} with {len(param_combinations)} parameter combinations...")
        start_time = time.time()
        
        best_score = float('inf')
        best_params = None
        results = []
        
        # Parallel evaluation with ThreadPoolExecutor (better for I/O bound tasks)
        batch_size = max(1, len(param_combinations) // (mp.cpu_count() * 2))
        batches = [param_combinations[i:i + batch_size] 
                  for i in range(0, len(param_combinations), batch_size)]
        
        with ThreadPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
            future_to_batch = {
                executor.submit(self.evaluate_params_batch, algorithm, batch): batch 
                for batch in batches
            }
            
            completed = 0
            for future in future_to_batch:
                try:
                    batch_results = future.result(timeout=300)  # 5 minute timeout per batch
                    
                    for params_dict, evaluation in batch_results:
                        results.append({
                            'algorithm': algorithm,
                            'params': params_dict,
                            's_epe': evaluation['s_epe'],
                            's_mse': evaluation['s_mse'],
                            'a_mse': evaluation['a_mse'],
                            'combined_score': evaluation['combined_score']
                        })
                        
                        if evaluation['combined_score'] < best_score:
                            best_score = evaluation['combined_score']
                            best_params = params_dict
                    
                    completed += len(batch_results)
                    
                    # Progress updates
                    if completed % 20 == 0 or completed == len(param_combinations):
                        elapsed = time.time() - start_time
                        progress = completed / len(param_combinations)
                        eta = elapsed / progress - elapsed if progress > 0 else 0
                        
                        print(f"Evaluated {completed}/{len(param_combinations)} combinations. "
                              f"Best score: {best_score:.4f} | "
                              f"Progress: {progress*100:.1f}% | "
                              f"ETA: {eta:.1f}s")
                        
                except Exception as e:
                    print(f"Batch evaluation failed: {e}")
                    continue
        
        total_time = time.time() - start_time
        print(f"Best parameters for {algorithm}: {best_params}")
        print(f"Best score: {best_score:.4f}")
        print(f"Total optimization time: {total_time:.1f} seconds")
        
        return {
            'algorithm': algorithm,
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'optimization_time': total_time
        }
    
    def optimize_all_algorithms(self):
        """Optimize all algorithms and save results"""
        optimization_results = {}
        total_start_time = time.time()
        
        for algorithm in self.param_grids.keys():
            print(f"\n{'='*50}")
            print(f"Optimizing {algorithm}")
            print(f"{'='*50}")
            
            result = self.optimize_algorithm(algorithm)
            if result:
                optimization_results[algorithm] = result
        
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"TOTAL OPTIMIZATION TIME: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"{'='*60}")
        
        # Save results
        self._save_optimization_results(optimization_results)
        
        return optimization_results
    
    def _save_optimization_results(self, results):
        """Save optimization results to files"""
        # Save best parameters
        best_params = {}
        for alg, result in results.items():
            best_params[alg] = result['best_params']
        
        import json
        with open(os.path.join(self.synthetic_root, "optimized_parameters.json"), "w") as f:
            json.dump(best_params, f, indent=2)
        
        # Save detailed results in the requested CSV format with three-factor scoring
        all_results = []
        for alg, result in results.items():
            for res in result['all_results']:
                params_str = "; ".join([f"{k}={v}" for k, v in res['params'].items()])
                all_results.append({
                    'algorithm_type': res['algorithm'],
                    'params': params_str,
                    's_epe': res['s_epe'],
                    's_mse': res['s_mse'],
                    'a_mse': res['a_mse'],
                    'combined_score': res['combined_score']
                })
        
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(self.synthetic_root, "parameter_optimization_results.csv"), index=False)
        
        print(f"\nOptimization results saved to:")
        print(f"- {os.path.join(self.synthetic_root, 'optimized_parameters.json')}")
        print(f"- {os.path.join(self.synthetic_root, 'parameter_optimization_results.csv')}")

def main():
    print("Starting optimized algorithm optimization...")
    main_start_time = time.time()
    
    # Initialize optimizer with three-factor scoring
    optimizer = AlgorithmOptimizer(
        synthetic_root="synthetic oct data 3",
        real_root=None,  # Set to real data path when available
        weight_s_epe=0.4,    # Synthetic EPE weight
        weight_s_mse=0.3,    # Synthetic MSE weight  
        weight_a_mse=0.3     # Actual MSE weight
    )
    
    # Optimize all algorithms
    results = optimizer.optimize_all_algorithms()
    
    main_total_time = time.time() - main_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    
    for alg, result in results.items():
        print(f"{alg}:")
        print(f"  Best Score: {result['best_score']:.4f}")
        print(f"  Best Params: {result['best_params']}")
        print(f"  Time: {result['optimization_time']:.1f}s")
        print()
    
    print(f"TOTAL RUNTIME: {main_total_time:.1f} seconds ({main_total_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()
