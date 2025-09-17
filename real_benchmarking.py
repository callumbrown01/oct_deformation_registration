import numpy as np
import cv2
import os
import pandas as pd
from scipy.io import loadmat
from skimage.metrics import mean_squared_error
from tracking_algorithms import TrackingAlgorithms

def load_oct_sequence(mat_path):
    mat = loadmat(mat_path)
    # Find the key with the actual data (skip __header__, __version__, __globals__)
    for k in mat:
        if not k.startswith('__'):
            arr = mat[k]
            break
    
    # Ensure shape is [H, W, N]
    if arr.ndim == 3:
        arr = arr
    elif arr.ndim == 2:
        arr = arr[..., np.newaxis]
    else:
        raise ValueError("Unexpected .mat file shape")
    
    # Clean the data first
    arr = np.nan_to_num(arr, nan=0.0, posinf=29.0, neginf=0.0)
    
    # Check the actual data range
    print(f"Raw data range: min={arr.min():.2f}, max={arr.max():.2f}")
    
    # More robust scaling - handle different data ranges
    if arr.max() <= 1.0:  # Data is already normalized
        arr = arr * 255.0
    elif arr.max() <= 29.0:  # Expected range 0-29
        arr = np.clip(arr, 0, 29)
        arr = (arr / 29.0 * 255.0)
    else:  # Unknown range - normalize to full range
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
    
    return arr.astype(np.float32)

def cumulative_align(frames, tracker, algorithm):
    # Clean up frames before casting to uint8
    def safe_uint8(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)
    
    ref = safe_uint8(frames[..., 0])
    cum_flow = np.zeros(ref.shape + (2,), dtype=np.float32)
    curr = ref.copy()
    
    # Accumulate flow fields
    for i in range(1, frames.shape[-1]):
        next_frame = safe_uint8(frames[..., i])
        flow = tracker.run(algorithm, curr, next_frame)
        cum_flow += flow
        curr = next_frame
    
    # Final mapping logic
    h, w = ref.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x + cum_flow[..., 0]
    map_y = y + cum_flow[..., 1]
    final_frame = safe_uint8(frames[..., -1])
    warped_final = cv2.remap(final_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return ref, warped_final

def benchmark_real_data(data_root):
    tracker = TrackingAlgorithms()
    algorithms = tracker.get_algorithm_names()
    results = []
    
    for sub in sorted(os.listdir(data_root)):
        sub_path = os.path.join(data_root, sub)
        if not os.path.isdir(sub_path):
            continue
        mat_files = [f for f in os.listdir(sub_path) if f.endswith('.mat')]
        if not mat_files:
            continue
        mat_path = os.path.join(sub_path, mat_files[0])
        frames = load_oct_sequence(mat_path)
        
        print(f"Processing {sub}: Frames shape: {frames.shape}")
        
        # MSE and pixel correlation between original and final frame (deformation impact)
        original_frame = frames[..., 0].astype(np.float32)
        final_frame = frames[..., -1].astype(np.float32)
        
        # Debug prints
        print(f"Original frame stats: min={original_frame.min():.2f}, max={original_frame.max():.2f}, mean={original_frame.mean():.2f}")
        print(f"Final frame stats: min={final_frame.min():.2f}, max={final_frame.max():.2f}, mean={final_frame.mean():.2f}")
        print(f"Original frame has NaN: {np.isnan(original_frame).any()}")
        print(f"Final frame has NaN: {np.isnan(final_frame).any()}")
        
        deformation_mse = np.mean((original_frame - final_frame) ** 2)
        
        # Safe pixel correlation calculation
        try:
            orig_flat = original_frame.flatten()
            final_flat = final_frame.flatten()
            
            # Remove any NaN or inf values
            valid_mask = np.isfinite(orig_flat) & np.isfinite(final_flat)
            if np.sum(valid_mask) > 1:
                deformation_pixel_corr = np.corrcoef(orig_flat[valid_mask], final_flat[valid_mask])[0, 1]
            else:
                deformation_pixel_corr = 0.0
        except Exception as e:
            print(f"Error calculating pixel correlation: {e}")
            deformation_pixel_corr = 0.0
            
        print(f"Deformation MSE: {deformation_mse:.2f}")
        print(f"Deformation pixel correlation: {deformation_pixel_corr:.4f}")
        
        for alg in algorithms:
            try:
                ref, warped_final = cumulative_align(frames, tracker, alg)
                
                # Ensure proper data types and no NaN values
                ref_clean = np.nan_to_num(ref.astype(np.float32))
                warped_clean = np.nan_to_num(warped_final.astype(np.float32))
                
                # MSE between original and algorithm warped result
                algorithm_mse = np.mean((original_frame - warped_clean) ** 2)
                
                # Safe pixel correlation calculation
                try:
                    orig_flat = original_frame.flatten()
                    warped_flat = warped_clean.flatten()
                    
                    valid_mask = np.isfinite(orig_flat) & np.isfinite(warped_flat)
                    if np.sum(valid_mask) > 1:
                        algorithm_pixel_corr = np.corrcoef(orig_flat[valid_mask], warped_flat[valid_mask])[0, 1]
                    else:
                        algorithm_pixel_corr = 0.0
                except Exception as e:
                    print(f"Error calculating algorithm pixel correlation: {e}")
                    algorithm_pixel_corr = 0.0
                
                print(f"{sub} | {alg}: Algorithm MSE = {algorithm_mse:.2f} | Pixel Corr = {algorithm_pixel_corr:.4f}")
                results.append({
                    'sample': sub,
                    'algorithm': alg,
                    'algorithm_mse': algorithm_mse,
                    'algorithm_pixel_corr': algorithm_pixel_corr,
                    'deformation_mse': deformation_mse,
                    'deformation_pixel_corr': deformation_pixel_corr
                })
            except Exception as e:
                print(f"{sub} | {alg}: Error - {e}")
                results.append({
                    'sample': sub,
                    'algorithm': alg,
                    'algorithm_mse': float('nan'),
                    'algorithm_pixel_corr': float('nan'),
                    'deformation_mse': deformation_mse,
                    'deformation_pixel_corr': deformation_pixel_corr
                })
    return results

def main():
    results = benchmark_real_data('./samples_mat')
    df = pd.DataFrame(results)
    output_path = 'real_data_performance.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()

