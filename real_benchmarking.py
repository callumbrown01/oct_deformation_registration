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
    # Scale values from 0-29 to 0-255
    arr = np.clip(arr, 0, 29)
    arr = (arr / 29.0 * 255).astype(np.float32)
    return arr

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
        
        # MSE between original and final frame (deformation impact)
        original_frame = frames[..., 0].astype(np.float32)
        final_frame = frames[..., -1].astype(np.float32)
        
        print(f"Original frame stats: min={original_frame.min():.2f}, max={original_frame.max():.2f}, mean={original_frame.mean():.2f}")
        print(f"Final frame stats: min={final_frame.min():.2f}, max={final_frame.max():.2f}, mean={final_frame.mean():.2f}")
        
        deformation_mse = np.mean((original_frame - final_frame) ** 2)
        print(f"Deformation MSE: {deformation_mse:.2f}")
        
        for alg in algorithms:
            try:
                ref, warped_final = cumulative_align(frames, tracker, alg)
                
                # MSE between original and algorithm warped result
                algorithm_mse = np.mean((original_frame - warped_final.astype(np.float32)) ** 2)
                
                print(f"{sub} | {alg}: Algorithm MSE = {algorithm_mse:.2f} | Deformation MSE = {deformation_mse:.2f}")
                results.append({
                    'sample': sub,
                    'algorithm': alg,
                    'algorithm_mse': algorithm_mse,
                    'deformation_mse': deformation_mse
                })
            except Exception as e:
                print(f"{sub} | {alg}: Error - {e}")
                results.append({
                    'sample': sub,
                    'algorithm': alg,
                    'algorithm_mse': float('nan'),
                    'deformation_mse': deformation_mse
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

