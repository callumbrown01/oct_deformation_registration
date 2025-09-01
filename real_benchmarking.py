import numpy as np
import cv2
import os
import numpy as np
import cv2
import os
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import matplotlib.pyplot as plt

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
        
def cumulative_align(frames, alg, params):
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
        flow = run_flow_algorithm(alg, params, curr, next_frame)
        cum_flow += flow
        curr = next_frame
    # Final mapping logic (as in final.ipynb)
    h, w = ref.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x + cum_flow[..., 0]
    map_y = y + cum_flow[..., 1]
    final_frame = safe_uint8(frames[..., -1])
    warped_final = cv2.remap(final_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return ref, warped_final

def benchmark_cumulative_on_folder(data_root, show_images=False):
    # Collect metrics for each algorithm across all folders
    alg_results = {alg: [] for alg, _ in algorithms}
    for sub in sorted(os.listdir(data_root)):
        sub_path = os.path.join(data_root, sub)
        if not os.path.isdir(sub_path):
            continue
        mat_files = [f for f in os.listdir(sub_path) if f.endswith('.mat')]
        if not mat_files:
            continue
        mat_path = os.path.join(sub_path, mat_files[0])
        frames = load_oct_sequence(mat_path)
        for alg, params in algorithms:
            try:
                ref, warped_final = cumulative_align(frames, alg, params)
                ref_f32 = ref.astype(np.float32)
                warped_f32 = warped_final.astype(np.float32)
                # Only compare valid region: where ref is nonzero (in bounds)
                mask = ref > 0
                if np.any(mask):
                    diff = np.abs(ref_f32 - warped_f32)
                    mean_diff = np.mean(diff[mask])
                    mse = mean_squared_error(ref[mask], warped_final[mask])
                    psnr = peak_signal_noise_ratio(ref[mask], warped_final[mask], data_range=255)
                    ssim = structural_similarity(ref, warped_final, data_range=255, mask=mask)
                else:
                    mean_diff = mse = psnr = ssim = float('nan')
                print(f"{sub} | {alg}: Mean abs diff = {mean_diff:.2f}, MSE = {mse:.2f}, PSNR = {psnr:.2f}, SSIM = {ssim:.3f}")
                alg_results[alg].append({
                    'mean_abs_diff': mean_diff,
                    'mse': mse,
                    'psnr': psnr,
                    'ssim': ssim
                })
                # Remove visualisation
                # if show_images:
                #     plt.figure(figsize=(8, 4))
                #     plt.suptitle(f"{sub} | {alg}", fontsize=14)
                #     plt.subplot(1, 2, 1)
                #     plt.imshow(ref, cmap='gray')
                #     plt.title('Original')
                #     plt.axis('off')
                #     plt.subplot(1, 2, 2)
                #     plt.imshow(warped_final, cmap='gray')
                #     plt.title('Final Mapped')
                #     plt.axis('off')
                #     plt.tight_layout()
                #     plt.show()
            except Exception as e:
                print(f"{sub} | {alg}: Error - {e}")
    # Compute average metrics for each algorithm
    avg_results = []
    for alg in alg_results:
        vals = alg_results[alg]
        if vals:
            mean_abs_diff = np.mean([v['mean_abs_diff'] for v in vals])
            mse = np.mean([v['mse'] for v in vals])
            psnr = np.mean([v['psnr'] for v in vals])
            ssim = np.mean([v['ssim'] for v in vals])
            print(f"{alg}: Avg MeanAbsDiff={mean_abs_diff:.2f}, Avg MSE={mse:.2f}, Avg PSNR={psnr:.2f}, Avg SSIM={ssim:.3f}")
            avg_results.append({
                'algorithm': alg,
                'average_mean_abs_diff': mean_abs_diff,
                'average_mse': mse,
                'average_psnr': psnr,
                'average_ssim': ssim
            })
        else:
            print(f"{alg}: No valid results.")
    return avg_results

# Example: Replace with your actual image loading logic
def load_image_pair(data_dir, idx=0):
    # Load two consecutive frames for benchmarking
    img_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
    img1 = cv2.imread(os.path.join(data_dir, img_files[idx]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(data_dir, img_files[idx+1]), cv2.IMREAD_GRAYSCALE)
    return img1, img2

# Algorithm parameter sets
algorithms = [
    ('TVL1', {'tau': 0.25, 'lambda_': 0.15, 'theta': 0.3, 'nscales': 3, 'warps': 3, 'epsilon': 0.02}),
    ('DIS', {'finest_scale': 0, 'patch_size': 4, 'patch_stride': 2, 'grad_descent_iter': 8, 'var_refine_iter': 5, 'var_refine_alpha': 40.0, 'var_refine_delta': 2.0, 'var_refine_gamma': 5.0, 'use_mean_normalization': True, 'use_spatial_propagation': False}),
    ('Farneback', {'pyr_scale': 0.3, 'levels': 4, 'winsize': 27, 'iterations': 1, 'poly_n': 5, 'poly_sigma': 1.1}),
    ('PCAFlow', {}),
    ('Lucas-Kanade', {'win_size': (27, 27), 'max_level': 2, 'criteria_eps': 0.05, 'criteria_count': 10}),
    ('DeepFlow', {})
]

def run_flow_algorithm(alg_name, params, img1, img2):
    if alg_name == 'TVL1':
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(
            tau=params['tau'],
            lambda_=params['lambda_'],
            theta=params['theta'],
            nscales=params['nscales'],
            warps=params['warps'],
            epsilon=params['epsilon']
        )
        flow = tvl1.calc(img1.astype(np.float32)/255.0, img2.astype(np.float32)/255.0, None)
    elif alg_name == 'DIS':
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(params['finest_scale'])
        dis.setPatchSize(params['patch_size'])
        dis.setPatchStride(params['patch_stride'])
        dis.setGradientDescentIterations(params['grad_descent_iter'])
        dis.setVariationalRefinementIterations(params['var_refine_iter'])
        dis.setVariationalRefinementAlpha(params['var_refine_alpha'])
        dis.setVariationalRefinementDelta(params['var_refine_delta'])
        dis.setVariationalRefinementGamma(params['var_refine_gamma'])
        dis.setUseMeanNormalization(params['use_mean_normalization'])
        dis.setUseSpatialPropagation(params['use_spatial_propagation'])
        flow = dis.calc(img1, img2, None)
    elif alg_name == 'Farneback':
        flow = cv2.calcOpticalFlowFarneback(
            img1, img2, None,
            pyr_scale=params['pyr_scale'],
            levels=params['levels'],
            winsize=params['winsize'],
            iterations=params['iterations'],
            poly_n=params['poly_n'],
            poly_sigma=params['poly_sigma'],
            flags=0
        )
    elif alg_name == 'PCAFlow':
        pca = cv2.optflow.createOptFlow_PCAFlow()
        flow = pca.calc(img1, img2, None)
    elif alg_name == 'Lucas-Kanade':
        h, w = img1.shape
        # Generate grid points for tracking
        y, x = np.mgrid[0:h:10, 0:w:10]
        pts = np.stack((x, y), axis=-1).reshape(-1, 1, 2).astype(np.float32)
        nextPts, status, err = cv2.calcOpticalFlowPyrLK(
            img1, img2, pts, None,
            winSize=params['win_size'],
            maxLevel=params['max_level'],
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, params['criteria_count'], params['criteria_eps'])
        )
        # Compute flow vectors for valid points
        flow_sparse = np.zeros_like(pts)
        valid = status.flatten() == 1
        flow_sparse[valid] = nextPts[valid] - pts[valid]
        # Interpolate sparse flow to dense flow
        flow = np.zeros((h, w, 2), dtype=np.float32)
        for i, pt in enumerate(pts.reshape(-1, 2)):
            x0, y0 = int(pt[0]), int(pt[1])
            if 0 <= x0 < w and 0 <= y0 < h:
                flow[y0, x0] = flow_sparse[i, 0]
        # Fill missing values by interpolation (optional, here just use cv2.resize)
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    elif alg_name == 'DeepFlow':
        df = cv2.optflow.createOptFlow_DeepFlow()
        flow = df.calc(img1, img2, None)
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")
    return flow

def warp_image(img, flow):
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = x + flow[..., 0]
    map_y = y + flow[..., 1]
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped

def benchmark_on_data(data_dir):
    img1, img2 = load_image_pair(data_dir)
    results = []
    for alg, params in algorithms:
        try:
            flow = run_flow_algorithm(alg, params, img1, img2)
            warped = warp_image(img2, flow)
            # Performance by subtraction
            diff = np.abs(img1.astype(np.float32) - warped.astype(np.float32))
            mean_diff = np.mean(diff)
            print(f"{alg}: Mean absolute difference = {mean_diff:.2f}")
            results.append({'algorithm': alg, 'mean_abs_diff': mean_diff})
        except Exception as e:
            print(f"{alg}: Error - {e}")
    return results

# Example usage:
# Run cumulative benchmarking across all subfolders in the dataset and show images
results = benchmark_cumulative_on_folder('./samples_mat', show_images=True)

print(results)

