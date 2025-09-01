import tkinter as tk
import ttk, filedialog
import cv2
import numpy as np
from pathlib import Path
import itertools
import csv
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import time

# Parameter grids for each algorithm
param_grids = {
    'Farneback': {
        'pyr_scale': [0.5],
        'levels': [3, 4],
        'winsize': [15, 20, 25],
        'iterations': [3],
        'poly_n': [5, 7],
        'poly_sigma': [1.1, 1.5],
        'flags': [0]  # Add required flags parameter
    },
    'Lucas-Kanade': {
        'win_size': [(15,15), (21,21)],
        'max_level': [3, 4],
        'criteria': [(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)]
    },
    'DIS': {
        'setFinestScale': [1, 2],
        'setPatchSize': [8, 12],
        'setPatchStride': [4, 6],
        'setGradientDescentIterations': [12, 16],
        'setVariationalRefinementAlpha': [1.0],
        'setVariationalRefinementDelta': [0.5],
        'setVariationalRefinementGamma': [0.5],
        'setUseMeanNormalization': [True],
        'setUseSpatialPropagation': [True]
    },
}

def load_oct_sequence(mat_file):
    """Load OCT sequence from .mat file with 364x364xN layers structure"""
    try:
        print(f"Loading OCT data from {mat_file}")
        oct_data = loadmat(mat_file)
        
        # Find array with specific shape (364x364xN)
        oct_array = None
        array_shape = None
        for key, value in oct_data.items():
            if key.startswith('__'): 
                continue  # Skip metadata
            if isinstance(value, np.ndarray) and len(value.shape) == 3 and value.shape[:2] == (364, 364):
                oct_array = value
                array_shape = value.shape
                print(f"Found OCT array with shape {array_shape}")
                break
        
        if oct_array is None:
            print(f"Could not find 364x364xN array in {mat_file}")
            print("Available arrays:")
            for key, value in oct_data.items():
                if not key.startswith('__'):
                    print(f"  {key}: {type(value)}")
                    if isinstance(value, np.ndarray):
                        print(f"    shape: {value.shape}")
            return None
            
        # Convert to sequence of normalized images
        num_layers = oct_array.shape[2]
        print(f"Processing {num_layers} layers")
        sequence = []
        
        # Process each layer
        for i in range(num_layers):
            layer_data = oct_array[:,:,i]
            
            # Skip layer if it's all zeros or NaN
            if np.all(layer_data == 0) or np.all(np.isnan(layer_data)):
                print(f"Skipping empty/invalid layer {i}")
                continue
                
            # Normalize to 0-29 range
            normalized = np.clip(layer_data, 0, 29)
            # Convert to 0-255 for display
            normalized = ((normalized / 29.0) * 255).astype(np.uint8)
            sequence.append(normalized)
            
        if not sequence:
            print("No valid layers found in OCT data")
            return None
            
        print(f"Successfully loaded {len(sequence)} valid layers")
        return np.array(sequence)
            
    except Exception as e:
        print(f"Error loading {mat_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compute_warped_error(img1, img2, flow):
    """Compute error between original and warped images"""
    h, w = img1.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Apply flow to get sampling coordinates
    sample_x = (x + flow[..., 0]).clip(0, w-1)
    sample_y = (y + flow[..., 1]).clip(0, h-1)
    
    # Interpolate
    warped = cv2.remap(img2, sample_x, sample_y, 
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
    
    # Compute error
    diff = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX) - \
           cv2.normalize(warped, None, 0, 255, cv2.NORM_MINMAX)
    error_map = np.abs(diff)
    
    return warped, error_map

def evaluate_real_sequence(alg_name, params, images):
    """Evaluate optical flow algorithm on a real OCT sequence"""
    if len(images) < 2:
        return None
        
    # Process first pair of images
    img1 = images[0].astype(np.float32)
    img2 = images[-1].astype(np.float32)  # Use last frame for max deformation
    
    # Normalize images
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)
    
    try:
        start_time = time.time()
        
        # Compute optical flow
        if alg_name == 'Farneback':
            # Create full parameter dict with all required parameters
            farneback_params = {
                'prev': img1,
                'next': img2,
                'flow': None,
                'pyr_scale': params['pyr_scale'],
                'levels': params['levels'],
                'winsize': params['winsize'],
                'iterations': params['iterations'],
                'poly_n': params['poly_n'],
                'poly_sigma': params['poly_sigma'],
                'flags': params['flags']
            }
            flow = cv2.calcOpticalFlowFarneback(**farneback_params)
        elif alg_name == 'Lucas-Kanade':
            # Create grid of points
            h, w = img1.shape
            y, x = np.mgrid[0:h:10, 0:w:10].reshape(-1, 1, 2).astype(np.float32)
            flow, status, _ = cv2.calcOpticalFlowPyrLK(
                img1, img2, x, None, **params
            )
            # Convert sparse to dense flow
            flow = cv2.resize(flow, (w, h))
            
        elif alg_name == 'SimpleFlow':
            flow = cv2.optflow.calcOpticalFlowSF(
                img1, img2, **params
            )
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
            
        elif alg_name == 'TVL1':
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            for k, v in params.items():
                if k != 'lambda_':  # Handle special case
                    setattr(tvl1, k, v)
                else:
                    setattr(tvl1, 'lambda', v)
            flow = tvl1.calc(img1, img2, None)
            
        else:
            print(f"Algorithm {alg_name} not implemented")
            return None
            
        runtime = time.time() - start_time
        
        # Compute metrics
        warped, error_map = compute_warped_error(img1, img2, flow)
        sim_score = ssim(img1, warped, data_range=255)
        
        return {
            'mean_error': float(np.mean(error_map)),
            'max_error': float(np.max(error_map)),
            'error_std': float(np.std(error_map)),
            'ssim': float(sim_score),
            'runtime_s': float(runtime),
            'error_map': error_map,
            'warped': warped,
            'flow': flow,
            'orig_img': img1
        }
        
    except Exception as e:
        print(f"Error evaluating {alg_name}: {e}")
        return None

class OCTBenchmarkingPage(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Controls frame
        ctrl = ttk.Frame(self)
        ctrl.pack(fill='x', pady=10)
        
        # Dataset selector
        ttk.Button(ctrl, text="Select OCT Data Folder", 
                  command=self.load_dataset_root).pack(side='left', padx=5)
        self.dataset_label = ttk.Label(ctrl, text="No dataset selected")
        self.dataset_label.pack(side='left', padx=5)
        
        # Algorithm selector
        ttk.Label(ctrl, text="Algorithm:").pack(side='left', padx=(20,5))
        self.algorithms = ['Farneback', 'Lucas-Kanade', 'SimpleFlow', 
                         'Speckle Tracking', 'DIS', 'DeepFlow', 'PCAFlow', 
                         'TVL1', 'BlockMatching']
        self.selected_alg = tk.StringVar(value=self.algorithms[0])
        alg_combo = ttk.Combobox(ctrl, textvariable=self.selected_alg,
                                values=self.algorithms, state='readonly', width=16)
        alg_combo.pack(side='left')
        
        # Results table
        cols = ("Algorithm", "Parameters", "Mean Error", "Max Error", 
                "Error STD", "SSIM", "Runtime (ms)")
        self.table = ttk.Treeview(self, columns=cols, show="headings", height=20)
        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, width=120)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y')
        
        # Control buttons
        button_frame = ttk.Frame(ctrl)
        button_frame.pack(side='right', padx=5)
        
        ttk.Button(button_frame, text="Run Benchmark", 
                  command=self.run_benchmark).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Export Results", 
                  command=self.export_results).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Visualize Last", 
                  command=self.visualize_last_result).pack(side='left', padx=5)
        
        self.status = ttk.Label(ctrl, text="Ready")
        self.status.pack(side='right', padx=20)
        
        # Store results
        self.results = []
        self.last_result = None
    
    def load_dataset_root(self):
        """Open a folder dialog and set dataset root"""
        folder = filedialog.askdirectory(title="Select OCT Data Root")
        if folder:
            self.dataset_root = folder
            self.dataset_label.config(text=os.path.basename(folder))
            self.status.config(text=f"Dataset loaded: {os.path.basename(folder)}")
        else:
            self.status.config(text="No dataset selected")
    
    def run_benchmark(self):
        if not hasattr(self, 'dataset_root'):
            self.status.config(text="Select dataset first")
            return
        
        # Clear previous results
        for item in self.table.get_children():
            self.table.delete(item)
        
        alg = self.selected_alg.get()
        param_grid = param_grids[alg]
        
        # Ensure all parameter values are lists
        param_grid = {k: v if isinstance(v, list) else [v] for k, v in param_grid.items()}
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Find all subfolders
        subfolders = [f for f in Path(self.dataset_root).iterdir() if f.is_dir()]
        if not subfolders:
            self.status.config(text="No subfolders found")
            return
            
        # Get OCT .mat files from each subfolder
        mat_files = []
        for folder in subfolders:
            oct_files = list(folder.glob('*OCT*.mat'))
            if oct_files:
                # Take the first OCT file from each subfolder
                mat_files.append(oct_files[0])
                
        if not mat_files:
            self.status.config(text="No OCT .mat files found in subfolders")
            return
            
        self.status.config(text=f"Found {len(mat_files)} OCT files in {len(subfolders)} subfolders")
        
        # All parameter combinations
        param_combinations = list(itertools.product(*param_values))
        total = len(param_combinations) * len(mat_files)
        count = 0
        
        self.results = []
        for mat_file in mat_files:
            # Load OCT sequence
            images = load_oct_sequence(mat_file)
            if images is None:
                continue
                
            for combo in param_combinations:
                count += 1
                param_dict = dict(zip(param_names, combo))
                
                # Evaluate algorithm
                result = evaluate_real_sequence(alg, param_dict, images)
                if result is None:
                    continue
                
                # Store result
                self.last_result = {
                    'algorithm': alg,
                    'parameters': param_dict,
                    'sequence': str(mat_file),
                    **result
                }
                self.results.append(self.last_result)
                
                # Update table
                self.table.insert("", "end", values=(
                    alg,
                    str(param_dict),
                    f"{result['mean_error']:.3f}",
                    f"{result['max_error']:.3f}",
                    f"{result['error_std']:.3f}",
                    f"{result['ssim']:.3f}",
                    f"{result['runtime_s']*1000:.1f}"
                ))
                
                self.status.config(text=f"Progress: {count}/{total}")
                self.update_idletasks()
        
        self.status.config(text="Benchmark complete")
    
    def visualize_last_result(self):
        """Visualize the results of the last run"""
        if self.last_result is None:
            self.status.config(text="No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0,0].imshow(self.last_result['orig_img'], cmap='gray')
        axes[0,0].set_title('Original Image')
        
        # Warped image
        axes[0,1].imshow(self.last_result['warped'], cmap='gray')
        axes[0,1].set_title('Warped Final Image')
        
        # Error map
        im = axes[1,0].imshow(self.last_result['error_map'], cmap='hot')
        axes[1,0].set_title('Error Map')
        plt.colorbar(im, ax=axes[1,0])
        
        # Flow visualization
        flow = self.last_result.get('flow')
        if flow is not None:
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[...,0] = ang * 180 / np.pi / 2
            hsv[...,1] = 255
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            axes[1,1].imshow(rgb)
            axes[1,1].set_title('Flow Visualization')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self):
        if not self.results:
            self.status.config(text="No results to export")
            return
        
        file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file:
            return
        
        # Define which fields to export (excluding image data)
        fieldnames = [
            'algorithm', 'parameters', 'sequence',
            'mean_error', 'max_error', 'error_std',
            'ssim', 'runtime_s'
        ]
        
        with open(file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.results:
                # Create a clean copy with only the fields we want to export
                row_copy = {
                    'algorithm': row['algorithm'],
                    'parameters': str(row['parameters']),
                    'sequence': row['sequence'],
                    'mean_error': row['mean_error'],
                    'max_error': row['max_error'],
                    'error_std': row['error_std'],
                    'ssim': row['ssim'],
                    'runtime_s': row['runtime_s']
                }
                writer.writerow(row_copy)
        
        self.status.config(text=f"Results exported to {os.path.basename(file)}")

if __name__ == '__main__':
    # Create main window
    root = tk.Tk()
    root.title("OCT Registration Benchmarking")
    root.geometry("1200x800")

    # Create and pack the benchmarking page
    page = OCTBenchmarkingPage(root)
    page.pack(fill='both', expand=True, padx=10, pady=10)

    # Start the main loop
    root.mainloop()
