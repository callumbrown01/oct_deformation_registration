"""
Data Loading Module

Handles loading and preprocessing of OCT, stress, and attenuation data from .mat files.
"""

import os
import glob
import numpy as np
import scipy.io as sio
import cv2
from typing import Dict, List, Tuple, Optional


class OCTDataLoader:
    """
    Loads and manages OCT image sequences with co-registered stress and attenuation data.
    
    Attributes:
        sequence_root (str): Path to the directory containing .mat files
        images (dict): RGB visualizations of OCT, stress, and attenuation
        scalar_images (dict): Raw scalar data for stress and attenuation
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.sequence_root: Optional[str] = None
        self.images: Dict[str, List[np.ndarray]] = {"oct": [], "stress": [], "att": []}
        self.scalar_images: Dict[str, List[np.ndarray]] = {"stress": [], "att": []}
    
    def load_sequence(self, folder_path: str) -> bool:
        """
        Load a complete OCT sequence from the specified folder.
        
        Args:
            folder_path: Path to folder containing OCT*.mat, Stress*.mat, and Att*.mat files
            
        Returns:
            bool: True if loading successful, False otherwise
            
        Raises:
            FileNotFoundError: If required .mat files are not found
            ValueError: If data dimensions are inconsistent
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Directory not found: {folder_path}")
        
        self.sequence_root = folder_path
        self.images = {"oct": [], "stress": [], "att": []}
        self.scalar_images = {"stress": [], "att": []}
        
        try:
            # Load OCT images
            oct_files = sorted(glob.glob(os.path.join(folder_path, "*OCT*.mat")))
            if not oct_files:
                raise FileNotFoundError("No OCT .mat files found")
            
            for f in oct_files:
                data = sio.loadmat(f)
                # Try different possible field names
                img = None
                for key in ['img', 'OCT', 'oct', 'image']:
                    if key in data:
                        img = data[key]
                        break
                
                if img is None:
                    # Get first non-metadata field
                    candidates = [k for k in data.keys() if not k.startswith('__')]
                    if candidates:
                        img = data[candidates[0]]
                
                if img is not None:
                    img = np.squeeze(img)
                    if img.ndim == 2:
                        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    self.images["oct"].append(img)
            
            # Load stress data
            stress_files = sorted(glob.glob(os.path.join(folder_path, "*Stress*.mat")))
            for f in stress_files:
                data = sio.loadmat(f)
                stress_array = self._extract_first_array(data)
                
                if stress_array is None:
                    continue
                
                stress_array = np.squeeze(stress_array)
                if stress_array.ndim == 3:
                    num_frames = stress_array.shape[2]
                else:
                    stress_array = stress_array[:, :, np.newaxis]
                    num_frames = 1
                
                for i in range(num_frames):
                    stress_kpa = stress_array[:, :, i] if stress_array.ndim == 3 else stress_array
                    stress_kpa = np.abs(stress_kpa)  # Take absolute values
                    stress_kpa = np.clip(stress_kpa, 0, 100)
                    self.scalar_images["stress"].append(stress_kpa)
                    
                    # Convert to RGB visualization using log scaling
                    rgb = self._apply_colormap_log(stress_kpa, 'jet', 1, 100)
                    self.images["stress"].append(rgb)
            
            # Load attenuation data
            att_files = sorted(glob.glob(os.path.join(folder_path, "*Att*.mat")))
            for f in att_files:
                data = sio.loadmat(f)
                att_array = self._extract_first_array(data)
                
                if att_array is None:
                    continue
                
                att_array = np.squeeze(att_array)
                if att_array.ndim == 3:
                    num_frames = att_array.shape[2]
                else:
                    att_array = att_array[:, :, np.newaxis]
                    num_frames = 1
                
                for i in range(num_frames):
                    slice_data = att_array[:, :, i] if att_array.ndim == 3 else att_array
                    slice_data = np.clip(slice_data, 0, 10)
                    self.scalar_images["att"].append(slice_data)
                    
                    # Convert to RGB visualization
                    rgb = self._apply_colormap(slice_data, 'viridis', 0, 10)
                    self.images["att"].append(rgb)
            
            return len(self.images["oct"]) > 0
            
        except Exception as e:
            print(f"Error loading sequence: {e}")
            return False
    
    def _extract_first_array(self, mat_dict: dict) -> Optional[np.ndarray]:
        """
        Extract the first non-metadata array from a .mat file dictionary.
        
        Args:
            mat_dict: Dictionary loaded from .mat file
            
        Returns:
            First numpy array found, or None if no valid array exists
        """
        for key in mat_dict.keys():
            if not key.startswith('__'):
                val = mat_dict[key]
                if isinstance(val, np.ndarray):
                    return val
        return None
    
    def _apply_colormap(self, data: np.ndarray, cmap_name: str, 
                       vmin: float, vmax: float) -> np.ndarray:
        """
        Apply a matplotlib colormap to scalar data.
        
        Args:
            data: 2D scalar array
            cmap_name: Name of matplotlib colormap
            vmin: Minimum value for normalization
            vmax: Maximum value for normalization
            
        Returns:
            RGB image as uint8 array
        """
        import matplotlib
        valid_data = np.nan_to_num(data, nan=vmin)
        norm = (np.clip(valid_data, vmin, vmax) - vmin) / (vmax - vmin)
        cmap = matplotlib.colormaps[cmap_name]
        rgb = cmap(norm)[..., :3]
        return (rgb * 255).astype(np.uint8)
    
    def _apply_colormap_log(self, data: np.ndarray, cmap_name: str,
                            vmin: float, vmax: float) -> np.ndarray:
        """
        Apply a matplotlib colormap with logarithmic scaling.
        
        Args:
            data: 2D scalar array (in original units)
            cmap_name: Name of matplotlib colormap
            vmin: Minimum value in original units
            vmax: Maximum value in original units
            
        Returns:
            RGB image as uint8 array
        """
        import matplotlib
        # Apply log transform
        data_log = np.log10(data + 1e-6)
        vmin_log = np.log10(vmin + 1e-6)
        vmax_log = np.log10(vmax + 1e-6)
        
        # Normalize
        valid_data = np.nan_to_num(data_log, nan=vmin_log)
        norm = (np.clip(valid_data, vmin_log, vmax_log) - vmin_log) / (vmax_log - vmin_log)
        
        # Apply colormap
        cmap = matplotlib.colormaps[cmap_name]
        rgb = cmap(norm)[..., :3]
        return (rgb * 255).astype(np.uint8)
    
    def get_frame_count(self) -> int:
        """Get the number of frames in the loaded sequence."""
        return len(self.images.get("oct", []))
    
    def get_frame(self, modality: str, index: int, 
                  as_scalar: bool = False) -> Optional[np.ndarray]:
        """
        Get a specific frame from the sequence.
        
        Args:
            modality: One of "oct", "stress", or "att"
            index: Frame index
            as_scalar: If True, return scalar data (for stress/att only)
            
        Returns:
            Frame data as numpy array, or None if not available
        """
        if as_scalar and modality in self.scalar_images:
            data = self.scalar_images[modality]
        else:
            data = self.images.get(modality, [])
        
        if 0 <= index < len(data):
            return data[index]
        return None
