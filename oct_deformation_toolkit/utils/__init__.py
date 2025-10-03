"""
Export Utilities for OCT Deformation Toolkit

This module provides utilities for exporting co-registered OCT images,
stress maps, attenuation maps, and analysis results.

Author: Callum Brown
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
import json


class DataExporter:
    """
    Handles exporting of co-registered images and analysis results.
    
    This class provides methods to save:
    - Co-registered OCT, stress, and attenuation images
    - Response ratio maps
    - Trajectory data
    - Analysis metrics in JSON format
    
    Attributes:
        output_dir (str): Base directory for exports
    """
    
    def __init__(self, output_dir: str = './exports'):
        """
        Initialize the data exporter.
        
        Args:
            output_dir: Directory where exported files will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_coregistered_sequence(
        self,
        oct_frames: List[np.ndarray],
        stress_frames: List[np.ndarray],
        attenuation_frames: List[np.ndarray],
        sequence_name: str = 'sequence'
    ) -> Dict[str, str]:
        """
        Export a complete co-registered sequence.
        
        Args:
            oct_frames: List of OCT images
            stress_frames: List of stress maps
            attenuation_frames: List of attenuation maps
            sequence_name: Name prefix for the exported files
            
        Returns:
            Dictionary mapping modality to output directory path
        """
        # Create subdirectories
        oct_dir = os.path.join(self.output_dir, sequence_name, 'oct')
        stress_dir = os.path.join(self.output_dir, sequence_name, 'stress')
        atten_dir = os.path.join(self.output_dir, sequence_name, 'attenuation')
        
        for directory in [oct_dir, stress_dir, atten_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Export each modality
        for i, (oct, stress, atten) in enumerate(zip(
            oct_frames, stress_frames, attenuation_frames
        )):
            # OCT (grayscale)
            oct_path = os.path.join(oct_dir, f'frame_{i:04d}.png')
            cv2.imwrite(oct_path, oct)
            
            # Stress (save as 16-bit for precision)
            stress_path = os.path.join(stress_dir, f'frame_{i:04d}.tiff')
            # Normalize to 16-bit range
            stress_norm = self._normalize_to_uint16(stress)
            cv2.imwrite(stress_path, stress_norm)
            
            # Attenuation (save as 16-bit for precision)
            atten_path = os.path.join(atten_dir, f'frame_{i:04d}.tiff')
            atten_norm = self._normalize_to_uint16(atten)
            cv2.imwrite(atten_path, atten_norm)
        
        return {
            'oct': oct_dir,
            'stress': stress_dir,
            'attenuation': atten_dir
        }
    
    def export_response_map(
        self,
        response_map: np.ndarray,
        frame_index: int,
        sequence_name: str = 'sequence'
    ) -> str:
        """
        Export a response ratio map.
        
        Args:
            response_map: Response ratio data (ΔAttenuation/ΔStress)
            frame_index: Frame number
            sequence_name: Name prefix for the file
            
        Returns:
            Path to the saved file
        """
        response_dir = os.path.join(self.output_dir, sequence_name, 'response')
        os.makedirs(response_dir, exist_ok=True)
        
        # Save as 32-bit float TIFF to preserve NaN values
        response_path = os.path.join(response_dir, f'frame_{frame_index:04d}.tiff')
        cv2.imwrite(response_path, response_map.astype(np.float32))
        
        return response_path
    
    def export_trajectories(
        self,
        trajectory_data: Dict[str, np.ndarray],
        sequence_name: str = 'sequence'
    ) -> str:
        """
        Export trajectory data as numpy arrays.
        
        Args:
            trajectory_data: Dictionary with keys like 'stress', 'attenuation'
                           and values as Nx2 arrays of (stress, attenuation) points
            sequence_name: Name prefix for the file
            
        Returns:
            Path to the saved .npz file
        """
        traj_dir = os.path.join(self.output_dir, sequence_name)
        os.makedirs(traj_dir, exist_ok=True)
        
        traj_path = os.path.join(traj_dir, 'trajectories.npz')
        np.savez(traj_path, **trajectory_data)
        
        return traj_path
    
    def export_metrics_json(
        self,
        metrics: Dict,
        sequence_name: str = 'sequence'
    ) -> str:
        """
        Export analysis metrics as JSON.
        
        Args:
            metrics: Dictionary containing analysis metrics
                    (response ratios, means, etc.)
            sequence_name: Name prefix for the file
            
        Returns:
            Path to the saved JSON file
        """
        metrics_dir = os.path.join(self.output_dir, sequence_name)
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_path = os.path.join(metrics_dir, 'metrics.json')
        
        # Convert numpy types to Python types for JSON serialization
        serializable_metrics = self._make_json_serializable(metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        return metrics_path
    
    def export_overlay_image(
        self,
        base_image: np.ndarray,
        overlay: np.ndarray,
        colormap_name: str,
        frame_index: int,
        sequence_name: str = 'sequence',
        alpha: float = 0.5
    ) -> str:
        """
        Export an image with overlay applied.
        
        Args:
            base_image: Base OCT image (grayscale)
            overlay: Overlay data (stress or attenuation)
            colormap_name: Name of colormap ('jet', 'viridis', 'plasma')
            frame_index: Frame number
            sequence_name: Name prefix
            alpha: Overlay transparency (0-1)
            
        Returns:
            Path to saved overlay image
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        overlay_dir = os.path.join(self.output_dir, sequence_name, 'overlays')
        os.makedirs(overlay_dir, exist_ok=True)
        
        # Apply colormap
        cmap = cm.get_cmap(colormap_name)
        
        # Normalize overlay
        valid_mask = np.isfinite(overlay)
        if np.any(valid_mask):
            vmin, vmax = np.nanmin(overlay), np.nanmax(overlay)
            overlay_norm = (overlay - vmin) / (vmax - vmin + 1e-10)
            overlay_colored = cmap(overlay_norm)
            overlay_colored = (overlay_colored[:, :, :3] * 255).astype(np.uint8)
        else:
            overlay_colored = np.zeros((*overlay.shape, 3), dtype=np.uint8)
        
        # Convert base to RGB
        if len(base_image.shape) == 2:
            base_rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = base_image
        
        # Blend
        blended = cv2.addWeighted(base_rgb, 1-alpha, overlay_colored, alpha, 0)
        
        # Apply mask to show only valid overlay regions
        blended[~valid_mask] = base_rgb[~valid_mask]
        
        # Save
        overlay_path = os.path.join(
            overlay_dir, 
            f'{colormap_name}_frame_{frame_index:04d}.png'
        )
        cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        return overlay_path
    
    @staticmethod
    def _normalize_to_uint16(data: np.ndarray) -> np.ndarray:
        """
        Normalize floating-point data to 16-bit unsigned integer range.
        
        Args:
            data: Input data array
            
        Returns:
            Normalized uint16 array
        """
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            return np.zeros(data.shape, dtype=np.uint16)
        
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        uint16_data = (normalized * 65535).astype(np.uint16)
        uint16_data[~valid_mask] = 0
        
        return uint16_data
    
    @staticmethod
    def _make_json_serializable(obj):
        """
        Convert numpy types to Python native types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: DataExporter._make_json_serializable(val) 
                   for key, val in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [DataExporter._make_json_serializable(item) for item in obj]
        else:
            return obj
