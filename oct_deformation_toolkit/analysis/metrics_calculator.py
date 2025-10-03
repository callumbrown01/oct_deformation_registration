"""
Metrics Calculation Module

Computes quantitative metrics for deformation analysis.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, List
from ..core.image_processing import ImageProcessor


class MetricsCalculator:
    """
    Calculates quantitative metrics for OCT deformation analysis.
    
    Includes:
    - Attenuation-stress response ratios
    - Mean values over regions of interest
    - Trajectory computations
    """
    
    def __init__(self, eps: float = 1e-6):
        """
        Initialize the metrics calculator.
        
        Args:
            eps: Small epsilon value to avoid division by zero
        """
        self.eps = eps
        self.processor = ImageProcessor()
    
    def compute_response_ratio(self,
                               att_ref: np.ndarray,
                               att_cur: np.ndarray,
                               stress_ref: np.ndarray,
                               stress_cur: np.ndarray,
                               valid_mask: Optional[np.ndarray] = None,
                               use_block_averaging: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attenuation-stress response ratio: ΔAtt/ΔStress.
        
        Args:
            att_ref: Reference attenuation map
            att_cur: Current attenuation map
            stress_ref: Reference stress map
            stress_cur: Current stress map
            valid_mask: Optional mask of valid pixels
            use_block_averaging: Whether to apply block averaging to attenuation
            
        Returns:
            Tuple of (response_ratio, validity_mask)
        """
        # Apply block averaging to attenuation for noise reduction
        if use_block_averaging:
            att_ref = self.processor.block_average(att_ref)
            att_cur = self.processor.block_average(att_cur)
        
        # Calculate changes
        d_att = (att_cur - att_ref).astype(np.float32)
        d_stress = (stress_cur - stress_ref).astype(np.float32)
        
        # Compute response ratio
        response = np.where(d_stress > self.eps, d_att / d_stress, np.nan)
        
        # Create validity mask
        valid = np.isfinite(response)
        if valid_mask is not None:
            valid &= valid_mask
        
        return response, valid
    
    def compute_region_means(self,
                            stress_map: np.ndarray,
                            att_map: np.ndarray,
                            extent_mask: Optional[np.ndarray] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute mean stress and attenuation over a region.
        
        Args:
            stress_map: Stress map
            att_map: Attenuation map
            extent_mask: Optional mask defining the region
            
        Returns:
            Tuple of (mean_stress, mean_attenuation) or (None, None) if invalid
        """
        if extent_mask is None:
            extent_mask = np.ones_like(stress_map, dtype=bool)
        
        if not np.any(extent_mask):
            return None, None
        
        # Calculate means only over valid pixels
        stress_valid = np.where(extent_mask, stress_map, np.nan)
        att_valid = np.where(extent_mask, att_map, np.nan)
        
        if not np.any(np.isfinite(stress_valid)) or not np.any(np.isfinite(att_valid)):
            return None, None
        
        mean_stress = float(np.nanmean(stress_valid))
        mean_att = float(np.nanmean(att_valid))
        
        return mean_stress, mean_att
    
    def compute_box_means(self,
                         stress_map: np.ndarray,
                         att_map: np.ndarray,
                         x_min: int,
                         x_max: int,
                         y_min: int,
                         y_max: int,
                         extent_mask: Optional[np.ndarray] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute mean values within a rectangular region.
        
        Args:
            stress_map: Stress map
            att_map: Attenuation map
            x_min, x_max: Horizontal bounds of box
            y_min, y_max: Vertical bounds of box
            extent_mask: Optional additional mask to apply
            
        Returns:
            Tuple of (mean_stress, mean_attenuation)
        """
        H, W = stress_map.shape[:2]
        
        # Clip coordinates to image bounds
        x_min = np.clip(x_min, 0, W - 1)
        x_max = np.clip(x_max, 0, W - 1)
        y_min = np.clip(y_min, 0, H - 1)
        y_max = np.clip(y_max, 0, H - 1)
        
        # Create box mask
        box_mask = np.zeros((H, W), dtype=bool)
        box_mask[y_min:y_max+1, x_min:x_max+1] = True
        
        # Combine with extent mask if provided
        if extent_mask is not None:
            box_mask &= extent_mask
        
        return self.compute_region_means(stress_map, att_map, box_mask)
    
    def compute_percentile_range(self,
                                data: np.ndarray,
                                lower_percentile: float = 2,
                                upper_percentile: float = 98) -> Tuple[float, float]:
        """
        Compute percentile-based range for robust visualization scaling.
        
        Args:
            data: Input array
            lower_percentile: Lower percentile (0-100)
            upper_percentile: Upper percentile (0-100)
            
        Returns:
            Tuple of (min_value, max_value)
        """
        valid_data = data[np.isfinite(data)]
        
        if len(valid_data) == 0:
            return 0.0, 1.0
        
        vmin = float(np.percentile(valid_data, lower_percentile))
        vmax = float(np.percentile(valid_data, upper_percentile))
        
        # Ensure vmax > vmin
        if vmax <= vmin:
            vmax = vmin + 1.0
        
        return vmin, vmax
    
    def filter_fluctuations(self,
                           response_current: np.ndarray,
                           response_previous: np.ndarray,
                           threshold: float = 2.0) -> np.ndarray:
        """
        Create mask filtering out aggressive response fluctuations.
        
        Args:
            response_current: Current response ratio map
            response_previous: Previous response ratio map
            threshold: Maximum allowed change between frames
            
        Returns:
            Boolean mask (True = valid, False = fluctuating)
        """
        if response_previous is None:
            return np.ones_like(response_current, dtype=bool)
        
        change = response_current - response_previous
        mask = np.abs(change) <= threshold
        
        return mask
