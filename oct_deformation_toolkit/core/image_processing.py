"""
Image Processing Module

Utilities for image manipulation, masking, and preprocessing.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Tuple, Optional


class ImageProcessor:
    """
    Handles image processing operations for OCT analysis.
    
    Includes methods for:
    - Block averaging for noise reduction
    - Tissue/adipose masking
    - Image format conversions
    - Smoothing and filtering
    """
    
    @staticmethod
    def block_average(data: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Apply block averaging to reduce speckle noise while preserving NaN values.
        
        Args:
            data: Input 2D array
            size: Block size for averaging
            
        Returns:
            Block-averaged array with NaN preserved
        """
        valid_mask = np.isfinite(data)
        data_filled = np.where(valid_mask, data, 0)
        
        # Use uniform filter for efficient block averaging
        count = ndimage.uniform_filter(valid_mask.astype(float), size=size, mode='nearest')
        summed = ndimage.uniform_filter(data_filled, size=size, mode='nearest')
        
        with np.errstate(invalid='ignore'):
            avg = summed / count
        
        avg[count < 1e-3] = np.nan
        return avg
    
    @staticmethod
    def create_adipose_mask(oct_image: np.ndarray, 
                           threshold_high: float = 5.5,
                           min_area: int = 24) -> Optional[np.ndarray]:
        """
        Create a binary mask identifying adipose tissue regions.
        
        Adipose tissue appears as bright regions in OCT images. This method
        identifies such regions using intensity thresholding and morphological
        operations.
        
        Args:
            oct_image: Grayscale OCT image
            threshold_high: Upper threshold multiplier for Otsu's method
            min_area: Minimum area (pixels) to keep in mask
            
        Returns:
            Binary mask (True = adipose tissue) or None if no adipose detected
        """
        if oct_image is None or oct_image.size == 0:
            return None
        
        # Ensure grayscale
        if oct_image.ndim == 3:
            gray = cv2.cvtColor(oct_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = oct_image
        
        # Otsu thresholding
        try:
            thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except:
            return None
        
        # High-intensity threshold for adipose
        upper_threshold = int(thresh_val * threshold_high)
        _, adipose_binary = cv2.threshold(gray, upper_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        adipose_binary = cv2.morphologyEx(adipose_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        adipose_binary = cv2.morphologyEx(adipose_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(adipose_binary, connectivity=8)
        
        cleaned_mask = np.zeros_like(adipose_binary, dtype=bool)
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned_mask[labels == i] = True
        
        return cleaned_mask if np.any(cleaned_mask) else None
    
    @staticmethod
    def to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert image to RGB format.
        
        Args:
            image: Input image (grayscale or already RGB)
            
        Returns:
            RGB image as uint8 array
        """
        if image.ndim == 2:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return image.astype(np.uint8)
    
    @staticmethod
    def smooth_with_nan_handling(data: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian smoothing while properly handling NaN values.
        
        Args:
            data: Input 2D array
            sigma: Gaussian kernel standard deviation
            
        Returns:
            Smoothed array with NaN preserved where appropriate
        """
        if sigma <= 0:
            return data
        
        # Create mask for valid (finite) values
        valid_mask = np.isfinite(data)
        
        if not np.any(valid_mask):
            return data  # Return original if all NaN
        
        # Temporarily fill NaN with 0 for filtering
        temp_data = data.copy()
        temp_data[~valid_mask] = 0
        
        # Apply Gaussian filter to both data and weights
        smoothed_values = ndimage.gaussian_filter(temp_data.astype(float), sigma=sigma)
        weight_sum = ndimage.gaussian_filter(valid_mask.astype(float), sigma=sigma)
        
        # Avoid division by zero
        weight_sum[weight_sum == 0] = 1
        
        # Calculate weighted average
        result = smoothed_values / weight_sum
        
        # Preserve NaN where we had no valid neighbors
        result[weight_sum < 0.1] = np.nan
        
        return result
    
    @staticmethod
    def compute_extent_mask(oct_image: np.ndarray,
                           geometric_mask: Optional[np.ndarray] = None,
                           adipose_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the valid tissue extent mask combining multiple criteria.
        
        Args:
            oct_image: OCT image (grayscale or RGB)
            geometric_mask: Optional geometric validity mask from warping
            adipose_mask: Optional mask identifying adipose tissue to exclude
            
        Returns:
            Boolean mask indicating valid tissue regions
        """
        # Convert to grayscale if needed
        if oct_image.ndim == 3:
            gray = cv2.cvtColor(oct_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = oct_image
        
        # Start with tissue presence (non-zero intensity)
        extent_mask = gray > 0
        
        # Apply geometric mask if provided
        if geometric_mask is not None:
            extent_mask &= geometric_mask
        
        # Exclude adipose tissue if mask provided
        if adipose_mask is not None:
            extent_mask &= ~adipose_mask
        
        return extent_mask
