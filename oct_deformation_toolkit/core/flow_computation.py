"""
Optical Flow Computation Module

Manages computation and composition of optical flow fields for image registration.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from tracking_algorithms import TrackingAlgorithms


class OpticalFlowEngine:
    """
    Computes and manages optical flow fields for OCT image registration.
    
    This class handles:
    - Computing incremental optical flow between consecutive frames
    - Composing flows to get cumulative displacement fields
    - Caching flow computations for efficiency
    
    Attributes:
        algorithm_name (str): Name of the optical flow algorithm to use
        cumulative_flows (dict): Cached cumulative flow fields
        incremental_flows (dict): Cached incremental flow fields
    """
    
    def __init__(self, algorithm_name: str = "DIS"):
        """
        Initialize the optical flow engine.
        
        Args:
            algorithm_name: Name of algorithm ("DIS", "Farneback", "PCAFlow", etc.)
        """
        self.algorithm_name = algorithm_name
        self.tracking = TrackingAlgorithms()
        self.cumulative_flows: Dict[int, np.ndarray] = {}
        self.incremental_flows: Dict[int, np.ndarray] = {}
        self._reference_frame: Optional[np.ndarray] = None
    
    def set_algorithm(self, algorithm_name: str):
        """Change the optical flow algorithm."""
        self.algorithm_name = algorithm_name
        self.clear_cache()
    
    def clear_cache(self):
        """Clear all cached flow computations."""
        self.cumulative_flows.clear()
        self.incremental_flows.clear()
        self._reference_frame = None
    
    def precompute_flows(self, frames: list) -> None:
        """
        Precompute all optical flow fields for a sequence.
        
        Args:
            frames: List of grayscale images (numpy arrays)
        """
        if not frames:
            return
        
        self.clear_cache()
        self._reference_frame = frames[0]
        
        # Compute incremental flows
        for i in range(1, len(frames)):
            prev_frame = self._to_grayscale(frames[i - 1])
            curr_frame = self._to_grayscale(frames[i])
            
            flow = self._compute_optical_flow(prev_frame, curr_frame)
            self.incremental_flows[i] = flow
        
        # Compose cumulative flows
        self.cumulative_flows[0] = np.zeros((*frames[0].shape[:2], 2), dtype=np.float32)
        
        for i in range(1, len(frames)):
            if i == 1:
                self.cumulative_flows[i] = self.incremental_flows[i]
            else:
                prev_cumulative = self.cumulative_flows[i - 1]
                curr_incremental = self.incremental_flows[i]
                self.cumulative_flows[i] = self._compose_flows(
                    prev_cumulative, curr_incremental
                )
    
    def get_cumulative_flow(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get the cumulative flow from reference frame to specified frame.
        
        Args:
            frame_index: Index of target frame
            
        Returns:
            Cumulative flow field (H, W, 2) or None if not available
        """
        return self.cumulative_flows.get(frame_index)
    
    def get_incremental_flow(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get the incremental flow from frame (index-1) to frame (index).
        
        Args:
            frame_index: Index of current frame
            
        Returns:
            Incremental flow field (H, W, 2) or None if not available
        """
        return self.incremental_flows.get(frame_index)
    
    def warp_image(self, image: np.ndarray, flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warp an image using a flow field.
        
        Args:
            image: Input image to warp
            flow: Flow field (H, W, 2)
            
        Returns:
            Tuple of (warped_image, validity_mask)
        """
        H, W = flow.shape[:2]
        
        # Create remapping coordinates
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        grid_x = xs[None, :].repeat(H, axis=0)
        grid_y = ys[:, None].repeat(W, axis=1)
        
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        
        # Warp the image
        warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Create validity mask
        valid_x = (map_x >= 0) & (map_x < W)
        valid_y = (map_y >= 0) & (map_y < H)
        mask = valid_x & valid_y
        
        return warped, mask
    
    def _compute_optical_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Compute optical flow between two frames using the selected algorithm.
        
        Args:
            img1: First frame (grayscale)
            img2: Second frame (grayscale)
            
        Returns:
            Flow field (H, W, 2)
        """
        # Use the tracking algorithms module
        if self.algorithm_name == "DIS":
            return self.tracking.dis_optical_flow(img1, img2)
        elif self.algorithm_name == "Farneback":
            return self.tracking.farneback(img1, img2)
        elif self.algorithm_name == "PCAFlow":
            return self.tracking.pca_flow(img1, img2)
        elif self.algorithm_name == "DeepFlow":
            return self.tracking.deep_flow(img1, img2)
        elif self.algorithm_name == "TVL1":
            return self.tracking.tvl1_optical_flow(img1, img2)
        else:
            # Default to Farneback
            return self.tracking.farneback(img1, img2)
    
    def _compose_flows(self, flow_0_to_k_minus_1: np.ndarray, 
                      flow_k_minus_1_to_k: np.ndarray) -> np.ndarray:
        """
        Compose two optical flows: flow(0->k) = flow(0->k-1) ∘ flow(k-1->k).
        
        Args:
            flow_0_to_k_minus_1: Cumulative flow from frame 0 to k-1
            flow_k_minus_1_to_k: Incremental flow from frame k-1 to k
            
        Returns:
            Composed cumulative flow from frame 0 to k
        """
        H, W = flow_0_to_k_minus_1.shape[:2]
        
        # Create coordinate grids
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        grid_x = xs[None, :].repeat(H, axis=0)
        grid_y = ys[:, None].repeat(W, axis=1)
        
        # Map coordinates from frame 0 to frame k-1
        map_x = grid_x + flow_0_to_k_minus_1[..., 0]
        map_y = grid_y + flow_0_to_k_minus_1[..., 1]
        
        # Sample the incremental flow at these warped positions
        warped_inc = np.zeros_like(flow_k_minus_1_to_k)
        for c in range(2):
            warped_inc[..., c] = cv2.remap(
                flow_k_minus_1_to_k[..., c], map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        
        # Compose the flows
        return flow_0_to_k_minus_1 + warped_inc
    
    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
