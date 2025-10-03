"""
Canvas Rendering Module

Handles rendering of OCT images, overlays, and trajectories on Tkinter canvas.
"""

import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


class CanvasRenderer:
    """
    Manages rendering of images and overlays onto Tkinter canvas widgets.
    
    Handles:
    - Image scaling and positioning
    - Trajectory visualization
    - Bounding box overlays
    - Scale bars and colorbars
    """
    
    def __init__(self):
        """Initialize the canvas renderer."""
        self.quiver_step = 20  # Spacing for trajectory arrows
        self.arrow_width = 0.25  # Thickness of trajectory arrows
    
    def draw_image_with_overlays(self,
                                 canvas: tk.Canvas,
                                 image: np.ndarray,
                                 flow: Optional[np.ndarray] = None,
                                 show_trajectories: bool = False,
                                 show_net_vectors: bool = False,
                                 bounding_boxes: Optional[List[Tuple]] = None,
                                 trajectory_extent: Optional[Tuple[float, float, float, float]] = None) -> None:
        """
        Draw image with optional overlays on a Tkinter canvas.
        
        Args:
            canvas: Target Tkinter canvas widget
            image: Base image to display (RGB or grayscale)
            flow: Optional optical flow field for trajectories
            show_trajectories: Whether to show pixel trajectories
            show_net_vectors: Whether to show net displacement vectors
            bounding_boxes: List of (start, end, color) tuples for boxes
            trajectory_extent: Optional (x_min, x_max, y_min, y_max) for canvas bounds
        """
        if image is None:
            return
        
        # Convert to RGB if grayscale
        if image.ndim == 2:
            display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display = image.copy()
        
        # Convert to PIL Image
        pil_img = Image.fromarray(display.astype(np.uint8))
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width() or pil_img.width
        canvas_height = canvas.winfo_height() or pil_img.height
        
        # Calculate scaling factor considering trajectories if needed
        if trajectory_extent is not None and (show_trajectories or show_net_vectors):
            x_min, x_max, y_min, y_max = trajectory_extent
            
            # Calculate image bounds in trajectory space
            img_width = pil_img.width
            img_height = pil_img.height
            
            # Total extent including trajectories
            total_width = max(img_width, x_max - min(0, x_min))
            total_height = max(img_height, y_max - min(0, y_min))
            
            # Calculate scale to fit in canvas
            scale_x = canvas_width / total_width
            scale_y = canvas_height / total_height
            scale = min(scale_x, scale_y)
            
            # Offset to center the content
            offset_x = (canvas_width - total_width * scale) / 2 - min(0, x_min * scale)
            offset_y = (canvas_height - total_height * scale) / 2 - min(0, y_min * scale)
        else:
            # Standard scaling to fit canvas
            scale = min(canvas_width / pil_img.width, canvas_height / pil_img.height)
            new_width = int(pil_img.width * scale)
            new_height = int(pil_img.height * scale)
            offset_x = (canvas_width - new_width) / 2
            offset_y = (canvas_height - new_height) / 2
        
        # Resize image
        new_width = int(pil_img.width * scale)
        new_height = int(pil_img.height * scale)
        resized = pil_img.resize((new_width, new_height), Image.BILINEAR)
        
        # Draw on canvas
        canvas.delete("all")
        photo = ImageTk.PhotoImage(resized)
        canvas.create_image(offset_x, offset_y, anchor="nw", image=photo)
        canvas.image = photo  # Keep reference to prevent garbage collection
        
        # Draw trajectories if requested
        if flow is not None and (show_trajectories or show_net_vectors):
            self._draw_trajectories(
                canvas, flow, scale, offset_x, offset_y,
                show_trajectories, show_net_vectors
            )
        
        # Draw bounding boxes
        if bounding_boxes:
            self._draw_bounding_boxes(canvas, bounding_boxes, scale, offset_x, offset_y)
    
    def _draw_trajectories(self,
                          canvas: tk.Canvas,
                          cumulative_flow: np.ndarray,
                          scale: float,
                          offset_x: float,
                          offset_y: float,
                          show_full: bool = True,
                          show_net: bool = False) -> None:
        """
        Draw pixel trajectories on canvas.
        
        Args:
            canvas: Target canvas
            cumulative_flow: Cumulative flow field
            scale: Scaling factor
            offset_x, offset_y: Canvas offsets
            show_full: Whether to show full trajectories
            show_net: Whether to show net displacement vectors
        """
        H, W = cumulative_flow.shape[:2]
        step = self.quiver_step
        
        # Base color for trajectories (blue)
        base_color = np.array([30, 120, 200])
        
        for y in range(0, H, step):
            for x in range(0, W, step):
                # End point after cumulative displacement
                end_x = x + cumulative_flow[y, x, 0]
                end_y = y + cumulative_flow[y, x, 1]
                
                # Convert to canvas coordinates
                start_cx = x * scale + offset_x
                start_cy = y * scale + offset_y
                end_cx = end_x * scale + offset_x
                end_cy = end_y * scale + offset_y
                
                if show_full:
                    # Draw trajectory arrow
                    # Fade color based on displacement magnitude
                    magnitude = np.sqrt(cumulative_flow[y, x, 0]**2 + cumulative_flow[y, x, 1]**2)
                    alpha = min(1.0, magnitude / 10.0)  # Normalize to reasonable range
                    fade = (base_color * alpha + 255 * (1 - alpha)).astype(int)
                    color = f"#{fade[0]:02x}{fade[1]:02x}{fade[2]:02x}"
                    
                    canvas.create_line(
                        start_cx, start_cy, end_cx, end_cy,
                        arrow=tk.LAST, fill=color, width=self.arrow_width
                    )
                
                if show_net:
                    # Draw net displacement vector in lime
                    canvas.create_line(
                        start_cx, start_cy, end_cx, end_cy,
                        arrow=tk.LAST, fill='lime', width=self.arrow_width
                    )
    
    def _draw_bounding_boxes(self,
                            canvas: tk.Canvas,
                            boxes: List[Tuple],
                            scale: float,
                            offset_x: float,
                            offset_y: float) -> None:
        """
        Draw bounding boxes on canvas.
        
        Args:
            canvas: Target canvas
            boxes: List of (start_point, end_point, color) tuples
            scale: Scaling factor
            offset_x, offset_y: Canvas offsets
        """
        for start, end, color in boxes:
            if start and end:
                x0, y0 = start
                x1, y1 = end
                
                # Convert to canvas coordinates
                x0c = x0 * scale + offset_x
                y0c = y0 * scale + offset_y
                x1c = x1 * scale + offset_x
                y1c = y1 * scale + offset_y
                
                canvas.create_rectangle(x0c, y0c, x1c, y1c, outline=color, width=4)
    
    def draw_scale_bar(self,
                      canvas: tk.Canvas,
                      scale: float,
                      canvas_width: int,
                      canvas_height: int,
                      bar_length_mm: float = 1.0) -> None:
        """
        Draw a scale bar on the canvas.
        
        Args:
            canvas: Target canvas
            scale: Pixels per mm scaling factor
            canvas_width: Canvas width in pixels
            canvas_height: Canvas height in pixels
            bar_length_mm: Length of scale bar in mm
        """
        # Position scale bar in bottom-left corner
        margin = 20
        bar_length_px = int(bar_length_mm * scale)
        
        x1 = margin
        y1 = canvas_height - margin
        x2 = x1 + bar_length_px
        y2 = y1
        
        # Draw scale bar
        canvas.create_line(x1, y1, x2, y2, fill='white', width=3)
        
        # Draw end caps
        cap_height = 5
        canvas.create_line(x1, y1 - cap_height, x1, y1 + cap_height, fill='white', width=3)
        canvas.create_line(x2, y2 - cap_height, x2, y2 + cap_height, fill='white', width=3)
        
        # Draw label
        label_text = f"{bar_length_mm} mm"
        canvas.create_text(
            (x1 + x2) / 2, y1 - 15,
            text=label_text, fill='white',
            font=('Arial', 9, 'bold')
        )
    
    def draw_colorbar(self,
                     canvas: tk.Canvas,
                     canvas_width: int,
                     canvas_height: int,
                     colormap_name: str,
                     vmin: float,
                     vmax: float,
                     label: str,
                     use_log_scale: bool = False) -> None:
        """
        Draw a colorbar legend on the canvas.
        
        Args:
            canvas: Target canvas
            canvas_width, canvas_height: Canvas dimensions
            colormap_name: Name of matplotlib colormap
            vmin, vmax: Value range
            label: Colorbar label text
            use_log_scale: Whether values are in log scale
        """
        # Colorbar dimensions
        cb_width = 15
        cb_height = 150
        margin = 20
        
        # Position in top-right corner
        x1 = canvas_width - margin - cb_width
        y1 = margin
        
        # Create colorbar gradient
        cmap = plt.cm.get_cmap(colormap_name)
        gradient = np.linspace(1, 0, cb_height)[:, np.newaxis]
        gradient = np.repeat(gradient, cb_width, axis=1)
        
        # Apply colormap
        rgba = cmap(gradient)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        
        # Convert to PIL and display
        cb_img = Image.fromarray(rgb)
        cb_photo = ImageTk.PhotoImage(cb_img)
        canvas.create_image(x1, y1, anchor='nw', image=cb_photo)
        
        # Keep reference
        if not hasattr(canvas, '_colorbar_images'):
            canvas._colorbar_images = []
        canvas._colorbar_images.append(cb_photo)
        
        # Draw border
        canvas.create_rectangle(x1, y1, x1 + cb_width, y1 + cb_height,
                              outline='white', width=1)
        
        # Draw labels
        if use_log_scale:
            # Convert from log scale for display
            vmax_display = 10**vmax
            vmin_display = 10**vmin
        else:
            vmax_display = vmax
            vmin_display = vmin
        
        canvas.create_text(x1 + cb_width + 5, y1,
                         text=f"{vmax_display:.1f}", fill='white',
                         font=('Arial', 8), anchor='w')
        canvas.create_text(x1 + cb_width + 5, y1 + cb_height,
                         text=f"{vmin_display:.1f}", fill='white',
                         font=('Arial', 8), anchor='w')
        canvas.create_text(x1 + cb_width + 5, y1 + cb_height//2,
                         text=label, fill='white',
                         font=('Arial', 9, 'bold'), anchor='w', angle=90)
