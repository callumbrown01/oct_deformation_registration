#!/usr/bin/env python3
import os
import re
import cv2
import numpy as np
import scipy.io as sio
from scipy import ndimage
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import glob
from tracking_algorithms import TrackingAlgorithms

# ------------------------------- helpers --------------------------------------
def compose_flows(fwd_0_to_kminus1, fwd_kminus1_to_k):
    """Compose optical flows: flow(0->k) = flow(0->k-1) ∘ flow(k-1->k)."""
    H, W = fwd_0_to_kminus1.shape[:2]
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    grid_x = xs[None, :].repeat(H, axis=0)
    grid_y = ys[:, None].repeat(W, axis=1)
    map_x = grid_x + fwd_0_to_kminus1[..., 0]
    map_y = grid_y + fwd_0_to_kminus1[..., 1]
    warped_inc = np.zeros_like(fwd_kminus1_to_k)
    for c in range(2):
        warped_inc[..., c] = cv2.remap(
            fwd_kminus1_to_k[..., c], map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    return fwd_0_to_kminus1 + warped_inc

# ------------------------------ main widget -----------------------------------
class StressAttenuationExplorer(ttk.Frame):
    """
    OCT Deformation Tracking Suite for analyzing stress-attenuation relationships
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # ── Root layout using PanedWindow for better resizing
        root_panes = ttk.Panedwindow(self, orient='horizontal')
        root_panes.pack(fill='both', expand=True)

        # Left (tools)
        tools_frame = ttk.Frame(root_panes, padding=(8,8))
        root_panes.add(tools_frame, weight=0)
        self.tools_frame = tools_frame

        # Right (views)
        views_frame = ttk.Frame(root_panes)
        root_panes.add(views_frame, weight=3)

        # ── Top controls (in tools)
        ctrl_top = ttk.Frame(tools_frame)
        ctrl_top.pack(fill='x', pady=(0,8))
        ttk.Button(ctrl_top, text="Select Sequence Folder", command=self._select_sequence_root)\
            .pack(side='left')
        self.seq_label = ttk.Label(ctrl_top, text="No folder", width=24)
        self.seq_label.pack(side='left', padx=8)
        ttk.Button(ctrl_top, text="Export Data", command=self._show_export_dialog)\
            .pack(side='right')

        # --- Algorithm selection dropdown ---
        self.tracking = TrackingAlgorithms()
        self.algorithm_names = self.tracking.get_algorithm_names()
        self.selected_algorithm = tk.StringVar(value="DIS")  
        algo_frame = ttk.Labelframe(self.tools_frame, text="Tracking Algorithm (DIS recommended)", padding=6)
        algo_frame.pack(fill='x', pady=(0,6))
        ttk.Label(algo_frame, text='Algorithm:').pack(side='left')
        algo_dropdown = ttk.Combobox(algo_frame, textvariable=self.selected_algorithm, values=self.algorithm_names, state='readonly')
        algo_dropdown.pack(side='left', padx=8)
        algo_dropdown.bind("<<ComboboxSelected>>", self._on_algorithm_change)

        # Options
        opt = ttk.Labelframe(tools_frame, text="Compare (Left Canvas)", padding=6)
        opt.pack(fill='x', pady=6)
        self.show_original_var = tk.BooleanVar(value=False)
        self.show_unwarped_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Toggle First Frame",
                        variable=self.show_original_var,
                        command=self._refresh_display).pack(anchor='w')
        ttk.Checkbutton(opt, text="Toggle Current Frame",
                        variable=self.show_unwarped_var,
                        command=self._refresh_display).pack(anchor='w')

        # Overlays
        overlays = ttk.Labelframe(tools_frame, text="Add Overlays (Right Canvas)", padding=6)
        overlays.pack(fill='x', pady=6)
        self.overlay_att_var   = tk.BooleanVar(value=False)
        self.overlay_str_var   = tk.BooleanVar(value=False)
        self.overlay_resp_var  = tk.BooleanVar(value=True)
        self.show_headtail_var = tk.BooleanVar(value=False)
        self.show_net_var      = tk.BooleanVar(value=False)
        self.sensitivity = tk.DoubleVar(value=9.0)
        self.mask_threshold_high = tk.DoubleVar(value=5.5)
        self.mask_min_area = tk.IntVar(value=12)
        self.att_smoothing = tk.DoubleVar(value=0.5)

        ttk.Checkbutton(overlays, text="Attenuation",
                        variable=self.overlay_att_var,
                        command=self._on_overlay_toggle).pack(anchor='w')
        ttk.Checkbutton(overlays, text="Stress",
                        variable=self.overlay_str_var,
                        command=self._on_overlay_toggle).pack(anchor='w')
        ttk.Checkbutton(overlays, text="Attenuation Load Response (ΔAtt/ΔStr)",
                        variable=self.overlay_resp_var,
                        command=self._on_overlay_toggle).pack(anchor='w')
        ttk.Separator(overlays, orient='horizontal').pack(fill='x', pady=6)
        ttk.Checkbutton(overlays, text="Head–Tail Pixel Trajectories",
                        variable=self.show_headtail_var,
                        command=self._refresh_display).pack(anchor='w')
        ttk.Checkbutton(overlays, text="Net Pixel Trajectories",
                        variable=self.show_net_var,
                        command=self._refresh_display).pack(anchor='w')

        # Attenuation/Stress Analysis subsection
        att_str_frame = ttk.Labelframe(overlays, text="Attenuation/Stress Analysis", padding=6)
        att_str_frame.pack(fill='x', pady=8)
        
        # Sensitivity slider
        sensitivity_frame = ttk.Frame(att_str_frame)
        sensitivity_frame.pack(fill='x', pady=2)
        ttk.Label(sensitivity_frame, text="Sensitivity: Low").pack(side='left')
        sensitivity_slider = ttk.Scale(
            sensitivity_frame,
            from_=0.1, to=10.0,
            variable=self.sensitivity,
            orient='horizontal',
            command=lambda _: self._refresh_display()
        )
        sensitivity_slider.pack(side='left', fill='x', expand=True, padx=4)
        ttk.Label(sensitivity_frame, text="High").pack(side='left')

        # Attenuation smoothing slider
        att_smooth_frame = ttk.Frame(att_str_frame)
        att_smooth_frame.pack(fill='x', pady=2)
        ttk.Label(att_smooth_frame, text="Att. Smoothing: None").pack(side='left')
        att_smooth_slider = ttk.Scale(
            att_smooth_frame,
            from_=0.0, to=3.0,
            variable=self.att_smoothing,
            orient='horizontal',
            command=lambda _: self._refresh_display()
        )
        att_smooth_slider.pack(side='left', fill='x', expand=True, padx=4)
        ttk.Label(att_smooth_frame, text="Strong").pack(side='left')

        # Masking Controls subsection
        mask_frame = ttk.Labelframe(overlays, text="Masking Controls", padding=6)
        mask_frame.pack(fill='x', pady=8)

        # Mask range (threshold high) slider
        mask_range_frame = ttk.Frame(mask_frame)
        mask_range_frame.pack(fill='x', pady=2)
        ttk.Label(mask_range_frame, text="Mask Range: Low").pack(side='left')
        mask_range_slider = ttk.Scale(
            mask_range_frame,
            from_=1.0, to=15.0,
            variable=self.mask_threshold_high,
            orient='horizontal',
            command=lambda _: self._refresh_display()
        )
        mask_range_slider.pack(side='left', fill='x', expand=True, padx=4)
        ttk.Label(mask_range_frame, text="High").pack(side='left')

        # Mask sensitivity (min area) slider
        mask_sensitivity_frame = ttk.Frame(mask_frame)
        mask_sensitivity_frame.pack(fill='x', pady=2)
        ttk.Label(mask_sensitivity_frame, text="Mask Sensitivity: Low").pack(side='left')
        mask_sensitivity_slider = ttk.Scale(
            mask_sensitivity_frame,
            from_=5, to=50,
            variable=self.mask_min_area,
            orient='horizontal',
            command=lambda _: self._refresh_display()
        )
        mask_sensitivity_slider.pack(side='left', fill='x', expand=True, padx=4)
        ttk.Label(mask_sensitivity_frame, text="High").pack(side='left')

        # Navigation
        nav = ttk.Frame(tools_frame)
        nav.pack(fill='x', pady=8)
        ttk.Button(nav, text="<< Prev", command=lambda: self._advance(-1)).pack(side='left')
        ttk.Button(nav, text="Next >>", command=lambda: self._advance(1)).pack(side='left', padx=6)
        self.index_label = ttk.Label(nav, text="Frame: N/A")
        self.index_label.pack(side='left', padx=8)

        # Scatter graph
        graph_box = ttk.Labelframe(tools_frame, text="Attenuation Response to Stress", padding=6)
        graph_box.pack(fill='both', expand=True, pady=(4,0))
        self.fig, self.ax = plt.subplots(figsize=(4,4))
        self.ax.set_aspect('equal', adjustable='box')
        self.scatter_canvas = FigureCanvasTkAgg(self.fig, master=graph_box)
        self.scatter_widget = self.scatter_canvas.get_tk_widget()
        self.scatter_widget.pack(fill='both', expand=True)
        self.ax.set_xlabel("Mean Stress (log₁₀ kPa)", fontsize=10)
        self.ax.set_ylabel("Mean Attenuation (mm⁻¹)", fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_facecolor("#f7f7f7")
        self.fig.tight_layout()

        tools_frame.bind("<Configure>", self._on_tools_resize)

        # ── Right: two canvases side-by-side
        right_top = ttk.Frame(views_frame)
        right_top.pack(fill='both', expand=True, padx=8, pady=8)

        left_area  = ttk.Frame(right_top)
        right_area = ttk.Frame(right_top)
        left_area.pack(side='left', fill='both', expand=True, padx=(0,4))
        right_area.pack(side='left', fill='both', expand=True, padx=(4,0))

        ttk.Label(left_area, text="Compare Images").pack(anchor='w')
        self.canvas_left = tk.Canvas(left_area, bg="black", highlightthickness=0)
        self.canvas_left.pack(fill='both', expand=True, pady=(2,0))
        self.canvas_left.bind("<ButtonPress-1>", self._on_box_start)
        self.canvas_left.bind("<B1-Motion>", self._on_box_drag)
        self.canvas_left.bind("<ButtonRelease-1>", self._on_box_end)

        ttk.Label(right_area, text="Add Overlays").pack(anchor='w')
        self.canvas_right = tk.Canvas(right_area, bg="black", highlightthickness=0)
        self.canvas_right.pack(fill='both', expand=True, pady=(2,0))
        self.canvas_right.bind("<ButtonPress-1>", self._on_box_start)
        self.canvas_right.bind("<B1-Motion>", self._on_box_drag)
        self.canvas_right.bind("<ButtonRelease-1>", self._on_box_end)

        # ── State
        self.sequence_root = None
        self.images = {"oct": [], "stress": [], "att": []}
        self.scalar_images = {"stress": [], "att": []}
        self.current_idx = 0
        self.cumulative = {"shared_flow": {}}
        self.incremental_flows = {}
        self.quiver_step = 20
        self.trajectory_s = []; self.trajectory_a = []
        self.box_start = None; self.box_end = None
        self._cursor = None
        self._active_canvas = None

        # Fixed overlay opacities
        self._alpha_att   = 0.5
        self._alpha_str   = 0.5
        
        # Scale and colorbar settings
        self.pixels_per_mm = 364 / 6.0  # 364 pixels = 6mm
        self._current_overlay_type = None
        self._current_overlay_data = None

    # ------------------------------- UI helpers --------------------------------
    def _on_overlay_toggle(self):
        self._refresh_display()

    def _safe_get(self, seq, idx):
        return seq[idx] if 0 <= idx < len(seq) else None

    # ------------------------------- data I/O ----------------------------------
    def _show_loading_dialog(self, message):
        top = tk.Toplevel(self)
        top.title("Loading")
        top.transient(self.winfo_toplevel())
        top.grab_set()
        top.resizable(False, False)
        ttk.Label(top, text=message).pack(padx=16, pady=(16,8))
        pb = ttk.Progressbar(top, mode='indeterminate', length=220)
        pb.pack(padx=16, pady=(0,16))
        pb.start(10)
        self.update_idletasks()
        px = self.winfo_rootx() + self.winfo_width()//2 - top.winfo_reqwidth()//2
        py = self.winfo_rooty() + self.winfo_height()//2 - top.winfo_reqheight()//2
        top.geometry(f"+{px}+{py}")
        return top, pb

    def _precompute_flows(self):
        total = len(self.images["oct"])
        self.cumulative = {"shared_flow": {}}
        self.incremental_flows = {}
        ref = self._to_gray(self.images["oct"][0]) if total > 0 else None
        if ref is None:
            return
        cumulative_flow = np.zeros((*ref.shape, 2), dtype=np.float32)
        prev = ref
        for idx in range(1, total):
            curr = self._to_gray(self.images["oct"][idx])
            flow_inc = self._compute_incremental_flow(prev, curr)
            self.incremental_flows[idx] = flow_inc
            cumulative_flow = flow_inc.copy() if idx == 1 else compose_flows(cumulative_flow, flow_inc)
            self.cumulative["shared_flow"][idx] = cumulative_flow.copy()
            prev = curr

    def _select_sequence_root(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        loading, pb = self._show_loading_dialog("Loading sequence and flows...")
        self.update_idletasks()
        try:
            self.sequence_root = folder
            self.seq_label.config(text=os.path.basename(folder))
            self._load_modalities()
            self.current_idx = 0
            self.trajectory_s = []; self.trajectory_a = []
            self.box_start = None; self.box_end = None
            self._precompute_flows()
        finally:
            loading.destroy()
        messagebox.showinfo("Flows Loaded", "Flows have been precomputed and loaded.")
        self._refresh_display()

    def _on_algorithm_change(self, event=None):
        loading, pb = self._show_loading_dialog("Loading flows for selected algorithm...")
        self.update_idletasks()
        try:
            self._precompute_flows()
        finally:
            loading.destroy()
        messagebox.showinfo("Flows Loaded", "Flows have been precomputed and loaded.")
        self._refresh_display()

    def _load_modalities(self):
        """Load OCT, Stress and Attenuation data from .mat files"""
        self.images = {"oct": [], "stress": [], "att": []}
        self.scalar_images = {"stress": [], "att": []}
        
        oct_files = glob.glob(os.path.join(self.sequence_root, "*OCT*.mat"))
        stress_files = glob.glob(os.path.join(self.sequence_root, "*Stress*.mat"))
        att_files = glob.glob(os.path.join(self.sequence_root, "*Attenuation*.mat"))
        
        if not oct_files or not stress_files or not att_files:
            print("Missing required .mat files")
            return
            
        try:
            oct_data = sio.loadmat(oct_files[0])
            stress_data = sio.loadmat(stress_files[0])
            att_data = sio.loadmat(att_files[0])
            
            # Find arrays with correct shape (364x364xN)
            oct_array = None
            stress_array = None
            att_array = None
            
            for key, value in oct_data.items():
                if key.startswith('__'): continue
                if isinstance(value, np.ndarray) and len(value.shape) == 3 and value.shape[:2] == (364, 364):
                    oct_array = value
                    break
                    
            for key, value in stress_data.items():
                if key.startswith('__'): continue
                if isinstance(value, np.ndarray) and len(value.shape) == 3 and value.shape[:2] == (364, 364):
                    stress_array = value
                    break
                    
            for key, value in att_data.items():
                if key.startswith('__'): continue
                if isinstance(value, np.ndarray) and len(value.shape) == 3 and value.shape[:2] == (364, 364):
                    att_array = value
                    break
            
            if oct_array is None or stress_array is None or att_array is None:
                print("Could not find valid arrays in .mat files")
                return
                
            num_frames = oct_array.shape[2]
            
            # OCT images - normalize to 0-255 uint8
            for i in range(num_frames):
                slice_data = oct_array[:,:,i]
                normalized = np.clip(slice_data, 0, 29)
                normalized = ((normalized / 29.0) * 255).astype(np.uint8)
                self.images["oct"].append(normalized)
            
            for i in range(num_frames):
                slice_data = stress_array[:,:,i]
                eps = 1e-6
                # Ensure we have negative stress values, make them positive for log
                stress_positive = np.abs(slice_data) + eps
                log_stress = np.log10(stress_positive)
                # Clamp log stress values to 0-2 range
                log_stress = np.clip(log_stress, 0, 2)
                self.scalar_images["stress"].append(log_stress)
                # Convert to RGB visualization using jet colormap for log10 values 0-2
                rgb = self._apply_colormap(log_stress, 'jet', 0, 2)
                self.images["stress"].append(rgb)
                
            for i in range(num_frames):
                slice_data = att_array[:,:,i]
                slice_data = np.clip(slice_data, 0, 10)
                # Store unblurred attenuation data
                self.scalar_images["att"].append(slice_data)
                rgb = self._apply_colormap(slice_data, 'viridis', 0, 10)
                self.images["att"].append(rgb)

            total = len(self.images["oct"])
            self.index_label.config(text=f"Frame: {self.current_idx+1}/{total}")
            
        except Exception as e:
            print(f"Error loading .mat files: {e}")
            return

    def _apply_colormap(self, data, cmap_name, vmin, vmax):
        """Convert scalar data to RGB using matplotlib colormap"""
        valid_data = np.nan_to_num(data, nan=vmin)
        norm = (np.clip(valid_data, vmin, vmax) - vmin) / (vmax - vmin)
        cmap = matplotlib.colormaps[cmap_name]
        rgb = cmap(norm)[..., :3]
        return (rgb * 255).astype(np.uint8)

    def _advance(self, delta):
        if not self.images["oct"]:
            return
        self.current_idx = int(np.clip(self.current_idx + delta, 0, len(self.images["oct"]) - 1))
        self.index_label.config(text=f"Frame: {self.current_idx+1}/{len(self.images['oct'])}")
        self._refresh_display()

    # ------------------------------ optical flow --------------------------------
    def _to_gray(self, img):
        if img is None: return None
        if img.ndim == 2: return img
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def _compute_incremental_flow(self, prev, curr):
        alg_name = self.selected_algorithm.get()
        prev_gray = prev if prev.ndim == 2 else cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
        curr_gray = curr if curr.ndim == 2 else cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
        flow = self.tracking.run(alg_name, prev_gray, curr_gray)
        return flow

    def _get_cumulative_flow(self, index):
        if index in self.cumulative["shared_flow"]:
            return self.cumulative["shared_flow"][index]
        if not self.images["oct"] or index >= len(self.images["oct"]):
            return None
        ref = self._to_gray(self.images["oct"][0])
        cumulative_flow = np.zeros((*ref.shape, 2), dtype=np.float32)
        prev = ref
        for idx in range(1, index + 1):
            curr = self._to_gray(self.images["oct"][idx])
            flow_inc = self._compute_incremental_flow(prev, curr)
            self.incremental_flows[idx] = flow_inc
            cumulative_flow = flow_inc.copy() if idx == 1 else compose_flows(cumulative_flow, flow_inc)
            prev = curr
        self.cumulative["shared_flow"][index] = cumulative_flow
        return cumulative_flow

    def _warp_with_shared_flow(self, arr, index):
        """Warp array (RGB or single-channel) defined in frame 'index' to frame-0 geometry."""
        if arr is None:
            return None, None
        flow = self._get_cumulative_flow(index)
        if flow is None:
            return None, None
        H, W = flow.shape[:2]
        xs = np.arange(W, dtype=np.float32); ys = np.arange(H, dtype=np.float32)
        map_x = xs[None, :] + flow[..., 0]
        map_y = ys[:, None] + flow[..., 1]
        if arr.ndim == 2:
            warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            warped = np.zeros_like(arr)
            for ch in range(arr.shape[2]):
                warped[..., ch] = cv2.remap(arr[..., ch], map_x, map_y,
                                            interpolation=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_mask = (map_x >= 0)&(map_x<=W-1)&(map_y>=0)&(map_y<=H-1)
        return warped, valid_mask

    # ------------------------------ RGB utils & blending ------------------------
    @staticmethod
    def _to_rgb(img):
        if img is None: return None
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            g = img.astype(np.uint8)
            if g.ndim == 3: g = g[..., 0]
            return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 3:
            return img.astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 4:
            arr = img.astype(np.uint8)
            try:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            except Exception:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        return img[..., :3].astype(np.uint8)

    def _alpha_blend_masked(self, base_rgb, over_rgb, alpha, mask):
        base_rgb = self._to_rgb(base_rgb)
        over_rgb = self._to_rgb(over_rgb)
        if base_rgb is None or over_rgb is None:
            return base_rgb if base_rgb is not None else over_rgb
        if over_rgb.shape[:2] != base_rgb.shape[:2]:
            over_rgb = cv2.resize(over_rgb, (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        out = base_rgb.copy()
        if mask is None:
            bf = base_rgb.astype(np.float32); of = over_rgb.astype(np.float32)
            out = np.clip((1.0 - alpha) * bf + alpha * of, 0, 255).astype(np.uint8)
            return out
        m = mask.astype(bool)
        if not np.any(m):
            return out
        bf = base_rgb[m].astype(np.float32)
        of = over_rgb[m].astype(np.float32)
        out[m] = np.clip((1.0 - alpha) * bf + alpha * of, 0, 255).astype(np.uint8)
        return out

    def _draw_scale_bar(self, canvas, scale, offset_x, offset_y, canvas_width, canvas_height):
        """Draw a scale bar on the canvas"""
        # Scale bar dimensions
        scale_length_mm = 1.0  # 1mm scale bar
        scale_length_pixels = int(scale_length_mm * self.pixels_per_mm * scale)
        scale_height = 4
        margin = 20
        
        # Position scale bar at bottom right
        x2 = canvas_width - margin
        x1 = x2 - scale_length_pixels
        y2 = canvas_height - margin
        y1 = y2 - scale_height
        
        # Draw scale bar
        canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='black', width=1)
        
        # Draw scale bar label above the bar
        text_x = x1 + scale_length_pixels // 2
        text_y = y1 - 8
        canvas.create_text(text_x, text_y, text=f'{scale_length_mm:.0f}mm', 
                          fill='white', font=('Arial', 10, 'bold'))

    def _get_canvas_title(self, canvas):
        """Get the appropriate title for the given canvas"""
        if canvas == self.canvas_left:
            # Left canvas title based on what's being displayed
            if self.show_original_var.get():
                return "First Frame"
            elif self.show_unwarped_var.get():
                return "Current Frame (Unwarped)"
            else:
                return "Current Frame (Warped)"
        elif canvas == self.canvas_right:
            # Right canvas shows overlay information (handled in color bar)
            return None
        return None

    def _draw_color_bar(self, canvas, canvas_width, canvas_height):
        """Draw a color bar for the current overlay"""
        if self._current_overlay_type is None:
            return
            
        # Color bar dimensions
        cb_width = 20
        cb_height = 150
        margin = 10  # Changed from 20 to 10
        label_space = 50  # Space for labels
        
        # Position at top right with space for labels
        x1 = canvas_width - margin - cb_width
        y1 = margin
        
        # Create color bar based on overlay type
        if self._current_overlay_type == 'attenuation':
            # Viridis colormap for attenuation (0-10 mm⁻¹)
            cmap = plt.cm.viridis
            vmin, vmax = 0, 10
            unit = 'mm⁻¹'
            label = 'Attenuation (mm⁻¹)'
            
        elif self._current_overlay_type == 'stress':
            # Jet colormap for stress (0-2 log₁₀ kPa)
            cmap = plt.cm.jet
            vmin, vmax = 0, 2
            unit = 'log₁₀ kPa'
            label = 'Stress (log₁₀ kPa)'
            
        elif self._current_overlay_type == 'att_str_ratio':
            # RdBu_r colormap for stability (0-1)
            cmap = plt.cm.RdBu_r
            vmin, vmax = 0, 1
            unit = ''
            label = 'Attenuation Load Response (mm⁻¹/log₁₀ kPa)'
        else:
            return
        
        # Draw color bar background
        canvas.create_rectangle(x1-2, y1-2, x1+cb_width+2, y1+cb_height+2, 
                               fill='white', outline='black', width=1)
        
        # Create color gradient
        for i in range(cb_height):
            # Normalize position to [0,1] range
            norm_pos = (cb_height - 1 - i) / (cb_height - 1)  # Flip y-axis
            color_rgba = cmap(norm_pos)
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(color_rgba[0] * 255),
                int(color_rgba[1] * 255),
                int(color_rgba[2] * 255)
            )
            canvas.create_line(x1, y1 + i, x1 + cb_width, y1 + i, fill=color_hex, width=1)
        
        # Add tick marks and labels
        num_ticks = 5
        for i in range(num_ticks):
            tick_pos = i / (num_ticks - 1)
            y_pos = y1 + cb_height - int(tick_pos * cb_height)
            value = vmin + tick_pos * (vmax - vmin)
            
            # Draw tick mark on the left side of the color bar
            canvas.create_line(x1 - 8, y_pos, x1, y_pos, 
                              fill='white', width=2)
            
            # Draw value label
            if self._current_overlay_type == 'att_str_ratio':
                value_text = f'{value:.1f}'
            else:
                value_text = f'{value:.0f}' if value == int(value) else f'{value:.1f}'
            
            # Draw white text to the left of the tick marks
            text_x = x1 - 15
            canvas.create_text(text_x, y_pos, text=value_text, 
                              fill='white', font=('Arial', 9, 'bold'), anchor='e')
        
        # Add title and units in top left corner of canvas
        if unit:
            title_text = f'{label} ({unit})'
        else:
            title_text = label
        canvas.create_text(10, 10, text=title_text, 
                          fill='white', font=('Arial', 10, 'bold'), anchor='nw')

    def _draw_grayscale_color_bar(self, canvas, canvas_width, canvas_height):
        """Draw a grayscale color bar for OCT intensity values"""
        # Color bar dimensions
        cb_width = 20
        cb_height = 150
        margin = 10  # Changed from 20 to 10
        
        # Position at top right
        x1 = canvas_width - margin - cb_width
        y1 = margin
        
        # Draw color bar background
        canvas.create_rectangle(x1-2, y1-2, x1+cb_width+2, y1+cb_height+2, 
                               fill='white', outline='black', width=1)
        
        # Create grayscale gradient (0-29 OCT intensity scale)
        for i in range(cb_height):
            # Normalize position to [0,1] range
            norm_pos = (cb_height - 1 - i) / (cb_height - 1)  # Flip y-axis
            # Map to 0-29 range, then convert to 0-255 for display
            oct_value = norm_pos * 29
            gray_value = int((oct_value / 29) * 255)
            color_hex = '#{:02x}{:02x}{:02x}'.format(gray_value, gray_value, gray_value)
            canvas.create_line(x1, y1 + i, x1 + cb_width, y1 + i, fill=color_hex, width=1)
        
        # Add tick marks and labels
        num_ticks = 5
        vmin, vmax = 0, 29  # Changed to OCT scale
        for i in range(num_ticks):
            tick_pos = i / (num_ticks - 1)
            y_pos = y1 + cb_height - int(tick_pos * cb_height)
            value = vmin + tick_pos * (vmax - vmin)
            
            # Draw tick mark on the left side of the color bar
            canvas.create_line(x1 - 8, y_pos, x1, y_pos, 
                              fill='white', width=2)
            
            # Draw value label
            value_text = f'{value:.0f}' if value == int(value) else f'{value:.1f}'
            
            # Draw white text to the left of the tick marks
            text_x = x1 - 15
            canvas.create_text(text_x, y_pos, text=value_text, 
                              fill='white', font=('Arial', 9, 'bold'), anchor='e')

    def _create_adipose_mask(self, oct_img, threshold_low=0, threshold_high=5.5, min_area=12):
        """Create mask for adipose tissue (very low signal regions in 0-29 scale)"""
        if oct_img is None:
            return None
        
        # Convert to grayscale if needed
        if oct_img.ndim == 3:
            gray = cv2.cvtColor(oct_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = oct_img.copy()
        
        # Normalize to 0-29 scale if it's in 0-255 range
        if gray.max() > 29:
            gray = (gray / 255.0) * 29.0
        
        # Find very low signal areas (0-2 in 0-29 scale)
        low_signal_mask = (gray >= threshold_low) & (gray <= threshold_high)
        
        # Remove small isolated pixels using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(low_signal_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Find connected components and filter by area
        # For 3 pixels diameter, area should be approximately π * (1.5)² ≈ 7 pixels minimum
        # Using min_area=28 for larger regions (approximately 6 pixel diameter)
        num_labels, labels = cv2.connectedComponents(cleaned_mask)
        final_mask = np.zeros_like(gray, dtype=bool)
        
        for label in range(1, num_labels):
            component_mask = labels == label
            if np.sum(component_mask) >= min_area:
                final_mask |= component_mask
                
        return final_mask

    # ------------------------------ render & metrics ----------------------------
    def _refresh_display(self):
        if not self.sequence_root or not self.images["oct"]:
            return

        # get the number of frames available
        total = len(self.images["oct"])
        self.index_label.config(text=f"Frame: {self.current_idx+1}/{total}")

        # the raw oct image at frame 0
        base_oct_raw0 = self._safe_get(self.images["oct"], 0)
        # the raw oct image at current frame
        cur_oct_raw   = self._safe_get(self.images["oct"], self.current_idx)

        # Helper: convert to gray float32, None -> zeros like base_oct_raw0
        def to_gray(img):
            # None images handled as zeros for overlays
            if img is None:
                if base_oct_raw0 is None: return None
                ref = base_oct_raw0 if base_oct_raw0.ndim == 2 else cv2.cvtColor(base_oct_raw0, cv2.COLOR_RGB2GRAY)
                return np.zeros_like(ref, dtype=np.float32)
            if img.ndim == 2: return img.astype(np.float32)
            # convert rgb to gray image
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # ---------- LEFT CANVAS: Compare ----------
        left_orig_gray = to_gray(base_oct_raw0)  # frame 0
        left_unwarp_gray = to_gray(cur_oct_raw)  # current frame unwarped
        warped_oct, mask_oct = self._warp_with_shared_flow(cur_oct_raw, self.current_idx)
        left_warp_gray = to_gray(warped_oct if warped_oct is not None else cur_oct_raw)

        if left_warp_gray is not None and mask_oct is not None:
            # Mask out invalid regions in warped image
            lw = left_warp_gray.copy()
            lw[~mask_oct] = 0
            left_warp_gray = lw

        # Choose what to display based on checkboxes
        if self.show_original_var.get():
            # Compare warped with original frame 0
            left_rgb = self._to_rgb(left_orig_gray)
        elif self.show_unwarped_var.get():
            # Compare warped with unwarped current frame
            left_rgb = self._to_rgb(left_unwarp_gray)
        else:
            # Show warped by default
            left_rgb = self._to_rgb(left_warp_gray)

        # draw the vectors of the deformation field over the image. optionally show head-tail vectors 
        cumulative_flow = self._get_cumulative_flow(self.current_idx)
        self._draw_on_canvas(self.canvas_left, left_rgb, cumulative_flow,
                             show_headtail=self.show_headtail_var.get(),
                             valid_mask=None)

        # ---------- RIGHT CANVAS: Overlay view ----------
        base_gray = left_warp_gray if left_warp_gray is not None else to_gray(cur_oct_raw)
        if base_gray is None:
            return
        display_rgb = self._to_rgb(base_gray)

        # Create adipose tissue mask from the base OCT image
        adipose_mask = self._create_adipose_mask(
            base_gray,
            threshold_high=self.mask_threshold_high.get(),
            min_area=self.mask_min_area.get()
        )

        # Overlays (fixed alpha) - strictly mask to warped OCT geometry
        cur_str = self._safe_get(self.images["stress"], self.current_idx)
        cur_att = self._safe_get(self.images["att"],    self.current_idx)
        warped_str_rgb, _ = self._warp_with_shared_flow(cur_str, self.current_idx) if cur_str is not None else (None, None)
        warped_att_rgb, _ = self._warp_with_shared_flow(cur_att, self.current_idx) if cur_att is not None else (None, None)

        geom_mask = mask_oct  # available extent is where the warped OCT is valid
        # Combine geometric mask with adipose mask
        if geom_mask is not None and adipose_mask is not None:
            final_mask = geom_mask & ~adipose_mask  # Exclude adipose regions
        elif adipose_mask is not None:
            final_mask = ~adipose_mask
        else:
            final_mask = geom_mask

        # Track active overlays for color bar
        self._current_overlay_type = None
        self._current_overlay_data = None

        if self.overlay_att_var.get() and warped_att_rgb is not None:
            display_rgb = self._alpha_blend_masked(display_rgb, warped_att_rgb, self._alpha_att, final_mask)
            self._current_overlay_type = 'attenuation'
        if self.overlay_str_var.get() and warped_str_rgb is not None:
            display_rgb = self._alpha_blend_masked(display_rgb, warped_str_rgb, self._alpha_str, final_mask)
            self._current_overlay_type = 'stress'

        # --- Preload scalar maps for ratio ---
        a_ref = self._safe_get(self.scalar_images["att"], 0)
        s_ref = self._safe_get(self.scalar_images["stress"], 0)
        a_cur = self._safe_get(self.scalar_images["att"], self.current_idx) if self.current_idx >= 1 else None
        s_cur = self._safe_get(self.scalar_images["stress"], self.current_idx) if self.current_idx >= 1 else None

        # ΔAtt/ΔStr ratio overlay
        if self.overlay_resp_var.get() and self.current_idx >= 1 \
        and all(v is not None for v in (a_ref, s_ref, a_cur, s_cur)):
            a_ref_w, mask_a_ref = self._warp_with_shared_flow(a_ref, 0)
            s_ref_w, mask_s_ref = self._warp_with_shared_flow(s_ref, 0)
            a_cur_w, mask_a_cur = self._warp_with_shared_flow(a_cur, self.current_idx)
            s_cur_w, mask_s_cur = self._warp_with_shared_flow(s_cur, self.current_idx)
            
            if all(v is not None for v in (a_ref_w, s_ref_w, a_cur_w, s_cur_w)):
                # Apply NaN-aware smoothing to attenuation data to reduce speckle noise
                def smooth_with_nan_handling(data, sigma):
                    """Apply Gaussian smoothing while handling NaN values properly"""
                    if sigma <= 0:
                        return data
                    
                    # Create mask for valid (finite) values
                    valid_mask = np.isfinite(data)
                    
                    if not np.any(valid_mask):
                        return data  # Return original if all NaN
                    
                    # Create smoothed version with NaN handling
                    smoothed_data = data.copy()
                    
                    # Fill NaN values with local average of valid neighbors for smoothing
                    temp_data = data.copy()
                    temp_data[~valid_mask] = 0  # Temporarily set NaN to 0
                    
                    # Apply Gaussian filter to both data and weights
                    smoothed_values = ndimage.gaussian_filter(temp_data.astype(float), sigma=sigma)
                    weight_sum = ndimage.gaussian_filter(valid_mask.astype(float), sigma=sigma)
                    
                    # Avoid division by zero
                    weight_sum[weight_sum == 0] = 1
                    
                    # Calculate weighted average (only where we had valid data to smooth)
                    result = smoothed_values / weight_sum
                    
                    # Preserve NaN where we had no valid neighbors to smooth with
                    result[weight_sum < 0.1] = np.nan
                    
                    return result
                
                smoothing_sigma = self.att_smoothing.get()
                a_ref_w_smooth = smooth_with_nan_handling(a_ref_w, smoothing_sigma)
                a_cur_w_smooth = smooth_with_nan_handling(a_cur_w, smoothing_sigma)
                
                # Calculate changes using smoothed attenuation data
                d_att = np.abs(a_cur_w_smooth - a_ref_w_smooth).astype(np.float32)
                d_str = np.abs(s_cur_w - s_ref_w).astype(np.float32)
                
                # Identify valid pixels (including adipose mask)
                eps = 1e-6
                valid = np.isfinite(d_att) & np.isfinite(d_str) & (d_str > eps)
                if final_mask is not None: 
                    valid &= final_mask
                for m in (mask_a_ref, mask_s_ref, mask_a_cur, mask_s_cur):
                    if m is not None: valid &= m
                    
                if np.any(valid):
                    # Create visualization highlighting stable attenuation regions
                    # Red: stable attenuation despite stress change (cancer)
                    # Green: attenuation changes with stress
                    rgba = np.zeros((*d_att.shape, 4), dtype=np.float32)
                    
                    # Normalize changes to [0,1] range for comparison
                    att_norm = d_att / (np.percentile(d_att[valid], 98) + eps)
                    str_norm = d_str / (np.percentile(d_str[valid], 98) + eps)
                    
                    # Stability metric: low att change relative to stress change
                    sensitivity = self.sensitivity.get()  # Get current sensitivity value
                    stability = np.exp(-sensitivity * att_norm / (str_norm + eps))
                    
                    # Use warm-cold colormap for better visualization
                    # Create RdBu_r colormap (red-white-blue reversed)
                    # Blue = cold = unstable/changing (healthy tissue response)
                    # Red = warm = stable (potential cancer)
                    cmap = plt.cm.RdBu_r
                    
                    # Map stability values to warm-cold colors
                    # stability = 0 (unstable/changing) -> blue (cold)
                    # stability = 1 (stable/cancer) -> red (warm)
                    warm_cold_colors = cmap(stability)
                    
                    # Extract RGB channels from warm-cold colormap
                    rgba[..., 0] = warm_cold_colors[..., 0]  # Red
                    rgba[..., 1] = warm_cold_colors[..., 1]  # Green  
                    rgba[..., 2] = warm_cold_colors[..., 2]  # Blue
                    
                    # Alpha channel: universal transparency of 75%
                    rgba[..., 3] = 1
                    
                    # Apply valid mask
                    rgba[~valid] = 0
                    
                    # Convert to uint8 and create overlay
                    rgba_uint8 = (rgba * 255.0 + 0.5).astype(np.uint8)
                    # DeprecationWarning fix: remove 'mode' argument
                    ratio_img = Image.fromarray(rgba_uint8).convert('RGBA')
                    base_rgba = Image.fromarray(display_rgb).convert('RGBA')
                    display_rgb = Image.alpha_composite(base_rgba, ratio_img)
                    display_rgb = np.array(display_rgb.convert('RGB'))
                    
                    # Set overlay type for color bar
                    self._current_overlay_type = 'att_str_ratio'

        self._draw_on_canvas(self.canvas_right, display_rgb, cumulative_flow,
                             show_headtail=self.show_headtail_var.get(),
                             valid_mask=None)

        # ---------- Metrics & scatter ----------
        self.trajectory_s = []; self.trajectory_a = []

        def _means_for_frame(idx):
            if idx == 0:
                oct_img = self._safe_get(self.images["oct"], 0)
                if oct_img is None: return None, None
                oct_gray = oct_img if oct_img.ndim == 2 else cv2.cvtColor(oct_img, cv2.COLOR_RGB2GRAY)
                extent_mask = oct_gray > 0
                s_src = self._safe_get(self.scalar_images["stress"], 0)
                a_src = self._safe_get(self.scalar_images["att"],    0)
                if s_src is None or a_src is None: return None, None
                s_arr = s_src
                a_arr = a_src
                
                # Apply smoothing to attenuation data for graph consistency
                smoothing_sigma = self.att_smoothing.get()
                if smoothing_sigma > 0:
                    # Use the same NaN-aware smoothing as in the overlay
                    valid_mask = np.isfinite(a_arr)
                    if np.any(valid_mask):
                        temp_data = a_arr.copy()
                        temp_data[~valid_mask] = 0
                        smoothed_values = ndimage.gaussian_filter(temp_data.astype(float), sigma=smoothing_sigma)
                        weight_sum = ndimage.gaussian_filter(valid_mask.astype(float), sigma=smoothing_sigma)
                        weight_sum[weight_sum == 0] = 1
                        result = smoothed_values / weight_sum
                        result[weight_sum < 0.1] = np.nan
                        a_arr = result
            else:
                w_oct_i, mask_oct_i = self._warp_with_shared_flow(self._safe_get(self.images["oct"], idx), idx)
                if w_oct_i is None or mask_oct_i is None: return None, None
                s_src = self._safe_get(self.scalar_images["stress"], idx)
                a_src = self._safe_get(self.scalar_images["att"],    idx)
                if s_src is None or a_src is None: return None, None
                s_arr, mask_s = self._warp_with_shared_flow(s_src, idx)
                a_arr, mask_a = self._warp_with_shared_flow(a_src, idx)
                if s_arr is None or a_arr is None: return None, None
                
                # Apply smoothing to attenuation data for graph consistency
                smoothing_sigma = self.att_smoothing.get()
                if smoothing_sigma > 0:
                    # Use the same NaN-aware smoothing as in the overlay
                    valid_mask = np.isfinite(a_arr)
                    if np.any(valid_mask):
                        temp_data = a_arr.copy()
                        temp_data[~valid_mask] = 0
                        smoothed_values = ndimage.gaussian_filter(temp_data.astype(float), sigma=smoothing_sigma)
                        weight_sum = ndimage.gaussian_filter(valid_mask.astype(float), sigma=smoothing_sigma)
                        weight_sum[weight_sum == 0] = 1
                        result = smoothed_values / weight_sum
                        result[weight_sum < 0.1] = np.nan
                        a_arr = result
                
                oct_gray = w_oct_i if w_oct_i.ndim == 2 else cv2.cvtColor(w_oct_i, cv2.COLOR_RGB2GRAY)
                extent_mask = (mask_oct_i) & (oct_gray > 0)
                for m in (mask_s, mask_a):
                    if m is not None: extent_mask &= m
            if not np.any(extent_mask): return None, None
            s_mean = float(np.nanmean(np.where(extent_mask, s_arr, np.nan)))
            a_mean = float(np.nanmean(np.where(extent_mask, a_arr, np.nan)))
            return s_mean, a_mean

        for idx in range(0, self.current_idx + 1):
            s_mean, a_mean = _means_for_frame(idx)
            if s_mean is not None and a_mean is not None:
                self.trajectory_s.append(s_mean)
                self.trajectory_a.append(a_mean)

        # Box trajectories
        box_trajectory_s, box_trajectory_a = [], []
        if self.box_start and self.box_end:
            xa, xb = sorted([int(self.box_start[0]), int(self.box_end[0])])
            ya, yb = sorted([int(self.box_start[1]), int(self.box_end[1])])

            def _box_means_for_frame(idx):
                if idx == 0:
                    oct_img = self._safe_get(self.images["oct"], 0)
                    if oct_img is None: return None, None
                    H, W = oct_img.shape[:2]
                    s_src = self._safe_get(self.scalar_images["stress"], 0)
                    a_src = self._safe_get(self.scalar_images["att"],    0)
                    if s_src is None or a_src is None: return None, None
                    xa_c = np.clip(xa, 0, W - 1); xb_c = np.clip(xb, 0, W - 1)
                    ya_c = np.clip(ya, 0, H - 1); yb_c = np.clip(yb, 0, H - 1)
                    rm = np.zeros((H, W), dtype=bool); rm[ya_c:yb_c+1, xa_c:xb_c+1] = True
                    oct_gray = oct_img if oct_img.ndim == 2 else cv2.cvtColor(oct_img, cv2.COLOR_RGB2GRAY)
                    extent = (oct_gray > 0) & rm
                    if not np.any(extent): return None, None
                    # Add check for valid values before mean
                    s_valid = np.where(extent, s_src, np.nan)
                    a_valid = np.where(extent, a_src, np.nan)
                    if not np.any(np.isfinite(s_valid)) or not np.any(np.isfinite(a_valid)):
                        return None, None
                    return (float(np.nanmean(s_valid)), float(np.nanmean(a_valid)))
                else:
                    w_oct_i, mask_oct_i = self._warp_with_shared_flow(self._safe_get(self.images["oct"], idx), idx)
                    if w_oct_i is None or mask_oct_i is None: return None, None
                    s_src = self._safe_get(self.scalar_images["stress"], idx)
                    a_src = self._safe_get(self.scalar_images["att"],    idx)
                    if s_src is None or a_src is None: return None, None
                    s_warp, mask_s = self._warp_with_shared_flow(s_src, idx)
                    a_warp, mask_a = self._warp_with_shared_flow(a_src, idx)
                    if s_warp is None or a_warp is None: return None, None
                    H, W = s_warp.shape[:2]
                    xa_c = np.clip(xa, 0, W - 1); xb_c = np.clip(xb, 0, W - 1)
                    ya_c = np.clip(ya, 0, H - 1); yb_c = np.clip(yb, 0, H - 1)
                    rm = np.zeros((H, W), dtype=bool); rm[ya_c:yb_c+1, xa_c:xb_c+1] = True
                    oct_gray = w_oct_i if w_oct_i.ndim == 2 else cv2.cvtColor(w_oct_i, cv2.COLOR_RGB2GRAY)
                    cmask = rm & mask_oct_i & (oct_gray > 0)
                    for m in (mask_s, mask_a):
                        if m is not None: cmask &= m
                    if not np.any(cmask): return None, None
                    # Add check for valid values before mean
                    s_valid = np.where(cmask, s_warp, np.nan)
                    a_valid = np.where(cmask, a_warp, np.nan)
                    if not np.any(np.isfinite(s_valid)) or not np.any(np.isfinite(a_valid)):
                        return None, None
                    return (float(np.nanmean(s_valid)), float(np.nanmean(a_valid)))

            for idx in range(0, self.current_idx + 1):
                bs, ba = _box_means_for_frame(idx)
                if bs is not None and ba is not None:
                    box_trajectory_s.append(bs)
                    box_trajectory_a.append(ba)

        # ── Scatter refresh
        self.ax.clear()
        if self.trajectory_s and self.trajectory_a:
            # Plot all points in the trajectory
            self.ax.plot(self.trajectory_s, self.trajectory_a, "-o", alpha=0.85, label="Image Average", color="#0072b2", markersize=6)
            # Highlight current frame if we have data
            if self.current_idx < len(self.trajectory_s):
                current_s = self.trajectory_s[self.current_idx]
                current_a = self.trajectory_a[self.current_idx]
                self.ax.plot(current_s, current_a, "o", color="red", markersize=8, markerfacecolor="red", markeredgecolor="darkred", markeredgewidth=2, label="Current Frame")
        
        if box_trajectory_s and box_trajectory_a:
            # Plot all points in the box trajectory
            self.ax.plot(box_trajectory_s, box_trajectory_a, "-s", alpha=0.9, label="Box", color="#d55e00", markersize=7)
            # Highlight current frame if we have data
            if self.current_idx < len(box_trajectory_s):
                current_box_s = box_trajectory_s[self.current_idx]
                current_box_a = box_trajectory_a[self.current_idx]
                self.ax.plot(current_box_s, current_box_a, "s", color="red", markersize=8, markerfacecolor="red", markeredgecolor="darkred", markeredgewidth=2)
        
        self.ax.set_xlabel("Mean Stress (log₁₀ kPa)", fontsize=11)
        self.ax.set_ylabel("Mean Attenuation (mm⁻¹)", fontsize=11)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_facecolor("#f7f7f7")
        self.ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)

        # Ensure the graph fits within the canvas at all times
        canvas_widget = self.scatter_widget
        canvas_widget.update_idletasks()
        w = canvas_widget.winfo_width()
        h = canvas_widget.winfo_height()
        if w > 0 and h > 0:
            dpi = self.fig.get_dpi()
            self.fig.set_size_inches(w / dpi, h / dpi, forward=True)

        # Lock axes to current data range to maintain square aspect
        s_all, a_all = [], []
        if self.trajectory_s and self.trajectory_a:
            s_all.extend(self.trajectory_s); a_all.extend(self.trajectory_a)
        if box_trajectory_s and box_trajectory_a:
            s_all.extend(box_trajectory_s); a_all.extend(box_trajectory_a)

        if s_all and a_all:
            s_all = np.asarray(s_all, dtype=float)
            a_all = np.asarray(a_all, dtype=float)
            mask = np.isfinite(s_all) & np.isfinite(a_all)
            if np.any(mask):
                s_all = s_all[mask]; a_all = a_all[mask]
                s_min, s_max = float(np.min(s_all)), float(np.max(s_all))
                a_min, a_max = float(np.min(a_all)), float(np.max(a_all))
                
                # Add padding
                s_range = s_max - s_min if s_max > s_min else 0.1
                a_range = a_max - a_min if a_max > a_min else 0.1
                s_pad = s_range * 0.05
                a_pad = a_range * 0.05
                
                # Make square by using the larger range for both axes
                max_range = max(s_range + 2*s_pad, a_range + 2*a_pad)
                s_center = (s_min + s_max) / 2
                a_center = (a_min + a_max) / 2
                
                self.ax.set_xlim(s_center - max_range/2, s_center + max_range/2)
                self.ax.set_ylim(a_center - max_range/2, a_center + max_range/2)
            else:
                # Default square ranges when no data
                self.ax.set_xlim(0, 2)
                self.ax.set_ylim(0, 2)
        else:
            # Default square ranges when no data
            self.ax.set_xlim(0, 2)
            self.ax.set_ylim(0, 2)
        
        self.ax.set_aspect('equal', adjustable='box')

        self.scatter_canvas.draw()

        # Hover cursor on lines - remove frame labeling
        if self._cursor:
            try: self._cursor.remove()
            except Exception: pass
        artists = [*self.ax.lines]
        if artists:
            self._cursor = mplcursors.cursor(artists, hover=True)
            @self._cursor.connect("add")
            def on_add(sel):
                x, y = sel.target
                stress_kpa = -(10**x)  # Convert back to kPa
                sel.annotation.set_text(
                    f"Stress: {stress_kpa:.1f} kPa\nAtten: {y:.1f} mm⁻¹"
                )

    # ------------------------------ canvas & events -----------------------------
    def _draw_on_canvas(self, canvas, img, flow, forward=True, show_headtail=False, valid_mask=None):
        if img is None:
            return
        if img.ndim == 2: disp = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else: disp = self._to_rgb(img)
        if valid_mask is not None:
            mask3 = np.repeat(valid_mask[..., None], 3, axis=2)
            disp = disp.copy(); disp[~mask3] = 0

        pil = Image.fromarray(disp)
        c_w = canvas.winfo_width() or pil.width
        c_h = canvas.winfo_height() or pil.height
        scale = min(c_w / pil.width, c_h / pil.height)
        nw, nh = int(pil.width * scale), int(pil.height * scale)
        resized = pil.resize((nw, nh), Image.BILINEAR)
        photo = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image((c_w - nw)//2, (c_h - nh)//2, anchor="nw", image=photo)
        canvas.image = photo

        offset_x = (c_w - nw) / 2; offset_y = (c_h - nh) / 2
        step = self.quiver_step
        base_color = np.array([30, 120, 200])

        H = flow.shape[0] if flow is not None else 0
        W = flow.shape[1] if flow is not None else 0

        for y in range(0, H, step):
            for x in range(0, W, step):
                point = np.array([x, y], dtype=np.float32)
                traj_points = [point.copy()]

                for idx in range(1, self.current_idx + 1):
                    flow_inc = self.incremental_flows.get(idx)
                    if flow_inc is None: break
                    xi = int(np.clip(point[0], 0, W - 1))
                    yi = int(np.clip(point[1], 0, H - 1))
                    delta = flow_inc[yi, xi]
                    point = point + delta
                    traj_points.append(point.copy())

                if show_headtail:
                    for i in range(len(traj_points) - 1):
                        p0 = traj_points[i]; p1 = traj_points[i+1]
                        x1 = p0[0] * scale + offset_x; y1 = p0[1] * scale + offset_y
                        x2 = p1[0] * scale + offset_x; y2 = p1[1] * scale + offset_y
                        age = self.current_idx - (i+1)
                        alpha = max(0.1, 1.0 - age * 0.1)
                        fade = (base_color * alpha + 255 * (1 - alpha)).astype(int)
                        col = f"#{fade[0]:02x}{fade[1]:02x}{fade[2]:02x}"
                        canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill=col, width=0.25)

                if self.show_net_var.get():
                    p_start = traj_points[0]; p_end = traj_points[-1]
                    x1_net = p_start[0] * scale + offset_x; y1_net = p_start[1] * scale + offset_y
                    x2_net = p_end[0] * scale + offset_x; y2_net = p_end[1] * scale + offset_y
                    canvas.create_line(x1_net, y1_net, x2_net, y2_net, arrow=tk.LAST, fill='lime', width=0.25)

        if self.box_start and self.box_end:
            x0, y0 = self.box_start; x1, y1 = self.box_end
            x0c = x0 * scale + offset_x; y0c = y0 * scale + offset_y
            x1c = x1 * scale + offset_x; y1c = y1 * scale + offset_y
            canvas.create_rectangle(x0c, y0c, x1c, y1c, outline='yellow', width=2)
        
        # Add scale bar and color bar if applicable
        self._draw_scale_bar(canvas, scale, offset_x, offset_y, c_w, c_h)
        # Draw appropriate color bar for each canvas
        if canvas == self.canvas_right:
            self._draw_color_bar(canvas, c_w, c_h)
        elif canvas == self.canvas_left:
            self._draw_grayscale_color_bar(canvas, c_w, c_h)
        
        # Add title for left canvas
        canvas_title = self._get_canvas_title(canvas)
        if canvas_title:
            canvas.create_text(10, 10, text=canvas_title, 
                              fill='white', font=('Arial', 10, 'bold'), anchor='nw')

    def _on_box_start(self, event):
        self._active_canvas = event.widget
        self.box_start = self._canvas_to_image(event.widget, event.x, event.y)
        self.box_end = self.box_start
        self._refresh_display()

    def _on_box_drag(self, event):
        self._active_canvas = event.widget
        self.box_end = self._canvas_to_image(event.widget, event.x, event.y)
        self._refresh_display()

    def _on_box_end(self, event):
        self._active_canvas = event.widget
        self.box_end = self._canvas_to_image(event.widget, event.x, event.y)
        self._refresh_display()

    def _canvas_to_image(self, canvas, cx, cy):
        flow = self._get_cumulative_flow(self.current_idx)
        if flow is None:
            return (cx, cy)
        H, W = flow.shape[:2]
        c_w = canvas.winfo_width() or W
        c_h = canvas.winfo_height() or H
        scale = min(c_w / W, c_h / H)
        offset_x = (c_w - int(W * scale)) / 2
        offset_y = (c_h - int(H * scale)) / 2
        ix = (cx - offset_x) / scale
        iy = (cy - offset_y) / scale
        return (np.clip(ix, 0, W-1), np.clip(iy, 0, H-1))

    # ------------------------------ simple controls -----------------------------
    def _toggle_compare(self):
        self.compare_show = 'orig' if self.compare_show == 'warped' else 'warped'
        self.show_original_oct_var.set(self.compare_show == 'orig')
        self._refresh_display()

    def _clear_box(self):
        self.box_start = None
        self.box_end   = None
        self._refresh_display()

    def _show_export_dialog(self):
        if not self.sequence_root or not self.images["oct"]:
            messagebox.showwarning("No Data", "Please load a sequence first.")
            return
            
        # Create export dialog
        export_dialog = tk.Toplevel(self)
        export_dialog.title("Export Data")
        export_dialog.transient(self.winfo_toplevel())
        export_dialog.grab_set()
        export_dialog.resizable(False, False)
        
        # Center the dialog
        export_dialog.update_idletasks()
        px = self.winfo_rootx() + self.winfo_width()//2 - 200
        py = self.winfo_rooty() + self.winfo_height()//2 - 150
        export_dialog.geometry(f"400x300+{px}+{py}")
        
        # Main frame
        main_frame = ttk.Frame(export_dialog, padding=(20, 20))
        main_frame.pack(fill='both', expand=True)
        
        ttk.Label(main_frame, text="Select data to export:", font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # Export options
        self.export_warped_oct = tk.BooleanVar(value=True)
        self.export_deformation = tk.BooleanVar(value=True)
        self.export_overlay = tk.BooleanVar(value=True)
        self.export_all = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(main_frame, text="Warped OCT Arrays (.mat)", 
                       variable=self.export_warped_oct).pack(anchor='w', pady=2)
        ttk.Checkbutton(main_frame, text="Deformation Fields (.mat)", 
                       variable=self.export_deformation).pack(anchor='w', pady=2)
        ttk.Checkbutton(main_frame, text="Attenuation/Stress Overlay Images (.png)", 
                       variable=self.export_overlay).pack(anchor='w', pady=2)
        
        ttk.Separator(main_frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Checkbutton(main_frame, text="Export All", 
                       variable=self.export_all, 
                       command=lambda: self._toggle_export_all()).pack(anchor='w', pady=2)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(20, 0))
        
        ttk.Button(button_frame, text="Cancel", 
                  command=export_dialog.destroy).pack(side='right', padx=(5, 0))
        ttk.Button(button_frame, text="Export", 
                  command=lambda: self._start_export(export_dialog)).pack(side='right')
    
    def _toggle_export_all(self):
        if self.export_all.get():
            self.export_warped_oct.set(True)
            self.export_deformation.set(True)
            self.export_overlay.set(True)
        
    def _start_export(self, dialog):
        # Check if at least one option is selected
        if not any([self.export_warped_oct.get(), self.export_deformation.get(), self.export_overlay.get()]):
            messagebox.showwarning("No Selection", "Please select at least one export option.")
            return
            
        # Select export folder
        export_folder = filedialog.askdirectory(title="Select Export Folder")
        if not export_folder:
            return
            
        dialog.destroy()
        
        # Show progress dialog
        progress_dialog = tk.Toplevel(self)
        progress_dialog.title("Exporting...")
        progress_dialog.transient(self.winfo_toplevel())
        progress_dialog.grab_set()
        progress_dialog.resizable(False, False)
        
        progress_frame = ttk.Frame(progress_dialog, padding=(20, 20))
        progress_frame.pack(fill='both', expand=True)
        
        status_label = ttk.Label(progress_frame, text="Preparing export...")
        status_label.pack(pady=(0, 10))
        
        progress_bar = ttk.Progressbar(progress_frame, mode='determinate', length=300)
        progress_bar.pack()
        
        # Center progress dialog
        progress_dialog.update_idletasks()
        px = self.winfo_rootx() + self.winfo_width()//2 - 170
        py = self.winfo_rooty() + self.winfo_height()//2 - 50
        progress_dialog.geometry(f"340x100+{px}+{py}")
        
        # Start export in separate thread-like manner using after()
        self._export_data(export_folder, status_label, progress_bar, progress_dialog)
    
    def _export_data(self, export_folder, status_label, progress_bar, progress_dialog):
        try:
            total_steps = 0
            current_step = 0
            
            # Count total steps
            if self.export_warped_oct.get():
                total_steps += len(self.images["oct"])
            if self.export_deformation.get():
                total_steps += len(self.images["oct"]) - 1  # No flow for first frame
            if self.export_overlay.get():
                total_steps += len(self.images["oct"]) - 1  # No overlay for first frame
                
            progress_bar['maximum'] = total_steps
            
            # Export warped OCT arrays
            if self.export_warped_oct.get():
                status_label.config(text="Exporting warped OCT arrays...")
                self.update_idletasks()
                
                warped_arrays = []
                for idx in range(len(self.images["oct"])):
                    if idx == 0:
                        warped_arrays.append(self.images["oct"][0])
                    else:
                        warped_oct, _ = self._warp_with_shared_flow(self.images["oct"][idx], idx)
                        warped_arrays.append(warped_oct if warped_oct is not None else self.images["oct"][idx])
                    
                    current_step += 1
                    progress_bar['value'] = current_step
                    self.update_idletasks()
                
                # Save as .mat file
                warped_stack = np.stack(warped_arrays, axis=2)
                sio.savemat(os.path.join(export_folder, 'warped_oct_arrays.mat'), 
                           {'warped_oct': warped_stack})
            
            # Export deformation fields
            if self.export_deformation.get():
                status_label.config(text="Exporting deformation fields...")
                self.update_idletasks()
                
                flow_arrays = []
                for idx in range(1, len(self.images["oct"])):
                    flow = self._get_cumulative_flow(idx)
                    if flow is not None:
                        flow_arrays.append(flow)
                    
                    current_step += 1
                    progress_bar['value'] = current_step
                    self.update_idletasks()
                
                if flow_arrays:
                    flow_stack = np.stack(flow_arrays, axis=2)
                    sio.savemat(os.path.join(export_folder, 'deformation_fields.mat'), 
                               {'flows': flow_stack})
            
            # Export overlay images
            if self.export_overlay.get():
                status_label.config(text="Exporting overlay images...")
                self.update_idletasks()
                
                overlay_folder = os.path.join(export_folder, 'overlay_images')
                os.makedirs(overlay_folder, exist_ok=True)
                
                for idx in range(1, len(self.images["oct"])):
                    overlay_img = self._generate_overlay_image(idx)
                    if overlay_img is not None:
                        filename = f'overlay_frame_{idx:03d}.png'
                        Image.fromarray(overlay_img).save(os.path.join(overlay_folder, filename))
                    
                    current_step += 1
                    progress_bar['value'] = current_step
                    self.update_idletasks()
            
            progress_dialog.destroy()
            messagebox.showinfo("Export Complete", f"Data exported successfully to:\n{export_folder}")
            
        except Exception as e:
            progress_dialog.destroy()
            messagebox.showerror("Export Error", f"An error occurred during export:\n{str(e)}")
    
    def _generate_overlay_image(self, frame_idx):
        """Generate the overlay image for a specific frame"""
        try:
            # Get warped OCT as base
            warped_oct, mask_oct = self._warp_with_shared_flow(self.images["oct"][frame_idx], frame_idx)
            if warped_oct is None:
                return None
                
            base_gray = warped_oct if warped_oct.ndim == 2 else cv2.cvtColor(warped_oct, cv2.COLOR_RGB2GRAY)
            display_rgb = self._to_rgb(base_gray)
            
            # Create adipose mask
            adipose_mask = self._create_adipose_mask(
                base_gray,
                threshold_high=self.mask_threshold_high.get(),
                min_area=self.mask_min_area.get()
            )
            
            # Combine masks
            geom_mask = mask_oct
            if geom_mask is not None and adipose_mask is not None:
                final_mask = geom_mask & ~adipose_mask
            elif adipose_mask is not None:
                final_mask = ~adipose_mask
            else:
                final_mask = geom_mask
                
            # Generate overlay if we have the required data
            a_ref = self._safe_get(self.scalar_images["att"], 0)
            s_ref = self._safe_get(self.scalar_images["stress"], 0)
            a_cur = self._safe_get(self.scalar_images["att"], frame_idx)
            s_cur = self._safe_get(self.scalar_images["stress"], frame_idx)
            
            if all(v is not None for v in (a_ref, s_ref, a_cur, s_cur)):
                a_ref_w, mask_a_ref = self._warp_with_shared_flow(a_ref, 0)
                s_ref_w, mask_s_ref = self._warp_with_shared_flow(s_ref, 0)
                a_cur_w, mask_a_cur = self._warp_with_shared_flow(a_cur, frame_idx)
                s_cur_w, mask_s_cur = self._warp_with_shared_flow(s_cur, frame_idx)
                
                if all(v is not None for v in (a_ref_w, s_ref_w, a_cur_w, s_cur_w)):
                    # Apply NaN-aware smoothing to attenuation data to reduce speckle noise
                    def smooth_with_nan_handling(data, sigma):
                        """Apply Gaussian smoothing while handling NaN values properly"""
                        if sigma <= 0:
                            return data
                        
                        # Create mask for valid (finite) values
                        valid_mask = np.isfinite(data)
                        
                        if not np.any(valid_mask):
                            return data  # Return original if all NaN
                        
                        # Create smoothed version with NaN handling
                        smoothed_data = data.copy()
                        
                        # Fill NaN values with local average of valid neighbors for smoothing
                        temp_data = data.copy()
                        temp_data[~valid_mask] = 0  # Temporarily set NaN to 0
                        
                        # Apply Gaussian filter to both data and weights
                        smoothed_values = ndimage.gaussian_filter(temp_data.astype(float), sigma=sigma)
                        weight_sum = ndimage.gaussian_filter(valid_mask.astype(float), sigma=sigma)
                        
                        # Avoid division by zero
                        weight_sum[weight_sum == 0] = 1
                        
                        # Calculate weighted average (only where we had valid data to smooth)
                        result = smoothed_values / weight_sum
                        
                        # Preserve NaN where we had no valid neighbors to smooth with
                        result[weight_sum < 0.1] = np.nan
                        
                        return result
                    
                    smoothing_sigma = self.att_smoothing.get()
                    a_ref_w_smooth = smooth_with_nan_handling(a_ref_w, smoothing_sigma)
                    a_cur_w_smooth = smooth_with_nan_handling(a_cur_w, smoothing_sigma)
                    
                    # Calculate changes using smoothed attenuation data
                    d_att = np.abs(a_cur_w_smooth - a_ref_w_smooth).astype(np.float32)
                    d_str = np.abs(s_cur_w - s_ref_w).astype(np.float32)
                    
                    # Create overlay
                    eps = 1e-6
                    valid = np.isfinite(d_att) & np.isfinite(d_str) & (d_str > eps)
                    if final_mask is not None: 
                        valid &= final_mask
                    for m in (mask_a_ref, mask_s_ref, mask_a_cur, mask_s_cur):
                        if m is not None: valid &= m
                        
                    if np.any(valid):
                        rgba = np.zeros((*d_att.shape, 4), dtype=np.float32)
                        
                        att_norm = d_att / (np.percentile(d_att[valid], 98) + eps)
                        str_norm = d_str / (np.percentile(d_str[valid], 98) + eps)
                        
                        sensitivity = self.sensitivity.get()
                        stability = np.exp(-sensitivity * att_norm / (str_norm + eps))
                        
                        # Use warm-cold colormap for better visualization
                        # Create RdBu_r colormap (red-white-blue reversed)
                        # Blue = cold = unstable/changing (healthy tissue response)
                        # Red = warm = stable (potential cancer)
                        cmap = plt.cm.RdBu_r
                        
                        # Map stability values to warm-cold colors
                        # stability = 0 (unstable/changing) -> blue (cold)
                        # stability = 1 (stable/cancer) -> red (warm)
                        warm_cold_colors = cmap(stability)
                        
                        # Extract RGB channels from warm-cold colormap
                        rgba[..., 0] = warm_cold_colors[..., 0]  # Red
                        rgba[..., 1] = warm_cold_colors[..., 1]  # Green  
                        rgba[..., 2] = warm_cold_colors[..., 2]  # Blue
                        rgba[..., 3] = 1            # Full opacity
                        
                        rgba[~valid] = 0
                        
                        # Apply blur
                        for channel in range(4):
                            rgba[..., channel] = ndimage.gaussian_filter(rgba[..., channel], sigma=1.5)
                        
                        # Composite with base image
                        rgba_uint8 = (rgba * 255.0 + 0.5).astype(np.uint8)
                        ratio_img = Image.fromarray(rgba_uint8).convert('RGBA')
                        base_rgba = Image.fromarray(display_rgb).convert('RGBA')
                        display_rgb = Image.alpha_composite(base_rgba, ratio_img)
                        display_rgb = np.array(display_rgb.convert('RGB'))
            
            return display_rgb
            
        except Exception as e:
            print(f"Error generating overlay for frame {frame_idx}: {e}")
            return None

    # ------------------------------ sizing logic --------------------------------
    def _on_tools_resize(self, event):
        """Keep scatter square and match height of tools column."""
        h_px = max(50, self.tools_frame.winfo_width())
        dpi = self.fig.get_dpi()
        side_in = h_px / dpi
        side_in = max(2.5, side_in)
        self.fig.set_size_inches(side_in, side_in, forward=True)
        self.scatter_canvas.draw_idle()

# ------------------------------- app shell ------------------------------------
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OCT Deformation Tracking Suite")
        self.geometry("1280x900")
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('TNotebook.Tab', padding=(10, 10))

        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True)
        pages = [ (StressAttenuationExplorer, "Relationship Explorer") ]
        for PageClass, title in pages:
            page = PageClass(notebook)
            notebook.add(page, text=title)

if __name__ == "__main__":
    MainApp().mainloop()