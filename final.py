# ---- Overlay Attenuation Response Range ----
# Constants removed - using dynamic ranges now
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

def block_average(data, size=3):
    # Use uniform_filter for block mean, preserve NaN as much as possible
    valid_mask = np.isfinite(data)
    data_filled = np.where(valid_mask, data, 0)
    count = ndimage.uniform_filter(valid_mask.astype(float), size=size, mode='nearest')
    summed = ndimage.uniform_filter(data_filled, size=size, mode='nearest')
    with np.errstate(invalid='ignore'):
        avg = summed / count
    avg[count < 1e-3] = np.nan
    return avg

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
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # Min/max filters removed - using dynamic scaling

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
        # Unused variables removed for cleaner code


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

        # Navigation
        nav = ttk.Frame(tools_frame)
        nav.pack(fill='x', pady=8)
        ttk.Button(nav, text="<< Prev", command=lambda: self._advance(-1)).pack(side='left')
        ttk.Button(nav, text="Next >>", command=lambda: self._advance(1)).pack(side='left', padx=6)
        ttk.Button(nav, text="Reset Boxes", command=self._clear_box).pack(side='left', padx=6)
        ttk.Button(nav, text="Show Histogram", command=self._show_histogram).pack(side='left', padx=6)
        self.index_label = ttk.Label(nav, text="Frame: N/A")
        self.index_label.pack(side='left', padx=8)

        # Scatter graph
        graph_box = ttk.Labelframe(tools_frame, text="Attenuation Response to Stress", padding=6)
        graph_box.pack(fill='both', expand=True, pady=(4,0))
        self.fig, self.ax = plt.subplots(figsize=(4,4.3))
        self.ax.set_aspect('equal', adjustable='box')
        self.scatter_canvas = FigureCanvasTkAgg(self.fig, master=graph_box)
        self.scatter_widget = self.scatter_canvas.get_tk_widget()
        self.scatter_widget.pack(anchor='n', fill='none')
        self.ax.set_xlabel("Mean Stress (kPa)", fontsize=10)
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

        # State
        self.sequence_root = None
        self.images = {"oct": [], "stress": [], "att": []}
        self.scalar_images = {"stress": [], "att": []}
        self.current_idx = 0
        self.cumulative = {"shared_flow": {}}
        self.incremental_flows = {}
        self.quiver_step = 20
        self.trajectory_s = []; self.trajectory_a = []
        self.boxes = []  # list of (start, end, color)
        self.colors = ['blue', 'lime', 'red', 'magenta']
        self.current_box_idx = 0
        self._cursor = None
        self._active_canvas = None

        # Fixed overlay opacities
        self._alpha_att   = 1
        self._alpha_str   = 1
        
        # Scale and colorbar settings
        self.pixels_per_mm = 364 / 6.0  # 364 pixels = 6mm
        self._current_overlay_type = None
        self._current_overlay_data = None

        # Epsilon for numerical stability
        self.eps = 0

    # ------------------------------- UI helpers --------------------------------
    # _reset_boxes method removed - functionality moved to _clear_box

    def _safe_get(self, seq, idx):
        return seq[idx] if 0 <= idx < len(seq) else None

    def _box_means_for_frame(self, idx, xa, xb, ya, yb):
        if idx == 0:
            oct_img = self._safe_get(self.images["oct"], 0)
            if oct_img is None: return None, None
            H, W = oct_img.shape[:2]
            s_src = self._safe_get(self.scalar_images["stress"], 0)
            a_src = self._safe_get(self.scalar_images["att"],    0)
            if s_src is None or a_src is None: return None, None
            a_src = block_average(a_src)
            xa_c = np.clip(xa, 0, W - 1); xb_c = np.clip(xb, 0, W - 1)
            ya_c = np.clip(ya, 0, H - 1); yb_c = np.clip(yb, 0, H - 1)
            rm = np.zeros((H, W), dtype=bool); rm[ya_c:yb_c+1, xa_c:xb_c+1] = True
            oct_gray = oct_img if oct_img.ndim == 2 else cv2.cvtColor(oct_img, cv2.COLOR_RGB2GRAY)
            extent_mask = (oct_gray > 0) & rm
            if not np.any(extent_mask): return None, None
            # Add check for valid values before mean
            s_valid = np.where(extent_mask, s_src, np.nan)
            a_valid = np.where(extent_mask, a_src, np.nan)
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
            a_warp = block_average(a_warp)
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
            self.boxes = []
            self.current_box_idx = 0
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

    def _on_overlay_toggle(self):
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
                # Ensure stress is positive
                stress_kpa = np.abs(slice_data)
                # Clamp stress values to 0-100 kPa
                stress_kpa = np.clip(stress_kpa, 0, 100)
                self.scalar_images["stress"].append(stress_kpa)
                # Convert to RGB visualization using jet colormap for 0-100 kPa
                rgb = self._apply_colormap(stress_kpa, 'jet', 0, 100)
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
            
        # Horizontal color bar at bottom left
        cb_length = int(canvas_width * 0.4)
        cb_height = 20
        margin = 20
        x1 = margin
        y1 = canvas_height - margin - cb_height
        
        # Create color bar based on overlay type
        if self._current_overlay_type == 'attenuation':
            # Viridis colormap for attenuation (0-10 mm⁻¹)
            cmap = plt.cm.viridis
            vmin, vmax = 0, 10
            unit = 'mm⁻¹'
            label = 'Attenuation (mm⁻¹)'
        elif self._current_overlay_type == 'stress':
            # Jet colormap for stress (log scale display, kPa values)
            cmap = plt.cm.jet
            if hasattr(self, '_stress_data') and self._stress_data is not None:
                valid_data = self._stress_data[self._stress_data > 0]
            vmin, vmax = 0, 2
            unit = 'kPa'
            label = 'Stress (kPa) [Log Scale]'
        elif self._current_overlay_type == 'att_str_ratio':
            # Plasma colormap for attenuation response (2nd-98th percentile linear scale)
            cmap = plt.cm.jet
            if hasattr(self, '_response_data') and self._response_data is not None:
                valid_data = self._response_data[np.isfinite(self._response_data)]
                if len(valid_data) > 0:
                    vmin = np.percentile(valid_data, 3)
                    vmax = np.percentile(valid_data, 97)
                else:
                    vmin, vmax = -1, 3
            else:
                vmin, vmax = -1, 3
            unit = ''
            label = 'Attenuation Load Response (mm⁻¹/kPa) [3-97%]'
        else:
            return
        
        # Draw color bar background
        canvas.create_rectangle(x1-2, y1-2, x1+cb_length+2, y1+cb_height+2, fill='white', outline='black', width=1)
        # Create horizontal color gradient
        for i in range(cb_length):
            norm_pos = i / (cb_length - 1)
            color_rgba = cmap(norm_pos)
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(color_rgba[0] * 255),
                int(color_rgba[1] * 255),
                int(color_rgba[2] * 255)
            )
            canvas.create_line(x1 + i, y1, x1 + i, y1 + cb_height, fill=color_hex, width=1)
        
        # Add tick marks and labels (horizontal, above color bar)
        num_ticks = 5
        for i in range(num_ticks):
            tick_pos = i / (num_ticks - 1)
            x_pos = x1 + int(tick_pos * cb_length)
            value = vmin + tick_pos * (vmax - vmin)
            # Draw tick mark above color bar
            canvas.create_line(x_pos, y1 - 8, x_pos, y1, fill='white', width=2)
            # Draw value label above tick
            value_text = f'{value:.1f}' if self._current_overlay_type == 'att_str_ratio' else (f'{value:.0f}' if value == int(value) else f'{value:.1f}')
            canvas.create_text(x_pos, y1 - 18, text=value_text, fill='white', font=('Arial', 9, 'bold'), anchor='s')
        # Add title at left of color bar, move up to avoid overlap
        if unit:
            title_text = f'{label} ({unit})'
        else:
            title_text = label
        canvas.create_text(x1, y1 - 38, text=title_text, fill='white', font=('Arial', 10, 'bold'), anchor='sw')

    def _draw_grayscale_color_bar(self, canvas, canvas_width, canvas_height):
        """Draw a grayscale color bar for OCT intensity values"""
        # Horizontal grayscale color bar at bottom left
        cb_length = int(canvas_width * 0.4)
        cb_height = 20
        margin = 20
        x1 = margin
        y1 = canvas_height - margin - cb_height
        canvas.create_rectangle(x1-2, y1-2, x1+cb_length+2, y1+cb_height+2, fill='white', outline='black', width=1)
        for i in range(cb_length):
            norm_pos = i / (cb_length - 1)
            oct_value = norm_pos * 29
            gray_value = int((oct_value / 29) * 255)
            color_hex = '#{:02x}{:02x}{:02x}'.format(gray_value, gray_value, gray_value)
            canvas.create_line(x1 + i, y1, x1 + i, y1 + cb_height, fill=color_hex, width=1)
        num_ticks = 5
        vmin, vmax = 0, 29
        for i in range(num_ticks):
            tick_pos = i / (num_ticks - 1)
            x_pos = x1 + int(tick_pos * cb_length)
            value = vmin + tick_pos * (vmax - vmin)
            canvas.create_line(x_pos, y1 - 8, x_pos, y1, fill='white', width=2)
            value_text = f'{value:.0f}' if value == int(value) else f'{value:.1f}'
            canvas.create_text(x_pos, y1 - 18, text=value_text, fill='white', font=('Arial', 9, 'bold'), anchor='s')
        canvas.create_text(x1, y1 - 38, text='OCT Intensity (0-29)', fill='white', font=('Arial', 10, 'bold'), anchor='sw')

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
            threshold_high=5.5,
            min_area=24
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
        if self.overlay_str_var.get():
            # Apply stress overlay with log scaling using scalar data
            s_cur_scalar = self._safe_get(self.scalar_images["stress"], self.current_idx)
            if s_cur_scalar is not None:
                s_cur_warped, mask_s = self._warp_with_shared_flow(s_cur_scalar, self.current_idx)
                if s_cur_warped is not None:
                    self._stress_data = s_cur_warped  # Store for colorbar
                    
                    # Apply log scaling for visualization (kPa values for calculation)
                    stress_log = np.log10(s_cur_warped + 1e-6)
                    stress_log_clipped = np.clip(stress_log, 0, 2)  # log10(1) to log10(100)
                    
                    # Create RGB using log-scaled values
                    rgb_stress = self._apply_colormap(stress_log_clipped, 'jet', 0, 2)
                    rgba_stress = np.dstack([rgb_stress, np.full(rgb_stress.shape[:2], int(255 * self._alpha_str))])
                    stress_img = Image.fromarray(rgba_stress.astype(np.uint8), mode='RGBA')
                    
                    # Apply mask and blend
                    if final_mask is not None:
                        # Set masked areas to transparent
                        alpha_channel = rgba_stress[..., 3].copy()
                        alpha_channel[~final_mask] = 0
                        rgba_stress[..., 3] = alpha_channel
                        stress_img = Image.fromarray(rgba_stress.astype(np.uint8), mode='RGBA')
                    
                    base_rgba = Image.fromarray(display_rgb).convert('RGBA')
                    display_rgb = Image.alpha_composite(base_rgba, stress_img)
                    display_rgb = np.array(display_rgb.convert('RGB'))
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
                # Get warped OCT for current frame to build extent_mask
                w_oct_i, mask_oct_i = self._warp_with_shared_flow(self._safe_get(self.images["oct"], self.current_idx), self.current_idx)
                
                # --- 3x3 block averaging for attenuation (no further smoothing) ---
                a_ref_w_block = block_average(a_ref_w)
                a_cur_w_block = block_average(a_cur_w)
                d_att = (a_cur_w_block - a_ref_w_block).astype(np.float32)
                d_str = (s_cur_w - s_ref_w).astype(np.float32)

                # Calculate response r = ΔAtt/ΔStr
                r_current = np.where(d_str > self.eps, d_att / d_str, np.nan)

                # Filter for aggressive fluctuations: check change from previous frame
                fluctuation_mask = np.ones_like(d_att, dtype=bool)
                if self.current_idx >= 2:
                    # Get previous frame data
                    a_prev = self._safe_get(self.scalar_images["att"], self.current_idx - 1)
                    s_prev = self._safe_get(self.scalar_images["stress"], self.current_idx - 1)
                    if a_prev is not None and s_prev is not None:
                        a_prev_w, _ = self._warp_with_shared_flow(a_prev, self.current_idx - 1)
                        s_prev_w, _ = self._warp_with_shared_flow(s_prev, self.current_idx - 1)
                        if a_prev_w is not None and s_prev_w is not None:
                            a_prev_w_block = block_average(a_prev_w)
                            d_att_prev = (a_prev_w_block - a_ref_w_block).astype(np.float32)
                            d_str_prev = (s_prev_w - s_ref_w).astype(np.float32)
                            r_prev = np.where(d_str_prev > self.eps, d_att_prev / d_str_prev, np.nan)
                            
                            # Calculate change in response
                            dr = r_current - r_prev
                            # Mask out where change is too aggressive (e.g., > 2.0)
                            fluctuation_mask = np.abs(dr) <= 2.0

                # Build extent_mask like in graph calculation
                if w_oct_i is not None and mask_oct_i is not None:
                    oct_gray = w_oct_i if w_oct_i.ndim == 2 else cv2.cvtColor(w_oct_i, cv2.COLOR_RGB2GRAY)
                    extent_mask = (mask_oct_i) & (oct_gray > 0)
                    for m in (mask_a_ref, mask_s_ref, mask_a_cur, mask_s_cur):
                        if m is not None:
                            extent_mask &= m
                    if final_mask is not None:
                        extent_mask &= final_mask
                else:
                    extent_mask = np.zeros_like(d_att, dtype=bool)

                # Calculate response ratio
                response_ratio = np.where(d_str > self.eps, d_att / d_str, np.nan)
                
                # Identify valid pixels
                valid = extent_mask & np.isfinite(response_ratio)
                
                if np.any(valid):
                    self._response_data = response_ratio  # Store for colorbar and histogram
                    
                    # Use linear scaling with 2nd-98th percentile range
                    response_valid = response_ratio[valid]
                    if len(response_valid) > 0:
                        vmin = np.percentile(response_valid, 2)
                        vmax = np.percentile(response_valid, 98)
                        
                        # Apply linear scaling with percentile range
                        response_clipped = np.clip(response_ratio, vmin, vmax)
                        response_norm = (response_clipped - vmin) / (vmax - vmin + 1e-8)
                        response_norm = np.clip(response_norm, 0, 1)
                        
                        # Apply plasma colormap
                        rgba = np.zeros((*response_ratio.shape, 4), dtype=np.float32)
                        cmap = plt.cm.jet
                        colors = cmap(response_norm)
                        rgba[..., 0] = colors[..., 0]
                        rgba[..., 1] = colors[..., 1]
                        rgba[..., 2] = colors[..., 2]
                        rgba[..., 3] = 1
                        rgba[~valid] = 0
                        
                        rgba_uint8 = (rgba * 255.0 + 0.5).astype(np.uint8)
                        ratio_img = Image.fromarray(rgba_uint8).convert('RGBA')
                        base_rgba = Image.fromarray(display_rgb).convert('RGBA')
                        display_rgb = Image.alpha_composite(base_rgba, ratio_img)
                        display_rgb = np.array(display_rgb.convert('RGB'))
                        self._current_overlay_type = 'att_str_ratio'
                        self._current_overlay_data = response_ratio

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
                a_arr = block_average(a_src)
            else:
                w_oct_i, mask_oct_i = self._warp_with_shared_flow(self._safe_get(self.images["oct"], idx), idx)
                if w_oct_i is None or mask_oct_i is None: return None, None
                s_src = self._safe_get(self.scalar_images["stress"], idx)
                a_src = self._safe_get(self.scalar_images["att"],    idx)
                if s_src is None or a_src is None: return None, None
                s_arr, mask_s = self._warp_with_shared_flow(s_src, idx)
                a_arr, mask_a = self._warp_with_shared_flow(a_src, idx)
                if s_arr is None or a_arr is None: return None, None
                a_arr = block_average(a_arr)
                
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
        box_trajectories = []
        for box in self.boxes:
            start, end, color = box
            if start and end:
                xa, xb = sorted([int(start[0]), int(end[0])])
                ya, yb = sorted([int(start[1]), int(end[1])])
                trajectory_s, trajectory_a = [], []
                for idx in range(0, self.current_idx + 1):
                    bs, ba = self._box_means_for_frame(idx, xa, xb, ya, yb)
                    if bs is not None and ba is not None:
                        trajectory_s.append(bs)
                        trajectory_a.append(ba)
                box_trajectories.append((trajectory_s, trajectory_a, color))

        # ── Scatter refresh
        self.ax.clear()
        if self.trajectory_s and self.trajectory_a:
            # Plot all points in the trajectory
            self.ax.plot(self.trajectory_s, self.trajectory_a, "-o", alpha=0.85, label="Image Average", color="#0072b2", markersize=6)
            # Highlight current frame if we have data
            if self.current_idx < len(self.trajectory_s):
                current_s = self.trajectory_s[self.current_idx]
                current_a = self.trajectory_a[self.current_idx]
                self.ax.plot(current_s, current_a, "o", color="red", markersize=7, markerfacecolor="red", markeredgecolor="darkred", markeredgewidth=1, label="Current Frame")
        
        for i, (trajectory_s, trajectory_a, color) in enumerate(box_trajectories):
            if trajectory_s and trajectory_a:
                # Plot all points in the box trajectory
                self.ax.plot(trajectory_s, trajectory_a, "-s", alpha=0.9, label=f"Box {i+1}", color=color, markersize=6)
                # Highlight current frame if we have data
                if self.current_idx < len(trajectory_s):
                    current_box_s = trajectory_s[self.current_idx]
                    current_box_a = trajectory_a[self.current_idx]
                    self.ax.plot(current_box_s, current_box_a, "s", color="red", markersize=7, markerfacecolor="red", markeredgecolor="darkred", markeredgewidth=1)
        
        self.ax.set_xlabel("Mean Stress (kPa)", fontsize=11)
        self.ax.set_ylabel("Mean Attenuation (mm⁻¹)", fontsize=11)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_facecolor("#f7f7f7")
        self.ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)

        # Ensure the graph fits within the canvas at all times and remains square
        canvas_widget = self.scatter_widget
        canvas_widget.update_idletasks()
        dpi = self.fig.get_dpi()
        # Set figure to a fixed square size to keep graph height constant
        fixed_size = 4  # inches
        self.fig.set_size_inches(fixed_size, fixed_size, forward=True)
        # Set canvas to fixed size matching the figure
        pixel_size = int(fixed_size * dpi)
        canvas_widget.config(width=pixel_size, height=pixel_size)

        # Lock axes to current data range, but do not force equal units, just make the plot square
        s_all, a_all = [], []
        if self.trajectory_s and self.trajectory_a:
            s_all.extend(self.trajectory_s); a_all.extend(self.trajectory_a)
        for trajectory_s, trajectory_a, _ in box_trajectories:
            if trajectory_s and trajectory_a:
                s_all.extend(trajectory_s); a_all.extend(trajectory_a)

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
                self.ax.set_xlim(s_min - s_pad, s_max + s_pad)
                self.ax.set_ylim(a_min - a_pad, a_max + a_pad)
                # Make the plot square in shape (not units)
                self.ax.set_aspect('auto', adjustable='box')
                box = self.ax.get_position()
                size = min(box.width, box.height)
                self.ax.set_position([box.x0, box.y0, size, size])
            else:
                self.ax.set_xlim(0, 2)
                self.ax.set_ylim(0, 2)
        else:
            self.ax.set_xlim(0, 2)
            self.ax.set_ylim(0, 2)
        
        self.scatter_canvas.draw()

        # Remove mplcursors hover box
        if self._cursor:
            try:
                self._cursor.remove()
            except Exception:
                pass
            self._cursor = None
        # Clear the canvas before drawing to prevent old plots from showing
        self.scatter_canvas.draw()

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
        
        # Calculate trajectory extents if trajectories are enabled
        img_min_x, img_max_x = 0, pil.width
        img_min_y, img_max_y = 0, pil.height
        
        if (show_headtail or self.show_net_var.get()) and flow is not None:
            H, W = flow.shape[:2]
            step = self.quiver_step
            
            # Calculate all trajectory endpoints to find full extent
            all_points = []
            for y in range(0, H, step):
                for x in range(0, W, step):
                    point = np.array([x, y], dtype=np.float32)
                    all_points.append(point.copy())
                    
                    for idx in range(1, self.current_idx + 1):
                        flow_inc = self.incremental_flows.get(idx)
                        if flow_inc is None: break
                        xi = int(np.clip(point[0], 0, W - 1))
                        yi = int(np.clip(point[1], 0, H - 1))
                        delta = flow_inc[yi, xi]
                        point = point + delta
                        all_points.append(point.copy())
            
            if all_points:
                all_points = np.array(all_points)
                traj_min_x, traj_max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
                traj_min_y, traj_max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
                
                # Expand image bounds to include trajectory extents with padding
                padding = 20
                img_min_x = min(0, traj_min_x - padding)
                img_max_x = max(pil.width, traj_max_x + padding)
                img_min_y = min(0, traj_min_y - padding)
                img_max_y = max(pil.height, traj_max_y + padding)
        
        # Calculate effective image dimensions including trajectory extents
        effective_width = img_max_x - img_min_x
        effective_height = img_max_y - img_min_y
        
        # Scale to fit canvas while maintaining aspect ratio
        scale = min(c_w / effective_width, c_h / effective_height)
        
        # Scale and position the actual image
        nw, nh = int(pil.width * scale), int(pil.height * scale)
        resized = pil.resize((nw, nh), Image.BILINEAR)
        photo = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        
        # Position image accounting for trajectory extents
        img_offset_x = -img_min_x * scale
        img_offset_y = -img_min_y * scale
        canvas_center_x = (c_w - effective_width * scale) / 2
        canvas_center_y = (c_h - effective_height * scale) / 2
        
        final_x = canvas_center_x + img_offset_x
        final_y = canvas_center_y + img_offset_y
        
        canvas.create_image(int(final_x), int(final_y), anchor="nw", image=photo)
        canvas.image = photo
        
        # Update offsets for trajectory drawing
        offset_x = canvas_center_x - img_min_x * scale
        offset_y = canvas_center_y - img_min_y * scale
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
                        canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill=col, width=0.15, arrowshape=(4, 5, 2))

                if self.show_net_var.get():
                    p_start = traj_points[0]; p_end = traj_points[-1]
                    x1_net = p_start[0] * scale + offset_x; y1_net = p_start[1] * scale + offset_y
                    x2_net = p_end[0] * scale + offset_x; y2_net = p_end[1] * scale + offset_y
                    canvas.create_line(x1_net, y1_net, x2_net, y2_net, arrow=tk.LAST, fill='lime', width=0.15, arrowshape=(4, 5, 2))

        for i, (start, end, color) in enumerate(self.boxes):
            if start and end:
                x0, y0 = start; x1, y1 = end
                x0c = x0 * scale + offset_x; y0c = y0 * scale + offset_y
                x1c = x1 * scale + offset_x; y1c = y1 * scale + offset_y
                canvas.create_rectangle(x0c, y0c, x1c, y1c, outline=color, width=4)
        
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
        if self.current_box_idx >= len(self.colors):
            return  # All boxes selected
        self._active_canvas = event.widget
        start = self._canvas_to_image(event.widget, event.x, event.y)
        self.boxes.append((start, start, self.colors[self.current_box_idx]))
        self._refresh_display()

    def _on_box_drag(self, event):
        if not self.boxes or self.current_box_idx >= len(self.colors):
            return
        self._active_canvas = event.widget
        end = self._canvas_to_image(event.widget, event.x, event.y)
        self.boxes[-1] = (self.boxes[-1][0], end, self.boxes[-1][2])
        self._refresh_display()

    def _on_box_end(self, event):
        if not self.boxes or self.current_box_idx >= len(self.colors):
            return
        self._active_canvas = event.widget
        end = self._canvas_to_image(event.widget, event.x, event.y)
        self.boxes[-1] = (self.boxes[-1][0], end, self.boxes[-1][2])
        self.current_box_idx += 1
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
        self.boxes = []
        self.current_box_idx = 0
        self._refresh_display()
    
    def _show_histogram(self):
        """Generate and save histogram for current response values with 2nd-98th percentile range"""
        if not hasattr(self, '_response_data') or self._response_data is None:
            messagebox.showwarning("No Data", "Please enable the Attenuation/Stress overlay and advance to frame 1 or later.")
            return
            
        valid_response = self._response_data[np.isfinite(self._response_data)]
        if len(valid_response) == 0:
            messagebox.showwarning("No Data", "No finite response values found in current frame.")
            return
            
        try:
            # Create histogram figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Calculate statistics
            mean_val = np.mean(valid_response)
            std_val = np.std(valid_response)
            median_val = np.median(valid_response)
            p2 = np.percentile(valid_response, 2)
            p98 = np.percentile(valid_response, 98)
            p5 = np.percentile(valid_response, 5)
            p95 = np.percentile(valid_response, 95)
            
            # Create histogram with 40 bins, constrained to 2nd-98th percentile range
            n, bins, patches = ax.hist(valid_response, bins=40, alpha=0.7, color='skyblue', 
                                     edgecolor='black', range=(p2, p98))
            
            # Add vertical lines for statistics
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            ax.axvline(p2, color='purple', linestyle=':', linewidth=2, label=f'2nd %: {p2:.3f}')
            ax.axvline(p98, color='purple', linestyle=':', linewidth=2, label=f'98th %: {p98:.3f}')
            ax.axvline(p5, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'5th %: {p5:.3f}')
            ax.axvline(p95, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'95th %: {p95:.3f}')
            
            # Set x-axis limits to 2nd-98th percentile range
            ax.set_xlim(p2, p98)
            
            # Labels and title
            ax.set_xlabel('Attenuation Response (mm⁻¹/kPa)', fontsize=12)
            ax.set_ylabel('Pixel Count', fontsize=12)
            ax.set_title(f'Attenuation Response Histogram - Frame {self.current_idx}\n'
                        f'Total Pixels: {len(valid_response)}, Display Range: [{p2:.3f}, {p98:.3f}] (2nd-98th %)',
                        fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Save figure
            filename = f'response_histogram_frame_{self.current_idx:03d}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            messagebox.showinfo("Histogram Saved", f"Histogram saved as: {filename}\nRange: 2nd-98th percentile [{p2:.3f}, {p98:.3f}]")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create histogram: {str(e)}")

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
        
        ttk.Checkbutton(main_frame, text="Co-registered OCT Images (.png)", 
                       variable=self.export_warped_oct).pack(anchor='w', pady=2)
        ttk.Checkbutton(main_frame, text="Co-registered Attenuation Maps (.png)", 
                       variable=self.export_deformation).pack(anchor='w', pady=2)
        ttk.Checkbutton(main_frame, text="Co-registered Stress Maps (.png)", 
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
            num_frames = len(self.images["oct"])
            if self.export_warped_oct.get():
                total_steps += num_frames
            if self.export_deformation.get():
                total_steps += num_frames
            if self.export_overlay.get():
                total_steps += num_frames
                
            progress_bar['maximum'] = total_steps
            
            # Export co-registered OCT images
            if self.export_warped_oct.get():
                status_label.config(text="Exporting co-registered OCT images...")
                self.update_idletasks()
                
                oct_folder = os.path.join(export_folder, 'coregistered_oct')
                os.makedirs(oct_folder, exist_ok=True)
                
                for idx in range(len(self.images["oct"])):
                    if idx == 0:
                        # Reference frame - no warping needed
                        oct_img = self.images["oct"][0]
                    else:
                        # Warp to reference frame
                        oct_img, _ = self._warp_with_shared_flow(self.images["oct"][idx], idx)
                        if oct_img is None:
                            oct_img = self.images["oct"][idx]  # Fallback to original
                    
                    # Convert to PIL Image and save
                    if oct_img.ndim == 2:
                        pil_img = Image.fromarray(oct_img.astype(np.uint8), mode='L')
                    else:
                        pil_img = Image.fromarray(oct_img.astype(np.uint8))
                    
                    filename = f'oct_frame_{idx:03d}.png'
                    pil_img.save(os.path.join(oct_folder, filename))
                    
                    current_step += 1
                    progress_bar['value'] = current_step
                    self.update_idletasks()
            
            # Export co-registered attenuation maps
            if self.export_deformation.get():
                status_label.config(text="Exporting co-registered attenuation maps...")
                self.update_idletasks()
                
                att_folder = os.path.join(export_folder, 'coregistered_attenuation')
                os.makedirs(att_folder, exist_ok=True)
                
                for idx in range(len(self.scalar_images["att"])):
                    if idx == 0:
                        # Reference frame - no warping needed
                        att_data = self.scalar_images["att"][0]
                    else:
                        # Warp to reference frame
                        att_data, _ = self._warp_with_shared_flow(self.scalar_images["att"][idx], idx)
                        if att_data is None:
                            att_data = self.scalar_images["att"][idx]  # Fallback to original
                    
                    # Apply block averaging for consistency
                    att_data = block_average(att_data)
                    
                    # Convert to RGB using viridis colormap (0-10 mm⁻¹)
                    att_clipped = np.clip(att_data, 0, 10)
                    rgb_img = self._apply_colormap(att_clipped, 'viridis', 0, 10)
                    pil_img = Image.fromarray(rgb_img)
                    
                    filename = f'attenuation_frame_{idx:03d}.png'
                    pil_img.save(os.path.join(att_folder, filename))
                    
                    current_step += 1
                    progress_bar['value'] = current_step
                    self.update_idletasks()
            
            # Export co-registered stress maps
            if self.export_overlay.get():
                status_label.config(text="Exporting co-registered stress maps...")
                self.update_idletasks()
                
                stress_folder = os.path.join(export_folder, 'coregistered_stress')
                os.makedirs(stress_folder, exist_ok=True)
                
                for idx in range(len(self.scalar_images["stress"])):
                    if idx == 0:
                        # Reference frame - no warping needed
                        stress_data = self.scalar_images["stress"][0]
                    else:
                        # Warp to reference frame
                        stress_data, _ = self._warp_with_shared_flow(self.scalar_images["stress"][idx], idx)
                        if stress_data is None:
                            stress_data = self.scalar_images["stress"][idx]  # Fallback to original
                    
                    # Use log scaling for display (same as in overlay)
                    stress_abs = np.abs(stress_data)
                    stress_log = np.log10(stress_abs + 1e-6)
                    stress_log_clipped = np.clip(stress_log, 0, 2)  # log10(1) to log10(100)
                    
                    # Convert to RGB using jet colormap with log scaling
                    rgb_img = self._apply_colormap(stress_log_clipped, 'jet', 0, 2)
                    pil_img = Image.fromarray(rgb_img)
                    
                    filename = f'stress_frame_{idx:03d}.png'
                    pil_img.save(os.path.join(stress_folder, filename))
                    
                    current_step += 1
                    progress_bar['value'] = current_step
                    self.update_idletasks()
            
            progress_dialog.destroy()
            exported_items = []
            if self.export_warped_oct.get():
                exported_items.append('Co-registered OCT images')
            if self.export_deformation.get():
                exported_items.append('Co-registered attenuation maps')
            if self.export_overlay.get():
                exported_items.append('Co-registered stress maps')
            
            items_text = '\n• '.join([''] + exported_items)
            messagebox.showinfo("Export Complete", f"Successfully exported:{items_text}\n\nTo folder: {export_folder}")
            
        except Exception as e:
            progress_dialog.destroy()
            messagebox.showerror("Export Error", f"An error occurred during export:\n{str(e)}")
    
    # _generate_overlay_image method removed - replaced with direct export functionality

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