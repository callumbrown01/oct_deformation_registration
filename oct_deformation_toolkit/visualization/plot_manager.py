"""
Plot Management Module

Manages matplotlib plotting for scatter plots and histograms.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Tuple, Optional
import tkinter as tk


class PlotManager:
    """
    Manages matplotlib plots for data visualization.
    
    Handles:
    - Stress-attenuation scatter plots
    - Trajectory plotting with multiple regions
    - Histogram generation
    """
    
    def __init__(self, figure, axis, canvas_widget):
        """
        Initialize the plot manager.
        
        Args:
            figure: Matplotlib Figure object
            axis: Matplotlib Axes object
            canvas_widget: FigureCanvasTkAgg widget
        """
        self.fig = figure
        self.ax = axis
        self.canvas = canvas_widget
    
    def plot_trajectories(self,
                         stress_values: List[float],
                         att_values: List[float],
                         box_trajectories: Optional[List[Tuple]] = None,
                         current_idx: Optional[int] = None,
                         xlabel: str = "Mean Stress (kPa)",
                         ylabel: str = "Mean Attenuation (mm⁻¹)") -> None:
        """
        Plot stress-attenuation trajectories.
        
        Args:
            stress_values: List of stress values over time
            att_values: List of attenuation values over time
            box_trajectories: Optional list of (stress_list, att_list, color) for boxes
            current_idx: Index of current frame to highlight
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        self.ax.clear()
        
        # Plot main trajectory
        if stress_values and att_values:
            self.ax.plot(stress_values, att_values, "-o", 
                        alpha=0.85, label="Image Average",
                        color="#0072b2", markersize=6)
            
            # Highlight current frame
            if current_idx is not None and current_idx < len(stress_values):
                self.ax.plot(stress_values[current_idx], att_values[current_idx],
                           "o", color="red", markersize=7,
                           markerfacecolor="red", markeredgecolor="darkred",
                           markeredgewidth=1, label="Current Frame")
        
        # Plot box trajectories
        if box_trajectories:
            for i, (box_stress, box_att, color) in enumerate(box_trajectories):
                if box_stress and box_att:
                    self.ax.plot(box_stress, box_att, "-s",
                               alpha=0.9, label=f"Box {i+1}",
                               color=color, markersize=6)
                    
                    # Highlight current frame for this box
                    if current_idx is not None and current_idx < len(box_stress):
                        self.ax.plot(box_stress[current_idx], box_att[current_idx],
                                   "s", color="red", markersize=7,
                                   markerfacecolor="red", markeredgecolor="darkred",
                                   markeredgewidth=1)
        
        # Configure plot
        self.ax.set_xlabel(xlabel, fontsize=11)
        self.ax.set_ylabel(ylabel, fontsize=11)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_facecolor("#f7f7f7")
        self.ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
                      ncol=2, fontsize=10)
        
        # Auto-scale axes with padding
        self._auto_scale_axes(stress_values, att_values, box_trajectories)
        
        self.canvas.draw()
    
    def _auto_scale_axes(self,
                        main_stress: List[float],
                        main_att: List[float],
                        box_trajectories: Optional[List[Tuple]] = None) -> None:
        """
        Automatically scale axes based on data range.
        
        Args:
            main_stress: Main trajectory stress values
            main_att: Main trajectory attenuation values
            box_trajectories: Optional box trajectory data
        """
        # Collect all values
        all_stress = list(main_stress) if main_stress else []
        all_att = list(main_att) if main_att else []
        
        if box_trajectories:
            for box_stress, box_att, _ in box_trajectories:
                all_stress.extend(box_stress)
                all_att.extend(box_att)
        
        if not all_stress or not all_att:
            self.ax.set_xlim(0, 2)
            self.ax.set_ylim(0, 2)
            return
        
        # Convert to arrays and filter out NaN/inf
        all_stress = np.asarray(all_stress, dtype=float)
        all_att = np.asarray(all_att, dtype=float)
        
        mask = np.isfinite(all_stress) & np.isfinite(all_att)
        if not np.any(mask):
            self.ax.set_xlim(0, 2)
            self.ax.set_ylim(0, 2)
            return
        
        all_stress = all_stress[mask]
        all_att = all_att[mask]
        
        # Calculate ranges with padding
        s_min, s_max = float(np.min(all_stress)), float(np.max(all_stress))
        a_min, a_max = float(np.min(all_att)), float(np.max(all_att))
        
        s_range = s_max - s_min if s_max > s_min else 0.1
        a_range = a_max - a_min if a_max > a_min else 0.1
        
        s_pad = s_range * 0.05
        a_pad = a_range * 0.05
        
        self.ax.set_xlim(s_min - s_pad, s_max + s_pad)
        self.ax.set_ylim(a_min - a_pad, a_max + a_pad)
    
    def save_histogram(self,
                      data: np.ndarray,
                      filename: str,
                      title: str = "Response Histogram",
                      xlabel: str = "Response Value",
                      bins: int = 25,
                      percentile_range: Tuple[float, float] = (2, 98)) -> None:
        """
        Generate and save a histogram figure.
        
        Args:
            data: Data array to plot
            filename: Output filename
            title: Plot title
            xlabel: X-axis label
            bins: Number of histogram bins
            percentile_range: (lower, upper) percentiles for x-axis range
        """
        # Filter to finite values only
        valid_data = data[np.isfinite(data)]
        
        if len(valid_data) == 0:
            print("No valid data for histogram")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate percentile range
        p_low, p_high = percentile_range
        vmin = np.percentile(valid_data, p_low)
        vmax = np.percentile(valid_data, p_high)
        
        # Plot histogram
        ax.hist(valid_data, bins=bins, range=(vmin, vmax),
               color='steelblue', edgecolor='black', alpha=0.7)
        
        # Add statistics text
        stats_text = (f"Mean: {np.mean(valid_data):.3f}\n"
                     f"Std: {np.std(valid_data):.3f}\n"
                     f"Median: {np.median(valid_data):.3f}\n"
                     f"N = {len(valid_data)}")
        
        ax.text(0.95, 0.95, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
        
        # Configure plot
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for percentiles
        ax.axvline(vmin, color='red', linestyle='--', linewidth=1.5,
                  label=f'{p_low}th percentile')
        ax.axvline(vmax, color='red', linestyle='--', linewidth=1.5,
                  label=f'{p_high}th percentile')
        ax.legend()
        
        # Save figure
        fig.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Histogram saved to: {filename}")
