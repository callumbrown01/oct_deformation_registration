#!/usr/bin/env python3
"""
Example Usage of OCT Deformation Toolkit (Modular Components)

This script demonstrates how to use the modular components of the toolkit
programmatically for batch processing and custom analysis workflows.

Author: Callum Brown
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oct_deformation_toolkit import (
    OCTDataLoader,
    OpticalFlowEngine,
    ImageProcessor,
    MetricsCalculator,
    PlotManager
)
from oct_deformation_toolkit.utils import DataExporter


def example_1_load_and_process():
    """
    Example 1: Load data and compute optical flow
    """
    print("=" * 60)
    print("Example 1: Loading and Processing OCT Data")
    print("=" * 60)
    
    # Initialize data loader
    loader = OCTDataLoader()
    
    # Load sequence from folder
    # Replace with your actual data path
    data_path = "./sample_data"
    
    print(f"\nAttempting to load data from: {data_path}")
    success = loader.load_sequence(data_path)
    
    if not success:
        print("Failed to load data. Please check the path and file format.")
        print("\nExpected file structure:")
        print("  - *OCT*.mat files (OCT images)")
        print("  - *Stress*.mat files (stress measurements)")
        print("  - *Att*.mat files (attenuation coefficients)")
        return
    
    # Get frame count
    n_frames = loader.get_frame_count()
    print(f"\nLoaded sequence with {n_frames} frames")
    
    # Initialize optical flow engine
    print("\nInitializing optical flow with DIS algorithm...")
    flow_engine = OpticalFlowEngine(algorithm_name='DIS')
    
    # Get OCT frames
    oct_frames = [loader.get_frame('oct', i) for i in range(n_frames)]
    
    # Precompute all flows
    print("Computing optical flows (this may take a moment)...")
    flow_engine.precompute_flows(oct_frames)
    print("Flow computation complete!")
    
    # Get cumulative flow for a specific frame
    frame_idx = min(5, n_frames - 1)
    cumulative_flow = flow_engine.get_cumulative_flow(frame_idx)
    
    print(f"\nFlow field shape for frame {frame_idx}: {cumulative_flow.shape}")
    print(f"Max displacement: {np.max(np.linalg.norm(cumulative_flow, axis=2)):.2f} pixels")


def example_2_warp_and_export():
    """
    Example 2: Warp images and export co-registered data
    """
    print("\n" + "=" * 60)
    print("Example 2: Warping Images and Exporting Data")
    print("=" * 60)
    
    # Initialize components
    loader = OCTDataLoader()
    data_path = "./sample_data"
    
    if not loader.load_sequence(data_path):
        print("Failed to load data.")
        return
    
    n_frames = loader.get_frame_count()
    print(f"\nProcessing {n_frames} frames...")
    
    # Initialize flow engine and exporter
    flow_engine = OpticalFlowEngine(algorithm_name='DIS')
    exporter = DataExporter(output_dir='./exports')
    
    # Get all frames
    oct_frames = [loader.get_frame('oct', i) for i in range(n_frames)]
    stress_frames = [loader.get_frame('stress', i) for i in range(n_frames)]
    atten_frames = [loader.get_frame('attenuation', i) for i in range(n_frames)]
    
    # Compute flows
    print("Computing optical flows...")
    flow_engine.precompute_flows(oct_frames)
    
    # Warp all frames to reference (frame 0)
    warped_stress = []
    warped_atten = []
    
    print("Warping stress and attenuation maps...")
    for i in range(n_frames):
        flow = flow_engine.get_cumulative_flow(i)
        
        # Warp stress
        stress_warped, _ = flow_engine.warp_image(stress_frames[i], flow)
        warped_stress.append(stress_warped)
        
        # Warp attenuation
        atten_warped, _ = flow_engine.warp_image(atten_frames[i], flow)
        warped_atten.append(atten_warped)
    
    # Export co-registered sequence
    print("\nExporting co-registered data...")
    output_dirs = exporter.export_coregistered_sequence(
        oct_frames=oct_frames,
        stress_frames=warped_stress,
        attenuation_frames=warped_atten,
        sequence_name='example_sequence'
    )
    
    print("\nData exported to:")
    for modality, path in output_dirs.items():
        print(f"  {modality}: {path}")


def example_3_compute_metrics():
    """
    Example 3: Compute and visualize metrics
    """
    print("\n" + "=" * 60)
    print("Example 3: Computing Response Metrics")
    print("=" * 60)
    
    # Initialize components
    loader = OCTDataLoader()
    processor = ImageProcessor()
    metrics_calc = MetricsCalculator()
    
    data_path = "./sample_data"
    
    if not loader.load_sequence(data_path):
        print("Failed to load data.")
        return
    
    n_frames = loader.get_frame_count()
    
    # Get reference frames
    stress_ref = loader.get_frame('stress', 0)
    atten_ref = loader.get_frame('attenuation', 0)
    
    # Get a later frame
    frame_idx = min(10, n_frames - 1)
    stress_current = loader.get_frame('stress', frame_idx)
    atten_current = loader.get_frame('attenuation', frame_idx)
    
    # Apply smoothing
    print(f"\nProcessing frame {frame_idx}...")
    stress_smooth = processor.smooth_with_nan_handling(stress_current)
    atten_smooth = processor.smooth_with_nan_handling(atten_current)
    
    # Compute response ratio
    print("Computing response ratio (ΔAttenuation/ΔStress)...")
    response = metrics_calc.compute_response_ratio(
        atten_ref, atten_smooth, stress_ref, stress_smooth
    )
    
    # Get statistics
    valid_response = response[np.isfinite(response)]
    
    if len(valid_response) > 0:
        print(f"\nResponse Ratio Statistics:")
        print(f"  Mean: {np.mean(valid_response):.4f} mm⁻¹/kPa")
        print(f"  Median: {np.median(valid_response):.4f} mm⁻¹/kPa")
        print(f"  Std: {np.std(valid_response):.4f} mm⁻¹/kPa")
        
        # Compute percentile range
        p_min, p_max = metrics_calc.compute_percentile_range(response)
        print(f"  5-95th percentile: [{p_min:.4f}, {p_max:.4f}] mm⁻¹/kPa")


def example_4_region_analysis():
    """
    Example 4: Analyze specific regions of interest
    """
    print("\n" + "=" * 60)
    print("Example 4: Region-Based Analysis")
    print("=" * 60)
    
    # Initialize components
    loader = OCTDataLoader()
    flow_engine = OpticalFlowEngine(algorithm_name='DIS')
    metrics_calc = MetricsCalculator()
    
    data_path = "./sample_data"
    
    if not loader.load_sequence(data_path):
        print("Failed to load data.")
        return
    
    n_frames = loader.get_frame_count()
    
    # Define region of interest (x, y, width, height)
    roi = (50, 50, 100, 100)  # Example coordinates
    
    print(f"\nAnalyzing region: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    
    # Compute flows
    oct_frames = [loader.get_frame('oct', i) for i in range(n_frames)]
    flow_engine.precompute_flows(oct_frames)
    
    # Track region through sequence
    stress_means = []
    atten_means = []
    
    print("\nTracking region through sequence...")
    for i in range(n_frames):
        # Get current data
        stress = loader.get_frame('stress', i)
        atten = loader.get_frame('attenuation', i)
        
        # Warp to reference
        flow = flow_engine.get_cumulative_flow(i)
        stress_warped, _ = flow_engine.warp_image(stress, flow)
        atten_warped, _ = flow_engine.warp_image(atten, flow)
        
        # Compute mean in ROI
        stress_mean = metrics_calc.compute_box_means([roi], stress_warped)[0]
        atten_mean = metrics_calc.compute_box_means([roi], atten_warped)[0]
        
        stress_means.append(stress_mean)
        atten_means.append(atten_mean)
    
    # Plot trajectory
    print("\nRegion trajectory:")
    for i, (s, a) in enumerate(zip(stress_means[:min(5, len(stress_means))], 
                                    atten_means[:min(5, len(atten_means))])):
        print(f"  Frame {i}: Stress={s:.2f} kPa, Attenuation={a:.4f} mm⁻¹")
    
    # Create simple plot
    try:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(stress_means, label='Stress')
        plt.xlabel('Frame')
        plt.ylabel('Stress (kPa)')
        plt.title('Stress over Time in ROI')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(atten_means, label='Attenuation', color='orange')
        plt.xlabel('Frame')
        plt.ylabel('Attenuation (mm⁻¹)')
        plt.title('Attenuation over Time in ROI')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('region_analysis.png', dpi=150)
        print("\nPlot saved to: region_analysis.png")
        
    except Exception as e:
        print(f"Could not create plot: {e}")


def main():
    """
    Main function to run examples.
    """
    print("\n" + "=" * 60)
    print("OCT Deformation Toolkit - Usage Examples")
    print("=" * 60)
    print("\nNOTE: These examples assume you have sample data in ./sample_data/")
    print("Modify the 'data_path' variable in each example to point to your data.")
    print("\n")
    
    # Run examples
    try:
        example_1_load_and_process()
        
        # Uncomment to run other examples
        # example_2_warp_and_export()
        # example_3_compute_metrics()
        # example_4_region_analysis()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
