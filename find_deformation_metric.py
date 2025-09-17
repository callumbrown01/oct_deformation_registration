import pandas as pd
import numpy as np
import cv2
import os
import glob
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity, mean_squared_error, normalized_root_mse
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

base_dir = "synthetic oct data 3"


def compute_optical_flow_magnitude(img1, img2):
    """Compute the average magnitude of optical flow between two images"""
    flow = cv2.calcOpticalFlowPyrLK(img1, img2, None, None)
    if flow[0] is not None:
        magnitude = np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2)
        return np.mean(magnitude)
    return 0.0

def compute_gradient_magnitude_diff(img1, img2):
    """Compute difference in gradient magnitudes"""
    grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    grad1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
    grad2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
    
    mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
    mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
    
    return np.mean(np.abs(mag1 - mag2))

def compute_histogram_distance(img1, img2, bins=256):
    """Compute histogram distance between images"""
    hist1 = cv2.calcHist([img1.astype(np.uint8)], [0], None, [bins], [0, 256])
    hist2 = cv2.calcHist([img2.astype(np.uint8)], [0], None, [bins], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

def compute_phase_correlation(img1, img2):
    """Compute phase correlation peak value"""
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    cross_power_spectrum = (f1 * np.conj(f2)) / np.abs(f1 * np.conj(f2))
    correlation = np.fft.ifft2(cross_power_spectrum)
    return np.max(np.abs(correlation))

def compute_texture_difference(img1, img2):
    """Compute local binary pattern difference"""
    lbp1 = local_binary_pattern(img1, 8, 1, method='uniform')
    lbp2 = local_binary_pattern(img2, 8, 1, method='uniform')
    return np.mean(np.abs(lbp1 - lbp2))

def compute_edge_difference(img1, img2):
    """Compute difference in edge maps"""
    edges1 = cv2.Canny(img1.astype(np.uint8), 50, 150)
    edges2 = cv2.Canny(img2.astype(np.uint8), 50, 150)
    return np.mean(np.abs(edges1.astype(float) - edges2.astype(float)))

def compute_moment_difference(img1, img2):
    """Compute difference in image moments"""
    moments1 = cv2.moments(img1.astype(np.uint8))
    moments2 = cv2.moments(img2.astype(np.uint8))
    
    # Compare central moments
    diff = 0
    for key in ['m00', 'm10', 'm01', 'm20', 'm11', 'm02']:
        if key in moments1 and key in moments2:
            diff += abs(moments1[key] - moments2[key])
    return diff

def analyze_synthetic_deformation():
    """Analyze synthetic data to find metrics correlated with EPE"""
    
    # Load synthetic data paths
    perf_csv = os.path.join(base_dir, "algorithm_performance.csv")
    
    # Load performance data
    perf_df = pd.read_csv(perf_csv)
    
    results = []
    
    # Process each sample
    sample_dirs = sorted(glob.glob(os.path.join(base_dir, "sample_*")))
    
    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)
        print(f"Processing {sample_name}...")
        
        # Load frames
        frame_files = sorted(glob.glob(os.path.join(sample_dir, "frame_*.png")))
        if len(frame_files) < 2:
            continue
            
        # Load first and last frames
        img1 = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img2 = cv2.imread(frame_files[-1], cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        if img1 is None or img2 is None:
            continue
        
        # Get average EPE for this sample from performance data
        sample_perf = perf_df[perf_df['sample'] == sample_name]
        if sample_perf.empty:
            continue
        avg_epe = sample_perf['avg_epe'].mean()
        
        # Compute various metrics ONLY from image comparison
        metrics = {
            'sample': sample_name,
            'avg_epe': avg_epe,
            'mse': mean_squared_error(img1, img2),
            'nrmse': normalized_root_mse(img1, img2),
            'ssim': structural_similarity(img1, img2, data_range=img1.max() - img1.min()),
            'gradient_diff': compute_gradient_magnitude_diff(img1, img2),
            'histogram_distance': compute_histogram_distance(img1, img2),
            'phase_correlation': compute_phase_correlation(img1, img2),
            'texture_difference': compute_texture_difference(img1, img2),
            'edge_difference': compute_edge_difference(img1, img2),
            'moment_difference': compute_moment_difference(img1, img2),
            'mean_abs_diff': np.mean(np.abs(img1 - img2)),
            'std_diff': np.std(img1 - img2),
            'max_abs_diff': np.max(np.abs(img1 - img2)),
            'pixel_correlation': np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
        }
        
        results.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate correlations with EPE (exclude sample and avg_epe from metrics)
    metric_names = [col for col in df.columns if col not in ['sample', 'avg_epe']]
    correlations = []
    
    print("\nCorrelation with Average EPE:")
    print("=" * 50)
    
    for metric in metric_names:
        try:
            corr, p_value = pearsonr(df[metric], df['avg_epe'])
            correlations.append({
                'metric': metric,
                'correlation': abs(corr),  # Use absolute correlation
                'raw_correlation': corr,
                'p_value': p_value
            })
            print(f"{metric:20s}: r = {corr:6.3f} (p = {p_value:.3f})")
        except:
            print(f"{metric:20s}: Could not compute correlation")
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x['correlation'], reverse=True)
    
    print(f"\nTop 5 image-based metrics correlated with EPE:")
    print("=" * 50)
    for i, corr in enumerate(correlations[:5]):
        print(f"{i+1}. {corr['metric']:20s}: |r| = {corr['correlation']:.3f}")
    
    # Save results
    df.to_csv(os.path.join(base_dir, "deformation_metrics.csv"), index=False)
    
    # Create correlation summary
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(os.path.join(base_dir, "metric_correlations.csv"), index=False)
    
    # Plot top correlations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, corr in enumerate(correlations[:6]):
        metric = corr['metric']
        ax = axes[i]
        
        ax.scatter(df[metric], df['avg_epe'], alpha=0.7)
        ax.set_xlabel(metric)
        ax.set_ylabel('Average EPE')
        ax.set_title(f'{metric}\nr = {corr["raw_correlation"]:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df[metric], df['avg_epe'], 1)
        p = np.poly1d(z)
        ax.plot(df[metric], p(df[metric]), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "metric_correlations_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to:")
    print(f"- {os.path.join(base_dir, 'deformation_metrics.csv')}")
    print(f"- {os.path.join(base_dir, 'metric_correlations.csv')}")
    print(f"- {os.path.join(base_dir, 'metric_correlations_plot.png')}")
    
    return df, correlations

if __name__ == "__main__":
    df, correlations = analyze_synthetic_deformation()

