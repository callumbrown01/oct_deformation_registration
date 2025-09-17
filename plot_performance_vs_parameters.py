import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import pearsonr

# Paths to CSVs
base_dir = "synthetic oct data 3"
perf_csv = os.path.join(base_dir, "algorithm_performance_new.csv")
param_csv = os.path.join(base_dir, "all_sample_parameters.csv")

# Load data
perf_df = pd.read_csv(perf_csv)
param_df = pd.read_csv(param_csv)

# Filter out Lucas-Kanade
perf_df = perf_df[perf_df['algorithm'] != 'Lucas-Kanade']

# Ensure both merge keys are strings and match sample naming
perf_df["sample"] = perf_df["sample"].astype(str)
param_df["sample_idx"] = param_df["sample_idx"].astype(int)
param_df["sample"] = param_df["sample_idx"].apply(lambda x: f"sample_{x:03d}")

merged = perf_df.merge(param_df, left_on="sample", right_on="sample", how="left")
algorithms = merged["algorithm"].unique()
param_names = [p for p in ["struct_sigma", "speckle_scale", "magnitude"] if p in merged.columns]

# 1. Bar plot: Average EPE across all algorithms for each sample with normalized parameters as lines
plt.figure(figsize=(20, 8))
avg_epe_per_sample = merged.groupby("sample")["avg_epe"].mean()
ax = plt.gca()
avg_epe_per_sample.plot(kind="bar", ax=ax, width=0.7, color="skyblue")
ax.set_ylabel("Average EPE (across all algorithms)")
ax.set_xlabel("Sample")
ax.set_title("Average EPE per Sample (across all algorithms) with Sample Parameters")
ax.grid(True, alpha=0.3)

# Overlay normalized parameter lines
ax2 = ax.twinx()
line_handles = []
line_labels = []
colors = ['red', 'green', 'orange']
for i, pname in enumerate(param_names):
    vals = merged.drop_duplicates("sample").set_index("sample")[pname]
    norm_vals = (vals - vals.min()) / (vals.max() - vals.min())
    lh, = ax2.plot(range(len(vals)), norm_vals, label=f"{pname} (norm)", 
                   marker='o', linestyle='--', color=colors[i], linewidth=2)
    line_handles.append(lh)
    line_labels.append(f"{pname} (norm)")

ax2.set_ylabel("Normalized Parameter Value")
ax2.legend(handles=line_handles, labels=line_labels, loc='upper right')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "1_average_epe_per_sample_with_params.png"), bbox_inches='tight', dpi=300)
plt.close()

# 2. Correlation plots: Average EPE vs each parameter with best fit lines and correlation coefficients
fig, axes = plt.subplots(1, len(param_names), figsize=(16, 5))
if len(param_names) == 1:
    axes = [axes]

# Collect all legend handles and labels
all_handles = []
all_labels = []

for i, param in enumerate(param_names):
    for j, alg in enumerate(algorithms):
        alg_data = merged[merged["algorithm"] == alg]
        scatter = axes[i].scatter(alg_data[param], alg_data["avg_epe"], label=alg, alpha=0.7)
        if i == 0:  # Only collect handles from first subplot
            all_handles.append(scatter)
            all_labels.append(alg)
    
    # Add best fit line for all data points
    all_x = merged[merged[param].notna()][param]
    all_y = merged[merged[param].notna()]["avg_epe"]
    if len(all_x) > 1:
        z = np.polyfit(all_x, all_y, 1)
        p = np.poly1d(z)
        axes[i].plot(all_x, p(all_x), "k--", alpha=0.8, linewidth=2)
        
        # Calculate correlation coefficient
        corr_coeff, p_value = pearsonr(all_x, all_y)
        axes[i].text(0.05, 0.95, f'r = {corr_coeff:.3f}\np = {p_value:.3f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[i].set_xlabel(param)
    axes[i].set_ylabel("Average EPE")
    axes[i].set_title(f"EPE vs {param}")
    axes[i].grid(True, alpha=0.3)

# Create single legend outside the plots but closer
fig.legend(all_handles, all_labels, bbox_to_anchor=(0.96, 0.5), loc='center left')
plt.subplots_adjust(right=0.94)
plt.savefig(os.path.join(base_dir, "2_correlation_epe_vs_parameters.png"), bbox_inches='tight', dpi=300)
plt.close()

# 3. Combined EPE and Pixel Correlation performance - better visualization approaches
avg_perf = merged.groupby("algorithm")[["avg_epe", "avg_pixel_corr"]].mean()

# Load actual data performance (real_data_performance.csv) from synthetic oct data 3
real_perf_path = os.path.join(base_dir, "real_data_performance.csv")
if os.path.exists(real_perf_path):
    real_df = pd.read_csv(real_perf_path)
    # Filter out Lucas-Kanade from real data too
    real_df = real_df[real_df['algorithm'] != 'Lucas-Kanade']
    avg_real_pixel_corr = real_df.groupby("algorithm")["algorithm_pixel_corr"].mean()
else:
    avg_real_pixel_corr = None

# Option 1: Scatter plot showing relationship between synthetic and actual performance
if avg_real_pixel_corr is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Raw values comparison
    algorithms_list = list(avg_perf.index)
    x_pos = np.arange(len(algorithms_list))
    
    # Plot EPE (inverted for better comparison - lower EPE = better)
    ax1_twin = ax1.twinx()
    bars1 = ax1.bar(x_pos - 0.2, avg_perf["avg_epe"], 0.4, label='EPE (synthetic)', color='lightcoral', alpha=0.7)
    
    # Plot pixel correlations
    bars2 = ax1_twin.bar(x_pos + 0.2, avg_perf["avg_pixel_corr"], 0.4, label='PC (synthetic)', color='lightblue', alpha=0.7)
    avg_real_pixel_corr_ordered = avg_real_pixel_corr.reindex(avg_perf.index)
    bars3 = ax1_twin.bar(x_pos + 0.2, avg_real_pixel_corr_ordered, 0.4, label='PC (actual)', color='darkblue', alpha=0.7, bottom=avg_perf["avg_pixel_corr"])
    
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Average EPE", color='red')
    ax1_twin.set_ylabel("Pixel Correlation", color='blue')
    ax1.set_title("Raw Performance Metrics Comparison")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms_list, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Right plot: Scatter plot showing synthetic vs actual pixel correlation
    avg_real_pixel_corr_ordered = avg_real_pixel_corr.reindex(avg_perf.index)
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms_list)))
    
    for i, alg in enumerate(algorithms_list):
        ax2.scatter(avg_perf.loc[alg, "avg_pixel_corr"], 
                   avg_real_pixel_corr_ordered[alg], 
                   label=alg, s=100, color=colors[i], alpha=0.7, edgecolors='black')
    
    # Add diagonal line for perfect correlation
    min_val = min(avg_perf["avg_pixel_corr"].min(), avg_real_pixel_corr_ordered.min())
    max_val = max(avg_perf["avg_pixel_corr"].max(), avg_real_pixel_corr_ordered.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect correlation')
    
    ax2.set_xlabel("Pixel Correlation (Synthetic Data)")
    ax2.set_ylabel("Pixel Correlation (Actual Data)")
    ax2.set_title("Synthetic vs Actual Performance Correlation")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "3a_performance_comparison_detailed.png"), bbox_inches='tight', dpi=300)
    plt.close()

# Option 2: Ranking comparison
fig, ax = plt.subplots(figsize=(12, 8))

# Create rankings for each metric - invert EPE so lower values get better (higher) ranks
epe_rank = avg_perf["avg_epe"].rank(ascending=True)  # Lower EPE = better rank
synthetic_pc_rank = avg_perf["avg_pixel_corr"].rank(ascending=False)  # Higher PC = better rank

if avg_real_pixel_corr is not None:
    avg_real_pixel_corr_ordered = avg_real_pixel_corr.reindex(avg_perf.index)
    actual_pc_rank = avg_real_pixel_corr_ordered.rank(ascending=False)
    
    # Plot rankings - invert EPE ranks to show better performance as higher bars
    x_pos = np.arange(len(avg_perf.index))
    width = 0.25
    
    # Invert EPE ranking so better (lower EPE) appears higher
    inverted_epe_rank = (len(avg_perf.index) + 1) - epe_rank
    
    bars1 = ax.bar(x_pos - width, inverted_epe_rank, width, label='EPE Performance (synthetic)', color='lightcoral', alpha=0.8)
    bars2 = ax.bar(x_pos, synthetic_pc_rank, width, label='PC Ranking (synthetic)', color='lightblue', alpha=0.8)
    bars3 = ax.bar(x_pos + width, actual_pc_rank, width, label='PC Ranking (actual)', color='darkblue', alpha=0.8)
    
    # Add value labels on bars
    for bars, values in [(bars1, inverted_epe_rank), (bars2, synthetic_pc_rank), (bars3, actual_pc_rank)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(val)}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel("Algorithm")
ax.set_ylabel("Performance Ranking (Higher = Better)")
ax.set_title("Algorithm Performance Rankings\n(Higher bars = better performance)")
ax.set_xticks(x_pos)
ax.set_xticklabels(avg_perf.index, rotation=45)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, len(avg_perf.index) + 0.5)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "3b_performance_rankings.png"), bbox_inches='tight', dpi=300)
plt.close()

# Option 3: Radar/Spider chart for comprehensive comparison
if avg_real_pixel_corr is not None:
    from math import pi
    
    # Prepare data for radar chart
    algorithms_subset = avg_perf.index[:4]  # Show top 4 algorithms to avoid clutter
    
    # Normalize metrics to 0-1 scale for radar chart
    epe_norm = 1 - (avg_perf["avg_epe"] - avg_perf["avg_epe"].min()) / (avg_perf["avg_epe"].max() - avg_perf["avg_epe"].min())  # Invert EPE
    synthetic_pc_norm = (avg_perf["avg_pixel_corr"] - avg_perf["avg_pixel_corr"].min()) / (avg_perf["avg_pixel_corr"].max() - avg_perf["avg_pixel_corr"].min())
    avg_real_pixel_corr_ordered = avg_real_pixel_corr.reindex(avg_perf.index)
    actual_pc_norm = (avg_real_pixel_corr_ordered - avg_real_pixel_corr_ordered.min()) / (avg_real_pixel_corr_ordered.max() - avg_real_pixel_corr_ordered.min())
    
    # Set up radar chart
    categories = ['EPE\n(inverted)', 'PC (synthetic)', 'PC (actual)']
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]  # Complete the circle
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms_subset)))
    
    for i, alg in enumerate(algorithms_subset):
        values = [
            epe_norm[alg],
            synthetic_pc_norm[alg], 
            actual_pc_norm[alg]
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title("Algorithm Performance Radar Chart\n(Outer edge = better performance)", y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "3c_performance_radar.png"), bbox_inches='tight', dpi=300)
    plt.close()

# 4. Compare algorithmic performance of each sample against its magnitude of deformation
plt.figure(figsize=(20, 8))
# For each sample, plot average EPE and average magnitude
avg_epe_per_sample = merged.groupby("sample")["avg_epe"].mean()
avg_magnitude_per_sample = merged.groupby("sample")["magnitude"].mean()
ax = plt.gca()
avg_epe_per_sample.plot(kind="bar", ax=ax, width=0.7, color="skyblue", label="Average EPE")
ax.set_ylabel("Average EPE (across all algorithms)")
ax.set_xlabel("Sample")
ax.set_title("Average EPE per Sample vs Magnitude of Deformation")
ax.grid(True, alpha=0.3)

# Overlay normalized magnitude as a line
ax2 = ax.twinx()
norm_magnitude = (avg_magnitude_per_sample - avg_magnitude_per_sample.min()) / (avg_magnitude_per_sample.max() - avg_magnitude_per_sample.min())
lh, = ax2.plot(range(len(norm_magnitude)), norm_magnitude, label="Magnitude (norm)", color="orange", marker='o', linestyle='--', linewidth=2)
ax2.set_ylabel("Normalized Magnitude")
ax2.legend([lh], ["Magnitude (norm)"], loc='upper right')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "4_epe_vs_magnitude.png"), bbox_inches='tight', dpi=300)
plt.close()

# 4. Compare algorithmic performance of each sample against its deformation pixel correlation (actual data)
real_perf_path = os.path.join(base_dir, "real_data_performance.csv")
if os.path.exists(real_perf_path):
    real_df = pd.read_csv(real_perf_path)
    # Filter out Lucas-Kanade
    real_df = real_df[real_df['algorithm'] != 'Lucas-Kanade']
    avg_pixel_corr_per_sample = real_df.groupby("sample")["algorithm_pixel_corr"].mean()
    avg_deformation_pixel_corr = real_df.groupby("sample")["deformation_pixel_corr"].mean()

    plt.figure(figsize=(20, 8))
    ax = plt.gca()
    avg_pixel_corr_per_sample.plot(kind="bar", ax=ax, width=0.7, color="lightgreen", label="Average Pixel Corr (actual)")
    ax.set_ylabel("Average Pixel Correlation (actual data)")
    ax.set_xlabel("Sample")
    ax.set_title("Average Pixel Correlation per Sample (actual data) vs Deformation Pixel Correlation")
    ax.grid(True, alpha=0.3)

    # Add deformation pixel correlation as a line on the same axis
    ax.plot(range(len(avg_deformation_pixel_corr)), avg_deformation_pixel_corr, label="Deformation Pixel Corr", color="purple", marker='o', linestyle='--', linewidth=2)
    ax.legend(loc='upper right')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "5_final_pixel_corr_vs_deformation_pixel_corr.png"), bbox_inches='tight', dpi=300)
    plt.close()


print("All plots saved to", base_dir)
