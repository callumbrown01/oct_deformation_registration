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

# 3. Combined EPE and MSE performance using grouped bars with MSE on secondary axis
avg_perf = merged.groupby("algorithm")[["avg_epe", "avg_mse"]].mean()

# Load actual data performance (real_data_performance.csv) from synthetic oct data 3
real_perf_path = os.path.join(base_dir, "real_data_performance.csv")
if os.path.exists(real_perf_path):
    real_df = pd.read_csv(real_perf_path)
    avg_real_mse = real_df.groupby("algorithm")["mse"].mean()
else:
    avg_real_mse = None

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(avg_perf.index))
width = 0.35

bars1 = ax.bar(x - width/2, avg_perf["avg_epe"], width, label='Average EPE', color='skyblue')
ax.set_xlabel("Algorithm")
ax.set_ylabel("Average EPE", color='skyblue')
ax.set_title("Average Performance: EPE and MSE by Algorithm")
ax.set_xticks(x)
ax.set_xticklabels(avg_perf.index, rotation=45)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='y', labelcolor='skyblue')

# Secondary axis for MSE
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, avg_perf["avg_mse"], width, label='Average MSE (synthetic)', color='lightcoral')
ax2.set_ylabel("Average MSE", color='lightcoral')
ax2.tick_params(axis='y', labelcolor='lightcoral')

# Add actual data MSE as a line
if avg_real_mse is not None:
    avg_real_mse_ordered = avg_real_mse.reindex(avg_perf.index)
    ax2.plot(x, avg_real_mse_ordered, label='Average MSE (actual)', color='black', marker='o', linewidth=2)

# Move legend outside the plot
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "3_combined_performance_epe_mse.png"), bbox_inches='tight', dpi=300)
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

# 4. Compare algorithmic performance of each sample against its magnitude of deformation (actual data)
real_perf_path = os.path.join(base_dir, "real_data_performance.csv")
if os.path.exists(real_perf_path):
    real_df = pd.read_csv(real_perf_path)
    # Use magnitude (visually implied) from the real data CSV
    avg_mse_per_sample = real_df.groupby("sample")["mse"].mean()
    magnitude_map = real_df.set_index("sample")["magnitude (visually implied)"].to_dict()
    # Map magnitude to numeric for plotting (low=0, medium=0.5, high=1)
    mag_numeric = {"low": 0, "medium": 0.5, "high": 1}
    norm_magnitude = pd.Series([mag_numeric.get(magnitude_map[sample], np.nan) for sample in avg_mse_per_sample.index], index=avg_mse_per_sample.index)

    plt.figure(figsize=(20, 8))
    ax = plt.gca()
    avg_mse_per_sample.plot(kind="bar", ax=ax, width=0.7, color="lightcoral", label="Average MSE (actual)")
    ax.set_ylabel("Average MSE (actual data)")
    ax.set_xlabel("Sample")
    ax.set_title("Average MSE per Sample (actual data) vs Magnitude of Deformation (visually implied)")
    ax.grid(True, alpha=0.3)

    # Overlay normalized magnitude as a line
    ax2 = ax.twinx()
    lh, = ax2.plot(range(len(norm_magnitude)), norm_magnitude, label="Magnitude (norm)", color="orange", marker='o', linestyle='--', linewidth=2)
    ax2.set_ylabel("Normalized Magnitude (visually implied)")
    ax2.legend([lh], ["Magnitude (norm)"], loc='upper right')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "4_actual_mse_vs_magnitude.png"), bbox_inches='tight', dpi=300)
    plt.close()

print("All plots saved to", base_dir)
