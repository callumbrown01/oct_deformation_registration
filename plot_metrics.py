import pandas as pd
import matplotlib.pyplot as plt

data = [
    {'algorithm': 'TVL1', 'average_mean_abs_diff': 23.625404, 'average_mse': 1214.1640855172743, 'average_psnr': 17.791705739576663, 'average_ssim': 0.14479234531106036},
    {'algorithm': 'DIS', 'average_mean_abs_diff': 22.854961, 'average_mse': 1160.8880604116607, 'average_psnr': 17.92773006350547, 'average_ssim': 0.16088579283049922},
    {'algorithm': 'Farneback', 'average_mean_abs_diff': 22.904186, 'average_mse': 1163.9131086646923, 'average_psnr': 17.920604469708557, 'average_ssim': 0.16131239991357793},
    {'algorithm': 'PCAFlow', 'average_mean_abs_diff': 22.69066, 'average_mse': 1145.3709233180025, 'average_psnr': 17.988830217340816, 'average_ssim': 0.16460394437656778},
    {'algorithm': 'Lucas-Kanade', 'average_mean_abs_diff': 28.007427, 'average_mse': 1607.0886297465186, 'average_psnr': 16.457658333115383, 'average_ssim': 0.09208023317823134},
    {'algorithm': 'DeepFlow', 'average_mean_abs_diff': 22.664768, 'average_mse': 1147.3290169898091, 'average_psnr': 17.978117653622363, 'average_ssim': 0.1650328209612481}
]

df = pd.DataFrame(data)
algorithms = df['algorithm']

metrics = [
    ('average_mean_abs_diff', 'Mean Abs Diff'),
    ('average_mse', 'MSE'),
    ('average_psnr', 'PSNR'),
    ('average_ssim', 'SSIM')
]

# Normalise each metric column independently to [0, 1]
df_norm = df.copy()
for key, _ in metrics:
    col = df[key]
    min_val, max_val = col.min(), col.max()
    if max_val > min_val:
        df_norm[key] = (col - min_val) / (max_val - min_val)
    else:
        df_norm[key] = 0.0

x = range(len(algorithms))
width = 0.5

# Plot each metric in a separate graph
for key, label in metrics:
    plt.figure(figsize=(8, 5))
    plt.bar(x, df[key], width, color='tab:blue')
    plt.xticks(x, algorithms, fontsize=12)
    plt.ylabel(label, fontsize=14)
    plt.xlabel('Algorithm', fontsize=14)
    plt.title(f'{label} by Algorithm', fontsize=16, weight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # Add value labels
    for xi, val in zip(x, df[key]):
        plt.text(xi, val + 0.02 * (df[key].max() - df[key].min()), f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

