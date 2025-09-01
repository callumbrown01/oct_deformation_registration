import pandas as pd
import matplotlib.pyplot as plt

# Manually enter the summary data_data 
synthetic_data = [
    {'algorithm': 'TVL1', 'epe': 7.366},
    {'algorithm': 'DIS', 'epe': 9.311},
    {'algorithm': 'Farneback', 'epe': 9.608},
    {'algorithm': 'PCAFlow', 'epe': 11.648},
    {'algorithm': 'Lucas-Kanade', 'epe': 11.78},
    {'algorithm': 'DeepFlow', 'epe': 12.101}
]
real_data = [
    {'algorithm': 'TVL1', 'mse': 2.8429081},
    {'algorithm': 'DIS', 'mse': 2.6045694},
    {'algorithm': 'Farneback', 'mse': 2.8896132},
    {'algorithm': 'PCAFlow', 'mse': 2.5830717},
    {'algorithm': 'Lucas-Kanade', 'mse': 3.0711842},
    {'algorithm': 'DeepFlow', 'mse': 2.5687835}
]

df_real = pd.DataFrame(real_data)
df_synth = pd.DataFrame(synthetic_data)

algorithms = df_real['algorithm']
x = range(len(algorithms))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar([i - width/2 for i in x], df_real['mse'], width, label='Real Data')
bars2 = plt.bar([i + width/2 for i in x], df_synth['epe'], width, label='Synthetic Data')

plt.title('End-Point Error (EPE) by Algorithm', fontsize=16, weight='bold')
plt.ylabel('EPE (pixels)', fontsize=14)
plt.xlabel('Algorithm', fontsize=14)
plt.xticks(x, algorithms, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(fontsize=12)

# Add value labels on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.2, f'{height:.2f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()