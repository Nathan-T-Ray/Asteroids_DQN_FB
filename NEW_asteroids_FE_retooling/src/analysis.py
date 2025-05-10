import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os

#matplot data
window_size = 10  # games included for moving average
metrics = ['Score', 'Lives', 'Rocks_Destroyed', 'Shots_Fired', 'Shot_Accuracy']
plot_dir = 'game_stats_plots'

#  directory for saving plots
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

csv_file = 'game_stats.csv'
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: {csv_file} not found. Ensure the file is in the current directory.")
    exit(1)

# Ensure all expected columns exist
expected_columns = ['Timestamp', 'Score', 'Lives', 'Rocks_Destroyed', 'Shots_Fired', 'Shot_Accuracy']
if not all(col in df.columns for col in expected_columns):
    print(f"Error: CSV missing required columns. Found: {df.columns}")
    exit(1)

# Convert Timestamp to datetime and create numeric index for regression
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Game_Index'] = np.arange(len(df))

# Compute moving averages
for metric in metrics:
    df[f'{metric}_MA'] = df[metric].rolling(window=window_size, min_periods=1).mean()

# Plot each metric's moving average with line of best fit
for metric in metrics:
    plt.figure(figsize=(10, 6))
    # Plot moving average
    plt.plot(df['Timestamp'], df[f'{metric}_MA'], label=f'{metric} (MA, window={window_size})', color='blue')
    
    # Linear regression
    X = df['Game_Index'].values.reshape(-1, 1)
    y = df[f'{metric}_MA'].dropna().values
    if len(y) > 1:  # Need at least 2 points for regression
        X_fit = X[:len(y)]  # Match X to y length (in case of NaNs)
        reg = LinearRegression().fit(X_fit, y)
        y_pred = reg.predict(X_fit)
        plt.plot(df['Timestamp'][:len(y)], y_pred, label=f'Best Fit (slope={reg.coef_[0]:.2f})', color='red', linestyle='--')
    
    plt.xlabel('Time')
    plt.ylabel(metric)
    plt.title(f'{metric} Moving Average with Line of Best Fit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save plot
    plt.savefig(os.path.join(plot_dir, f'{metric.lower()}_ma_fit_plot.png'))
    plt.close()

# Plot all moving averages with lines of best fit in one figure
plt.figure(figsize=(12, 8))
for metric in metrics:
    plt.plot(df['Timestamp'], df[f'{metric}_MA'], label=f'{metric} (MA)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(f'Game Stats Moving Averages (Window={window_size})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'all_metrics_ma_plot.png'))
plt.close()

print(f"Plots saved in '{plot_dir}' directory:")
for metric in metrics:
    print(f"- {metric.lower()}_ma_fit_plot.png")
print("- all_metrics_ma_plot.png")