import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File paths for the five CSV files (adjust these paths based on your actual file locations)
file_paths = {
    "Ep 401-500": "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-01_17-49-39_final1_485.csv",
    "Ep 301-400": "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-01_17-52-27_final1_385.csv",
    "Ep 201-300": "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-01_18-03-05_final1_290.csv",  # Replace with actual path
    "Ep 101-200": "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-01_18-05-49_final1_100.csv", # Replace with actual path
    "Ep 1-100": "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-01_18-09-06_final1_35.csv"  # Replace with actual path
}

# Directory to save plots
plot_dir = "eval_plot_images"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Read the CSV files into a list of DataFrames
dfs = {label: pd.read_csv(path) for label, path in file_paths.items()}

# Colors for each file in plots
colors = ['blue', 'red', 'green', 'purple', 'orange']

# Function to calculate time elapsed per frame and total time elapsed
def calculate_time_elapsed(df, label):
    df['time_per_frame'] = df['distance'] / df['vel']
    df['time_per_frame'] = df['time_per_frame'].replace([np.inf, -np.inf, np.nan], 0)
    total_time = df['time_per_frame'].sum()
    print(f"\n{label} - Total Time Elapsed: {total_time:.4f} seconds")
    return total_time, df['time_per_frame']

# Function to analyze CTE <= -1.0
def analyze_cte(df, label, total_time):
    cte_condition = df['cte'] <= -1.0
    num_times_cte = cte_condition.sum()
    time_cte = df[cte_condition]['time_per_frame'].sum() if 'time_per_frame' in df else 0
    print(f"{label} - Number of times CTE <= -1.0: {num_times_cte}")
    print(f"{label} - Total time with CTE <= -1.0: {time_cte:.4f} seconds")
    print(f"{label} - Percentage of time with CTE <= -1.0: {(time_cte/total_time)*100:.2f}%")

# Function to calculate mean velocity
def calculate_mean_velocity(df, label):
    mean_vel = df['vel'].mean()
    print(f"{label} - Mean Velocity: {mean_vel:.4f}")
    return mean_vel

# Function to smooth data using a moving average
def smooth_data(data, window_size=5):
    return data.rolling(window=window_size, center=True).mean().fillna(data)

# Function to calculate percentage of track completed
def calculate_track_completion(df, label):
    total_distance = df['distance'].abs().sum()
    track_completion = (total_distance / 80.0) * 100
    print(f"{label} - Percentage of Track Completed: {track_completion:.2f}%")
    return track_completion

# Function to plot steering differences with smoothing for multiple files
def plot_steering_diff(dfs_dict, window_size=5):
    for label, df in dfs_dict.items():
        df_copy = df.copy()
        df_copy['steer_diff'] = df_copy['action_steer'].diff().fillna(0)
        df_copy['steer_diff_smooth'] = smooth_data(np.abs(df_copy['steer_diff']), window_size)
        dfs_dict[label] = df_copy
    
    # Individual plots for each file
    # for (label, df), color in zip(dfs_dict.items(), colors):
    #     plt.figure(figsize=(10, 6))
    #     plt.ylim([0, 0.025])
    #     plt.plot(df.index, df['steer_diff_smooth'], label=label, color=color)
    #     plt.title(f"Steering Difference (Smoothed) - {label}")
    #     plt.xlabel("Frame (Row Number)")
    #     plt.ylabel("Steering Difference")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(plot_dir, f"steering_diff_{label.replace(' ', '_')}.png"))
    #     plt.close()
    
    # Combined plot for all files
    plt.figure(figsize=(10, 6))
    plt.ylim([0, 1.0])
    for (label, df), color in zip(dfs_dict.items(), colors):
        plt.plot(df.index, df['steer_diff_smooth'], label=label, color=color)
    plt.title("Steering Difference Comparison Over Training Stages")
    plt.xlabel("Frame")
    plt.ylabel("Steering Difference")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "steering_diff_all.png"))
    plt.close()

# Function to plot throttle with smoothing for multiple files
def plot_throttle(dfs_dict, window_size=5):
    for label, df in dfs_dict.items():
        df_copy = df.copy()
        df_copy['throttle_smooth'] = smooth_data(df_copy['action_throttle'], window_size)
        dfs_dict[label] = df_copy
    
    # Individual plots for each file
    # for (label, df), color in zip(dfs_dict.items(), colors):
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(df.index, df['throttle_smooth'], label=label, color=color)
    #     plt.title(f"Throttle (Smoothed) - {label}")
    #     plt.xlabel("Frame (Row Number)")
    #     plt.ylabel("Throttle")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(plot_dir, f"throttle_{label.replace(' ', '_')}.png"))
    #     plt.close()
    
    # Combined plot for all files
    plt.figure(figsize=(10, 6))
    for (label, df), color in zip(dfs_dict.items(), colors):
        plt.plot(df.index, df['throttle_smooth'], label=label, color=color)
    plt.title("Throttle Comparison Over Training Stages")
    plt.xlabel("Frame")
    plt.ylabel("Throttle")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "throttle_all.png"))
    plt.close()

# Function to plot velocity with smoothing for multiple files
def plot_velocity(dfs_dict, window_size=5):
    for label, df in dfs_dict.items():
        df_copy = df.copy()
        df_copy['vel_smooth'] = smooth_data(df_copy['vel'], window_size)
        dfs_dict[label] = df_copy
    
    # Individual plots for each file
    # for (label, df), color in zip(dfs_dict.items(), colors):
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(df.index, df['vel_smooth'], label=label, color=color)
    #     plt.title(f"Velocity (Smoothed) - {label}")
    #     plt.xlabel("Frame")
    #     plt.ylabel("Velocity")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(os.path.join(plot_dir, f"velocity_{label.replace(' ', '_')}.png"))
    #     plt.close()
    
    # Combined plot for all files
    plt.figure(figsize=(10, 6))
    for (label, df), color in zip(dfs_dict.items(), colors):
        plt.plot(df.index, df['vel_smooth'], label=label, color=color)
    plt.title("Velocity Comparison Over Training Stages")
    plt.xlabel("Frame")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "velocity_all.png"))
    plt.close()

# Perform analysis for each file
total_times = {}
for label, df in dfs.items():
    print(f"\n=== Analysis for {label} ===")
    total_time, _ = calculate_time_elapsed(df, label)
    total_times[label] = total_time
    analyze_cte(df, label, total_time)
    calculate_mean_velocity(df, label)
    calculate_track_completion(df, label)

# Trim DataFrames for plotting (removing first 100 and last 100 rows)
trimmed_dfs = {label: df[100:-100].copy() for label, df in dfs.items()}

# Generate and save smoothed plots with a window size of 50 (adjustable)
window_size = 150
plot_steering_diff(trimmed_dfs, window_size)
plot_throttle(trimmed_dfs, window_size)
plot_velocity(trimmed_dfs, window_size)

print(f"\nPlots have been saved to the directory: {plot_dir}")
