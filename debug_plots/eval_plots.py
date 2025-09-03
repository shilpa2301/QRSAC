import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File paths for the two CSV files (adjust these paths based on your actual file locations)
file1_path = "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-01_17-49-39_final1_485.csv"
file2_path = "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-01_18-30-10_RC1_470.csv"

# Directory to save plots
plot_dir = "eval_plot_images"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Read the CSV files into DataFrames
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Function to calculate time elapsed per frame and total time elapsed
def calculate_time_elapsed(df, label):
    # Calculate time per frame as distance of the current frame divided by velocity of the current frame
    df['time_per_frame'] = df['distance'] / df['vel']
    # Replace NaN or infinite values (if vel is 0) with 0
    df['time_per_frame'] = df['time_per_frame'].replace([np.inf, -np.inf, np.nan], 0)
    # Calculate total time elapsed as sum of time per frame
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

# Function to plot steering differences with smoothing
def plot_steering_diff(df1, df2, label1, label2, window_size=5):
    # df1 = df1[400:-400].copy()
    # df2 = df2[400:-400].copy()


    df1['steer_diff'] = df1['action_steer'].diff().fillna(0)
    df2['steer_diff'] = df2['action_steer'].diff().fillna(0)
    df1['steer_diff_smooth'] = smooth_data(np.abs(df1['steer_diff']), window_size)
    df2['steer_diff_smooth'] = smooth_data(np.abs(df2['steer_diff']), window_size)
    
    # Individual plot for File 1
    plt.figure(figsize=(10, 6))
    plt.ylim([0, 0.025])
    plt.plot(df1.index, np.abs(df1['steer_diff_smooth']), label=label1, color='blue')
    plt.title(f"Steering Difference (Smoothed) - {label1}")
    plt.xlabel("Frame (Row Number)")
    plt.ylabel("Steering Difference")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"steering_diff_{label1.replace(' ', '_')}.png"))
    plt.close()
    
    # Individual plot for File 2
    plt.figure(figsize=(10, 6))
    plt.ylim([0, 0.025])
    plt.plot(df2.index, np.abs(df2['steer_diff_smooth']), label=label2, color='red')
    plt.title(f"Steering Difference (Smoothed) - {label2}")
    plt.xlabel("Frame (Row Number)")
    plt.ylabel("Steering Difference")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"steering_diff_{label2.replace(' ', '_')}.png"))
    plt.close()
    
    # Combined plot
    plt.figure(figsize=(10, 6))
    plt.ylim([0, 1.0])
    plt.plot(df1.index, np.abs(df1['steer_diff_smooth']), label=label1, color='blue')
    plt.plot(df2.index, np.abs(df2['steer_diff_smooth']), label=label2, color='red')
    plt.title("Steering Difference (Smoothed) - Both Files")
    plt.xlabel("Frame (Row Number)")
    plt.ylabel("Steering Difference")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "steering_diff_both.png"))
    plt.close()

# Function to plot throttle with smoothing
def plot_throttle(df1, df2, label1, label2, window_size=5):
    df1['throttle_smooth'] = smooth_data(df1['action_throttle'], window_size)
    df2['throttle_smooth'] = smooth_data(df2['action_throttle'], window_size)
    
    # Individual plot for File 1
    plt.figure(figsize=(10, 6))
    plt.plot(df1.index, df1['throttle_smooth'], label=label1, color='blue')
    plt.title(f"Throttle (Smoothed) - {label1}")
    plt.xlabel("Frame (Row Number)")
    plt.ylabel("Throttle")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"throttle_{label1.replace(' ', '_')}.png"))
    plt.close()
    
    # Individual plot for File 2
    plt.figure(figsize=(10, 6))
    plt.plot(df2.index, df2['throttle_smooth'], label=label2, color='red')
    plt.title(f"Throttle (Smoothed) - {label2}")
    plt.xlabel("Frame (Row Number)")
    plt.ylabel("Throttle")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"throttle_{label2.replace(' ', '_')}.png"))
    plt.close()
    
    # Combined plot
    plt.figure(figsize=(10, 6))
    plt.plot(df1.index, df1['throttle_smooth'], label=label1, color='blue')
    plt.plot(df2.index, df2['throttle_smooth'], label=label2, color='red')
    plt.title("Throttle (Smoothed) - Both Files")
    plt.xlabel("Frame (Row Number)")
    plt.ylabel("Throttle")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "throttle_both.png"))
    plt.close()

# Function to plot velocity with smoothing
def plot_velocity(df1, df2, label1, label2, window_size=5):
    df1['vel_smooth'] = smooth_data(df1['vel'], window_size)
    df2['vel_smooth'] = smooth_data(df2['vel'], window_size)
    
    # Individual plot for File 1
    plt.figure(figsize=(10, 6))
    plt.plot(df1.index, df1['vel_smooth'], label=label1, color='blue')
    plt.title(f"Velocity (Smoothed) - {label1}")
    plt.xlabel("Frame (Row Number)")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"velocity_{label1.replace(' ', '_')}.png"))
    plt.close()
    
    # Individual plot for File 2
    plt.figure(figsize=(10, 6))
    plt.plot(df2.index, df2['vel_smooth'], label=label2, color='red')
    plt.title(f"Velocity (Smoothed) - {label2}")
    plt.xlabel("Frame (Row Number)")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"velocity_{label2.replace(' ', '_')}.png"))
    plt.close()
    
    # Combined plot
    plt.figure(figsize=(10, 6))
    plt.plot(df1.index, df1['vel_smooth'], label=label1, color='blue')
    plt.plot(df2.index, df2['vel_smooth'], label=label2, color='red')
    plt.title("Velocity (Smoothed) - Both Files")
    plt.xlabel("Frame (Row Number)")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "velocity_both.png"))
    plt.close()

# Function to calculate percentage of track completed
def calculate_track_completion(df, label):
    # Use total distance as the sum of differences in distance column
    total_distance = df['distance'].abs().sum()
    track_completion = (total_distance / 80.0) * 100
    print(f"{label} - Percentage of Track Completed: {track_completion:.2f}%")
    return track_completion

# Labels for the two files
label1 = "Final Reward"
label2 = "RC1"

# Perform analysis for each file
print("=== Analysis for File 1 ===")
total_time1, _ = calculate_time_elapsed(df1, label1)
analyze_cte(df1, label1, total_time1)
calculate_mean_velocity(df1, label1)
calculate_track_completion(df1, label1)

print("\n=== Analysis for File 2 ===")
total_time2, _ = calculate_time_elapsed(df2, label2)
analyze_cte(df2, label2, total_time2)
calculate_mean_velocity(df2, label2)
calculate_track_completion(df2, label2)

# Generate and save smoothed plots with a window size of 5 (adjustable)
window_size = 50
plot_steering_diff(df1, df2, label1, label2, window_size)
plot_throttle(df1, df2, label1, label2, window_size)
plot_velocity(df1, df2, label1, label2, window_size)

print(f"\nPlots have been saved to the directory: {plot_dir}")
