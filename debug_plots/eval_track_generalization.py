import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define categories and their file paths (replace placeholders with actual paths)
categories = {
    "RC1": [
        "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-06_12-40-38_RC1_1_445_evaltrack_2.csv",
        "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-06_12-49-32_RC1_1_445_evaltrack_3.csv",
        "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-06_13-03-35_RC1_1_445_evaltrack_5.csv"

    ],
    "RC2": [
        "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-06_12-37-35_RC2_1_425_evaltrack_2.csv",
        "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-06_12-50-23_RC2_1_425_evaltrack_3.csv",
        "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-06_13-04-45_RC2_1_425_evaltrack_5.csv"

    ],
    "Final Reward": [
        "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-06_12-43-37_final1_495_evaltrack_2.csv",
        "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-06_12-47-33_final1_495_evaltrack_3.csv",  # Replace with actual path
        "/home/smukh039/work/QRSAC/data/qrsac-donkey-generated-roads-normal-iqn-neutral/data_log_2025-09-06_13-01-36_final1_495_evaltrack_5.csv"  # Replace with actual path

    ]
    
}

# Directory to save plots
plot_dir = "eval_plot_images"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Colors for the 3 categories
colors = ['blue', 'red', 'green']

# Function to calculate time elapsed per frame and total time elapsed
def calculate_time_elapsed(df):
    df['time_per_frame'] = df['distance'] / df['vel']
    df['time_per_frame'] = df['time_per_frame'].replace([np.inf, -np.inf, np.nan], 0)
    total_time = df['time_per_frame'].sum()
    return total_time, df['time_per_frame']

# Function to analyze CTE <= -1.0
def analyze_cte(df, total_time):
    cte_condition = df['cte'] <= -1.0
    num_times_cte = cte_condition.sum()
    time_cte = df[cte_condition]['time_per_frame'].sum() if 'time_per_frame' in df else 0
    percent_cte = (time_cte / total_time) * 100 if total_time > 0 else 0
    return num_times_cte, time_cte, percent_cte

# Function to calculate mean velocity
def calculate_mean_velocity(df):
    mean_vel = df['vel'].mean()
    return mean_vel

# Function to calculate mean absolute steer change
def calculate_mean_abs_steer_change(df):
    df['steer_diff'] = df['action_steer'].diff().fillna(0)
    mean_abs_steer_change = np.abs(df['steer_diff']).mean()
    return mean_abs_steer_change

# Function to smooth data using a moving average
def smooth_data(data, window_size=5):
    return data.rolling(window=window_size, center=True).mean().fillna(data)

# Function to smoothen numpy array (for mean and std in plots)
def smoothen(arr, smoothing_window=70):
    smoothed_arr = (pd.Series(arr)
        .rolling(smoothing_window, min_periods=smoothing_window)
        .mean()
        )
    return smoothed_arr.values

# Function to calculate percentage of track completed
def calculate_track_completion(df):
    total_distance = df['distance'].abs().sum()
    track_completion = (total_distance / 80.0) * 100
    return track_completion

# Function to process data for all categories and plot combined mean/std curves
def process_and_plot_all_categories(window_size=50, smoothing_window=70):
    # Dictionaries to store mean and std data for plots for each category
    steer_diff_means_stds = {}
    throttle_means_stds = {}
    velocity_means_stds = {}
    
    for category, file_paths in categories.items():
        print(f"\n=== Processing Category: {category} ===")
        # Read DataFrames for this category
        dfs = {}
        for i, path in enumerate(file_paths):
            try:
                dfs[f"File {i+1} ({category})"] = pd.read_csv(path)
            except FileNotFoundError:
                print(f"File not found: {path}. Skipping.")
                continue
        
        if not dfs:
            print(f"No files loaded for {category}. Skipping.")
            continue
        
        # Perform analysis for each file and collect metrics
        total_times = []
        num_times_ctes = []
        time_ctes = []
        percent_ctes = []
        mean_vels = []
        track_completions = []
        mean_abs_steer_changes = []
        
        print(f"\n--- Individual File Analysis for {category} ---")
        for label, df in dfs.items():
            total_time, _ = calculate_time_elapsed(df)
            total_times.append(total_time)
            print(f"{label} - Total Time Elapsed: {total_time:.4f} seconds")
            
            num_times_cte, time_cte, percent_cte = analyze_cte(df, total_time)
            num_times_ctes.append(num_times_cte)
            time_ctes.append(time_cte)
            percent_ctes.append(percent_cte)
            print(f"{label} - Number of times CTE <= -1.0: {num_times_cte}")
            print(f"{label} - Total time with CTE <= -1.0: {time_cte:.4f} seconds")
            print(f"{label} - Percentage of time with CTE <= -1.0: {percent_cte:.2f}%")
            
            mean_vel = calculate_mean_velocity(df)
            mean_vels.append(mean_vel)
            print(f"{label} - Mean Velocity: {mean_vel:.4f}")
            
            track_completion = calculate_track_completion(df)
            track_completions.append(track_completion)
            print(f"{label} - Percentage of Track Completed: {track_completion:.2f}%")
            
            mean_abs_steer_change = calculate_mean_abs_steer_change(df)
            mean_abs_steer_changes.append(mean_abs_steer_change)
            print(f"{label} - Mean Absolute Steer Change: {mean_abs_steer_change:.4f}")
        
        # Calculate and print mean and std for the metrics across all files in this category
        print(f"\n--- Aggregated Metrics for {category} (Mean ± Std across all files) ---")
        print(f"Total Time Elapsed: {np.mean(total_times):.4f} ± {np.std(total_times):.4f} seconds")
        print(f"Number of times CTE <= -1.0: {np.mean(num_times_ctes):.2f} ± {np.std(num_times_ctes):.2f}")
        print(f"Total time with CTE <= -1.0: {np.mean(time_ctes):.4f} ± {np.std(time_ctes):.4f} seconds")
        print(f"Percentage of time with CTE <= -1.0: {np.mean(percent_ctes):.2f}% ± {np.std(percent_ctes):.2f}%")
        print(f"Mean Velocity: {np.mean(mean_vels):.4f} ± {np.std(mean_vels):.4f}")
        print(f"Percentage of Track Completed: {np.mean(track_completions):.2f}% ± {np.std(track_completions):.2f}%")
        print(f"Mean Absolute Steer Change: {np.mean(mean_abs_steer_changes):.4f} ± {np.std(mean_abs_steer_changes):.4f}")
        
        # Trim DataFrames for plotting (removing first 100 and last 100 rows if length allows)
        trimmed_dfs = {label: df[100:-100].copy() for label, df in dfs.items() if len(df) > 200}
        
        if not trimmed_dfs:
            print(f"No data after trimming for {category}. Skipping plots.")
            continue
        
        # Process Steering Difference for combined plot
        steer_diff_data = []
        for label, df in trimmed_dfs.items():
            df['steer_diff'] = df['action_steer'].diff().fillna(0)
            df['steer_diff_smooth'] = smooth_data(np.abs(df['steer_diff']), window_size)
            steer_diff_data.append(df['steer_diff_smooth'].values)
        
        if steer_diff_data:
            min_length = min(len(run) for run in steer_diff_data)
            steer_diff_data = [run[:min_length] for run in steer_diff_data]
            steer_diff_array = np.array(steer_diff_data)
            steer_diff_mean = np.mean(steer_diff_array, axis=0)
            steer_diff_std = np.std(steer_diff_array, axis=0)
            steer_diff_mean = smoothen(steer_diff_mean, smoothing_window)
            steer_diff_std = smoothen(steer_diff_std, smoothing_window)
            steer_diff_means_stds[category] = (steer_diff_mean, steer_diff_std, min_length)
        
        # Process Throttle for combined plot
        throttle_data = []
        for label, df in trimmed_dfs.items():
            df['throttle_smooth'] = smooth_data(df['action_throttle'], window_size)
            throttle_data.append(df['throttle_smooth'].values)
        
        if throttle_data:
            min_length = min(len(run) for run in throttle_data)
            throttle_data = [run[:min_length] for run in throttle_data]
            throttle_array = np.array(throttle_data)
            throttle_mean = np.mean(throttle_array, axis=0)
            throttle_std = np.std(throttle_array, axis=0)
            throttle_mean = smoothen(throttle_mean, smoothing_window)
            throttle_std = smoothen(throttle_std, smoothing_window)
            throttle_means_stds[category] = (throttle_mean, throttle_std, min_length)
        
        # Process Velocity for combined plot
        velocity_data = []
        for label, df in trimmed_dfs.items():
            df['vel_smooth'] = smooth_data(df['vel'], window_size)
            velocity_data.append(df['vel_smooth'].values)
        
        if velocity_data:
            min_length = min(len(run) for run in velocity_data)
            velocity_data = [run[:min_length] for run in velocity_data]
            velocity_array = np.array(velocity_data)
            velocity_mean = np.mean(velocity_array, axis=0)
            velocity_std = np.std(velocity_array, axis=0)
            velocity_mean = smoothen(velocity_mean, smoothing_window)
            velocity_std = smoothen(velocity_std, smoothing_window)
            velocity_means_stds[category] = (velocity_mean, velocity_std, min_length)
    
   

# Process and plot for all categories
process_and_plot_all_categories()

print(f"\nPlots have been saved to the directory: {plot_dir}")
