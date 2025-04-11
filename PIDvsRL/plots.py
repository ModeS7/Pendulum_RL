import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import re
import copy


def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    return angle - 2 * np.pi if angle > np.pi else angle


def break_angle_jumps(time, angle, threshold=np.pi):
    """
    Breaks lines at points where angle jumps by more than the threshold.

    Args:
        time: Array of time values
        angle: Array of angle values
        threshold: Angle difference threshold (default: π)

    Returns:
        Tuple of (time_with_breaks, angle_with_breaks) where breaks are represented by NaN values
    """
    # Make copies to avoid modifying the originals
    time_with_breaks = copy.deepcopy(time)
    angle_with_breaks = copy.deepcopy(angle)

    # Find jumps larger than threshold
    jumps = np.abs(np.diff(angle))
    jump_indices = np.where(jumps > threshold)[0]

    # Insert NaN values at jump points to break the line
    if len(jump_indices) > 0:
        # Create new arrays with NaNs inserted at jump points
        # We need to expand the arrays to accommodate the new NaN values
        new_length = len(time) + len(jump_indices)
        new_time = np.zeros(new_length)
        new_angle = np.zeros(new_length)

        # Insert original values and NaNs at appropriate positions
        insert_pos = 0
        for i in range(len(time)):
            new_time[insert_pos] = time[i]
            new_angle[insert_pos] = angle[i]
            insert_pos += 1

            # Check if we need to insert a NaN after this point
            if i in jump_indices:
                new_time[insert_pos] = time[i] + (time[i + 1] - time[i]) / 2
                new_angle[insert_pos] = np.nan
                insert_pos += 1

        return new_time[:insert_pos], new_angle[:insert_pos]

    return time_with_breaks, angle_with_breaks


def process_csv_in_excel(file_path):
    """
    Process Excel files that contain CSV data in a single column.

    Args:
        file_path: Path to the Excel file

    Returns:
        DataFrame with properly parsed data
    """
    # Read the Excel file with a single column
    df_raw = pd.read_excel(file_path, header=None)

    # Get the header from the first row
    if df_raw.shape[0] == 0:
        return None

    header_row = df_raw.iloc[0, 0]
    column_names = header_row.split(',')

    # Create a new DataFrame to hold the parsed data
    parsed_data = []

    # Process each row starting from the second row (index 1)
    for i in range(1, df_raw.shape[0]):
        row = df_raw.iloc[i, 0]
        if isinstance(row, str) and ',' in row:
            # Split by comma
            values = row.split(',')

            # Make sure we have the right number of values
            if len(values) >= len(column_names):
                parsed_data.append(values[:len(column_names)])
            else:
                # If we don't have enough values, pad with NaN
                parsed_data.append(values + [np.nan] * (len(column_names) - len(values)))

    # Create a new DataFrame with the parsed data
    df = pd.DataFrame(parsed_data, columns=column_names)

    # Convert columns to appropriate types
    for col in df.columns:
        # Skip conversion for Mode column
        if col.strip() == 'Mode':
            continue

        # Try to convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def read_csv_file(file_path):
    """
    Read a standard CSV file with QUBE data

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with parsed data
    """
    try:
        # Try reading with pandas directly
        df = pd.read_csv(file_path)

        # Convert columns to appropriate types
        for col in df.columns:
            # Skip conversion for Mode column
            if col.strip() == 'Mode':
                continue

            # Try to convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return None


def filter_balance_mode(df):
    """
    Filter data to only include rows where Mode is 'Balance'.
    Also resets the time axis to start from when Balance mode begins.

    Args:
        df: DataFrame with QUBE data

    Returns:
        Filtered DataFrame with only Balance mode data
    """
    if df is None or df.empty:
        return df

    # Check if 'Mode' column exists
    if 'Mode' not in df.columns:
        print("Warning: 'Mode' column not found in data. Cannot filter for Balance mode.")
        return df

    # Filter for Balance mode
    balance_df = df[df['Mode'].str.strip() == 'Balance'].copy()

    # Check if we have any Balance mode data
    if balance_df.empty:
        print("Warning: No 'Balance' mode data found in this file.")
        return balance_df

    # Reset time to start from when Balance mode begins
    if 'Time' in balance_df.columns:
        start_time = balance_df['Time'].iloc[0]
        balance_df['Time'] = balance_df['Time'] - start_time

    return balance_df


def plot_qube_data(file_path, save_plot=False, output_dir=None):
    """
    Plots data from a QUBE data file with normalized pendulum and arm angles on top subplot,
    and voltage on bottom subplot. Only shows data from Balance mode.

    Args:
        file_path: Path to the data file (Excel or CSV)
        save_plot: Boolean indicating whether to save the plot to file
        output_dir: Directory to save plots to (if save_plot is True)
    """
    print(f"Processing {os.path.basename(file_path)}...")

    # Check file type and process accordingly
    if file_path.endswith(('.xlsx', '.xls')):
        # Process Excel file containing CSV data
        df = process_csv_in_excel(file_path)
    elif file_path.endswith('.csv'):
        # Process direct CSV file
        df = read_csv_file(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return None, None

    if df is None or df.empty:
        print(f"No data found in {file_path}")
        return None, None

    # Create Time column from Step if not already present
    if 'Time' not in df.columns and 'Step' in df.columns:
        df['Time'] = df['Step'].astype(float) / 1000.0  # Convert milliseconds to seconds
    elif 'Time' not in df.columns:
        # If neither Step nor Time columns are found, create a time vector based on index
        df['Time'] = np.arange(len(df)) * 0.01  # Assuming 10ms steps

    # Filter to only include Balance mode data
    #df = filter_balance_mode(df)

    if df.empty:
        print(f"No Balance mode data found in {file_path}")
        return None, None

    # Print column names to verify
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Balance mode data points: {len(df)}")

    # Normalize angles from degrees to radians in range [-π, π]
    if 'PendulumAngle' in df.columns:
        # Convert from degrees to radians
        pendulum_radians = np.radians(df['PendulumAngle'])
        # Normalize to [-π, π]
        df['NormalizedPendulumAngle'] = np.array([normalize_angle(angle+np.pi) for angle in pendulum_radians])

    if 'MotorPosition' in df.columns:
        # Convert from degrees to radians
        motor_radians = np.radians(df['MotorPosition'])
        # Normalize to [-π, π]
        df['NormalizedMotorPosition'] = np.array([normalize_angle(angle) for angle in motor_radians])

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Extract filename for title
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]

    # Plot normalized pendulum angle and motor position on top subplot
    if 'NormalizedPendulumAngle' in df.columns:
        # Break lines at large angle jumps
        time_with_breaks, angle_with_breaks = break_angle_jumps(
            df['Time'].values, df['NormalizedPendulumAngle'].values)
        axs[0].plot(time_with_breaks, angle_with_breaks, 'b-', linewidth=1.5,
                    label='Pendulum Angle (rad)')

    if 'NormalizedMotorPosition' in df.columns:
        # Break lines at large angle jumps for motor position too
        time_with_breaks, angle_with_breaks = break_angle_jumps(
            df['Time'].values, df['NormalizedMotorPosition'].values)
        axs[0].plot(time_with_breaks, angle_with_breaks, 'g-', linewidth=1.5,
                    label='Arm Position (rad)')

    # Add reference line at 0 (upright position)
    axs[0].axhline(y=np.pi, color='k', linestyle=':', alpha=0.7, label='Upright Position')
    axs[0].axhline(y=-np.pi, color='k', linestyle=':', alpha=0.7, label='Upright Position')

    # Plot voltage on bottom subplot with RED color
    if 'Voltage' in df.columns:
        axs[1].plot(df['Time'], df['Voltage'], 'r-', linewidth=1.5, label='Voltage (V)')

    # Title and comment have been removed

    axs[0].set_ylabel('Angle (radians)', fontsize=12)
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # Calculate the range of the pendulum angle for auto-scaling
    if 'NormalizedPendulumAngle' in df.columns:
        pend_min = df['NormalizedPendulumAngle'].min()
        pend_max = df['NormalizedPendulumAngle'].max()
        range_pad = (pend_max - pend_min) * 0.1  # Add 10% padding
        axs[0].set_ylim([pend_min - range_pad, pend_max + range_pad])
    else:
        # Default to full -π to π range if no data
        axs[0].set_ylim([-np.pi - 0.2, np.pi + 0.2])

    # Set y-tick positions at important angles
    axs[0].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[0].set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    axs[1].set_xlabel('Time (seconds)', fontsize=12)
    axs[1].set_ylabel('Voltage (V)', fontsize=12)
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    # Subtitle/comment has been removed

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        if output_dir is None:
            output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{file_name_no_ext}_balance_mode_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    return fig, axs


def overlay_plots(file_paths, save_plot=False, output_dir=None, title="Balance Mode Comparison"):
    """
    Create a single plot with multiple datasets overlaid for comparison.
    Only shows data from Balance mode.

    Args:
        file_paths: List of paths to the data files (Excel or CSV)
        save_plot: Boolean indicating whether to save the plot to file
        output_dir: Directory to save plots to (if save_plot is True)
        title: Title for the plot
    """
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Color cycle for different datasets (for pendulum angles)
    angle_colors = ['b', 'g', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown', 'pink']

    # Color cycle for voltage plots (different shades of red)
    voltage_colors = ['r', 'darkred', 'firebrick', 'indianred', 'lightcoral', 'crimson', 'maroon', 'salmon']

    # Keep track of the overall pendulum angle range
    pend_min = float('inf')
    pend_max = float('-inf')

    for i, file_path in enumerate(file_paths):
        print(f"Adding {os.path.basename(file_path)} to overlay plot...")

        # Check file type and process accordingly
        if file_path.endswith(('.xlsx', '.xls')):
            # Process Excel file containing CSV data
            df = process_csv_in_excel(file_path)
        elif file_path.endswith('.csv'):
            # Process direct CSV file
            df = read_csv_file(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            continue

        if df is None or df.empty:
            print(f"No data found in {file_path}")
            continue

        # Create Time column from Step if not already present
        if 'Time' not in df.columns and 'Step' in df.columns:
            df['Time'] = df['Step'].astype(float) / 1000.0  # Convert milliseconds to seconds
        elif 'Time' not in df.columns:
            # If neither Step nor Time columns are found, create a time vector based on index
            df['Time'] = np.arange(len(df)) * 0.01  # Assuming 10ms steps

        # Filter to only include Balance mode data
        #df = filter_balance_mode(df)

        if df.empty:
            print(f"No Balance mode data found in {file_path}")
            continue

        # Normalize angles from degrees to radians in range [-π, π]
        if 'PendulumAngle' in df.columns:
            # Convert from degrees to radians
            pendulum_radians = np.radians(df['PendulumAngle'])
            # Normalize to [-π, π]
            df['NormalizedPendulumAngle'] = np.array([normalize_angle(angle+np.pi) for angle in pendulum_radians])

            # Update overall min/max
            curr_min = df['NormalizedPendulumAngle'].min()
            curr_max = df['NormalizedPendulumAngle'].max()
            if curr_min < pend_min:
                pend_min = curr_min
            if curr_max > pend_max:
                pend_max = curr_max

        # Extract filename for legend
        file_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]

        # Use the filename as the label
        label = file_name_no_ext

        # Get colors for this dataset
        angle_color = angle_colors[i % len(angle_colors)]
        voltage_color = voltage_colors[i % len(voltage_colors)]

        # Plot normalized pendulum angle on top subplot
        if 'NormalizedPendulumAngle' in df.columns:
            # Break lines at large angle jumps
            time_with_breaks, angle_with_breaks = break_angle_jumps(
                df['Time'].values, df['NormalizedPendulumAngle'].values)
            axs[0].plot(time_with_breaks, angle_with_breaks, color=angle_color, linewidth=1.5,
                        label=label)

        # Plot voltage on bottom subplot (using red shades)
        if 'Voltage' in df.columns:
            axs[1].plot(df['Time'], df['Voltage'], color=voltage_color, linewidth=1.5,
                        label=label)

    # Add reference line at 0 (upright position)
    axs[0].axhline(y=0, color='k', linestyle=':', alpha=0.7, label='Upright Position')

    # Title has been removed
    axs[0].set_ylabel('Pendulum Angle (radians)', fontsize=12)
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # Set y limits based on data range if valid
    if pend_min != float('inf') and pend_max != float('-inf'):
        range_pad = (pend_max - pend_min) * 0.1  # Add 10% padding
        axs[0].set_ylim([pend_min - range_pad, pend_max + range_pad])
    else:
        # Default to full -π to π range if no valid data
        axs[0].set_ylim([-np.pi - 0.2, np.pi + 0.2])

    # Set y-tick positions at important angles
    axs[0].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[0].set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    axs[1].set_xlabel('Time (seconds)', fontsize=12)
    axs[1].set_ylabel('Voltage (V)', fontsize=12)
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save plot if requested
    if save_plot:
        if output_dir is None:
            output_dir = os.path.dirname(file_paths[0])
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_balance_mode_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")

    return fig, axs


def process_directory(directory_path, save_plots=False, output_dir=None, create_overlay=False):
    """
    Process all data files (Excel or CSV) in a directory

    Args:
        directory_path: Path to directory containing data files
        save_plots: Boolean indicating whether to save plots
        output_dir: Directory to save plots to (if save_plots is True)
        create_overlay: Whether to create an overlay plot comparing all files
    """
    if output_dir is None:
        output_dir = os.path.join(directory_path, 'plots')

    os.makedirs(output_dir, exist_ok=True)

    # Find all data files (Excel and CSV)
    files = []
    for file in os.listdir(directory_path):
        if file.endswith(('.xlsx', '.xls', '.csv')):
            files.append(os.path.join(directory_path, file))

    print(f"Found {len(files)} data files")

    # Process each file
    for file_path in files:
        try:
            fig, _ = plot_qube_data(file_path, save_plot=save_plots, output_dir=output_dir)
            if fig is not None:
                plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Create overlay plot if requested
    if create_overlay and len(files) > 1:
        try:
            # Create overlay for all files
            fig, _ = overlay_plots(files, save_plot=save_plots, output_dir=output_dir, title="Balance Mode Comparison")
            if fig is not None and not save_plots:
                plt.show()
            else:
                plt.close(fig)

        except Exception as e:
            print(f"Error creating overlay plot: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot QUBE data from CSV or Excel files - Balance mode only.')
    parser.add_argument('path', help='Path to data file or directory containing data files')
    parser.add_argument('--save', action='store_true', help='Save plots to file')
    parser.add_argument('--output', help='Directory to save plots to')
    parser.add_argument('--overlay', action='store_true', help='Create overlay plot comparing all files')

    args = parser.parse_args()

    if os.path.isdir(args.path):
        # Process all files in directory
        process_directory(args.path, save_plots=args.save, output_dir=args.output, create_overlay=args.overlay)
    else:
        # Process single file
        fig, axs = plot_qube_data(args.path, save_plot=args.save, output_dir=args.output)
        if fig is not None:
            plt.show()