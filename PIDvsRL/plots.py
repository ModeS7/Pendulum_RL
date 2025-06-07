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
    column_names = [col.strip() for col in column_names]  # Strip whitespace from column names

    # Create a new DataFrame to hold the parsed data
    parsed_data = []

    # Process each row starting from the second row (index 1)
    for i in range(1, df_raw.shape[0]):
        row = df_raw.iloc[i, 0]
        if isinstance(row, str) and ',' in row:
            # Split by comma
            values = row.split(',')
            values = [val.strip() for val in values]  # Strip whitespace from values

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

    # Use Step values directly for Time
    if 'Step' in df.columns:
        # Just subtract the first Step to make time start at zero
        df['Time'] = (df['Step'] - df['Step'].iloc[0]) / 1000.0

    # Convert u_sat to Voltage for xlsx files
    # This assumes 'Voltage' column already exists but contains u_sat values
    if 'Voltage' in df.columns and file_path.endswith(('.xlsx', '.xls')):
        df['Voltage'] = df['Voltage'] * (8.4 * 0.095 * 0.085) / 0.042

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


def reset_time_axis(df):
    """
    Reset the time axis to start from 0.0 seconds

    Args:
        df: DataFrame with QUBE data

    Returns:
        DataFrame with Time reset to start from 0.0
    """
    if df is None or df.empty:
        return df

    # If we have dt column with valid values, create time using cumulative sum of dt
    if 'dt' in df.columns and df['dt'].notna().all() and (df['dt'] > 0).all():
        # Calculate cumulative sum of dt to get actual time points
        df['Time'] = df['dt'].cumsum()
    # Otherwise create Time column from Step if not already present
    elif 'Time' not in df.columns and 'Step' in df.columns:
        # Convert Step to seconds
        df['Time'] = df['Step'].astype(float) / 1000.0
    elif 'Time' not in df.columns:
        # If neither Step nor Time columns are found, create a time vector based on index
        df['Time'] = np.arange(len(df)) * 0.01  # Assuming 10ms steps

    # Reset time to start from 0.0
    if 'Time' in df.columns and not df.empty:
        start_time = df['Time'].iloc[0]
        df['Time'] = df['Time'] - start_time

    return df


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


def plot_qube_data(file_path, save_plot=False, output_dir=None, filter_mode=False):
    """
    Plots data from a QUBE data file with normalized pendulum and arm angles on top subplot,
    and voltage on bottom subplot. Can optionally filter to only show Balance mode data.

    Args:
        file_path: Path to the data file (Excel or CSV)
        save_plot: Boolean indicating whether to save the plot to file
        output_dir: Directory to save plots to (if save_plot is True)
        filter_mode: Boolean indicating whether to filter for Balance mode only
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

    # Reset time axis to start from 0.0 regardless of original Step values
    #df = reset_time_axis(df)

    # Filter to only include Balance mode data if requested
    if filter_mode:
        df_filtered = filter_balance_mode(df)
        # Only use filtered data if it's not empty
        if not df_filtered.empty:
            df = df_filtered
        else:
            print(f"No Balance mode data found, using all data instead.")

    # Print column names to verify
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Data points: {len(df)}")
    print(f"Time range: {df['Time'].min()} to {df['Time'].max()} seconds")

    # Normalize angles from degrees to radians in range [-π, π]
    if 'PendulumAngle' in df.columns:
        # Convert from degrees to radians
        pendulum_radians = np.radians(df['PendulumAngle'])
        # Normalize to [-π, π]
        df['NormalizedPendulumAngle'] = np.array([normalize_angle(angle + np.pi) for angle in pendulum_radians])

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

    # No title
    # fig.suptitle(f"{file_name_no_ext} - QUBE Data Plot", fontsize=14)

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

    # Plot voltage on bottom subplot with RED color
    if 'Voltage' in df.columns:
        axs[1].plot(df['Time'], df['Voltage'], 'r-', linewidth=1.5, label='Voltage (V)')

    axs[1].set_xlabel('Time (seconds)', fontsize=12)
    axs[1].set_ylabel('Voltage (V)', fontsize=12)
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    # Ensure x-axis starts at 0
    axs[1].set_xlim(left=0)

    # No mode annotation

    # Adjust spacing between subplots
    plt.tight_layout()
    # plt.subplots_adjust(top=0.92)  # No title adjustment needed

    # Save plot if requested
    if save_plot:
        if output_dir is None:
            output_dir = os.path.dirname(file_path)
        os.makedirs(output_dir, exist_ok=True)

        mode_suffix = "_balance_mode" if filter_mode else ""
        output_path = os.path.join(output_dir, f"{file_name_no_ext}{mode_suffix}_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    return fig, axs


def overlay_plots(file_paths, save_plot=False, output_dir=None, title="Data Comparison", filter_mode=False):
    """
    Create a single plot with multiple datasets overlaid for comparison.
    Can optionally filter to only show Balance mode data.

    Args:
        file_paths: List of paths to the data files (Excel or CSV)
        save_plot: Boolean indicating whether to save the plot to file
        output_dir: Directory to save plots to (if save_plot is True)
        title: Title for the plot
        filter_mode: Boolean indicating whether to filter for Balance mode only
    """
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # No title
    # fig.suptitle(title, fontsize=14)

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

        # Reset time axis to start from 0.0 regardless of original Step values
        #df = reset_time_axis(df)

        # Filter to only include Balance mode data if requested
        if filter_mode:
            df_filtered = filter_balance_mode(df)
            # Only use filtered data if it's not empty
            if not df_filtered.empty:
                df = df_filtered
            else:
                print(f"No Balance mode data found in {file_path}, using all data instead.")

        # Normalize angles from degrees to radians in range [-π, π]
        if 'PendulumAngle' in df.columns:
            # Convert from degrees to radians
            pendulum_radians = np.radians(df['PendulumAngle'])
            # Normalize to [-π, π]
            df['NormalizedPendulumAngle'] = np.array([normalize_angle(angle + np.pi) for angle in pendulum_radians])

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

    # Ensure x-axis starts at 0
    axs[1].set_xlim(left=0)

    # No mode annotation

    # Adjust spacing between subplots
    plt.tight_layout()
    # plt.subplots_adjust(top=0.92)  # No title adjustment needed

    # Save plot if requested
    if save_plot:
        if output_dir is None:
            output_dir = os.path.dirname(file_paths[0])
        os.makedirs(output_dir, exist_ok=True)

        mode_suffix = "_balance_mode" if filter_mode else ""
        output_path = os.path.join(output_dir, f"{title.replace(' ', '_')}{mode_suffix}_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {output_path}")

    return fig, axs


def process_directory(directory_path, save_plots=False, output_dir=None, create_overlay=False, filter_mode=False):
    """
    Process all data files (Excel or CSV) in a directory

    Args:
        directory_path: Path to directory containing data files
        save_plots: Boolean indicating whether to save plots
        output_dir: Directory to save plots to (if save_plots is True)
        create_overlay: Whether to create an overlay plot comparing all files
        filter_mode: Boolean indicating whether to filter for Balance mode only
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
            fig, _ = plot_qube_data(file_path, save_plot=save_plots, output_dir=output_dir, filter_mode=filter_mode)
            if fig is not None:
                plt.close(fig)  # Close the figure to free memory
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Create overlay plot if requested
    if create_overlay and len(files) > 1:
        try:
            # Create overlay for all files
            mode_str = "Balance Mode" if filter_mode else "Data"
            fig, _ = overlay_plots(files, save_plot=save_plots, output_dir=output_dir,
                                   title=f"{mode_str} Comparison", filter_mode=filter_mode)
            if fig is not None and not save_plots:
                plt.show()
            else:
                plt.close(fig)

        except Exception as e:
            print(f"Error creating overlay plot: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot QUBE data from CSV or Excel files')
    parser.add_argument('path', help='Path to data file or directory containing data files')
    parser.add_argument('--save', action='store_true', help='Save plots to file')
    parser.add_argument('--output', help='Directory to save plots to')
    parser.add_argument('--overlay', action='store_true', help='Create overlay plot comparing all files')
    parser.add_argument('--filter', action='store_true', help='Filter data to only include Balance mode')

    args = parser.parse_args()

    if os.path.isdir(args.path):
        # Process all files in directory
        process_directory(args.path, save_plots=args.save, output_dir=args.output,
                          create_overlay=args.overlay, filter_mode=args.filter)
    else:
        # Process single file
        fig, axs = plot_qube_data(args.path, save_plot=args.save, output_dir=args.output, filter_mode=args.filter)
        if fig is not None:
            plt.show()