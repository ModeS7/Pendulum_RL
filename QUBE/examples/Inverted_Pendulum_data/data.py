import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot(df):
    """
    Plot function for the original data format.
    """
    plt.figure(figsize=(15, 30))

    # Plot 1: Time vs Angle
    plt.subplot(3, 1, 1)
    plt.scatter(df['time'], df['angle'], marker='o', linestyle='-', color='blue')
    plt.title('Time vs Angle Pendulum')
    plt.xlabel('Time')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)

    # Plot 2: Time vs Position
    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['position'], marker='s', linestyle='-', color='red')
    plt.title('Time vs Position Motor')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.grid(True)

    # Plot 3: Time vs Voltage
    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['voltage'], marker='^', linestyle='-', color='green')
    plt.title('Time vs Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_p_d(df):
    """
    Plot function for the processed data with unwrapped angles.
    """
    plt.figure(figsize=(15, 30))

    plt.subplot(3, 1, 1)
    plt.scatter(df['time'], df['unwrapped_pendulum_angle'], marker='o', linestyle='-', color='blue')
    plt.title('Time vs Angle Pendulum')
    plt.xlabel('Time')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)

    # Plot 2: Time vs Position
    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['position'], marker='s', linestyle='-', color='red')
    plt.title('Time vs Position Motor')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.grid(True)

    # Plot 3: Time vs Voltage
    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['voltage'], marker='^', linestyle='-', color='green')
    plt.title('Time vs Voltage')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_new(df):
    """
    Plot function for the new data format.
    """
    plt.figure(figsize=(15, 30))

    # Plot 1: Time vs Pendulum Angle (from upright)
    plt.subplot(3, 1, 1)
    plt.scatter(df['Time(s)'], df['Pendulum_Angle_From_Upright(deg)'], marker='o', linestyle='-', color='blue')
    plt.title('Time vs Pendulum Angle From Upright')
    plt.xlabel('Time(s)')
    plt.ylabel('Angle (deg)')
    plt.grid(True)

    # Plot 2: Time vs Motor Angle
    plt.subplot(3, 1, 2)
    plt.plot(df['Time(s)'], df['Motor_Angle(deg)'], marker='s', linestyle='-', color='red')
    plt.title('Time vs Motor Angle')
    plt.xlabel('Time(s)')
    plt.ylabel('Angle (deg)')
    plt.grid(True)

    # Plot 3: Time vs Motor Voltage
    plt.subplot(3, 1, 3)
    plt.plot(df['Time(s)'], df['Motor_Voltage(V)'], marker='^', linestyle='-', color='green')
    plt.title('Time vs Motor Voltage')
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def read_new_data(filename):
    """
    Reads the new data file format.
    """
    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(filename)
        elif filename.endswith('.csv'):
            # Try standard CSV format first
            df = pd.read_csv(filename)

            # Check if file was read correctly
            if len(df.columns) == 1 and ',' in df.columns[0]:
                print(f"File appears to be space-delimited. Trying alternative parsing method...")

                # Read the file as text
                with open(filename, 'r') as f:
                    content = f.read()

                # Split by lines and parse manually
                lines = content.strip().split('\n')
                header = lines[0].split()

                # Create data rows
                data = []
                for line in lines[1:]:
                    if line.strip():  # Skip empty lines
                        values = line.split()
                        if len(values) >= len(header):  # Ensure we have enough values
                            data.append(values[:len(header)])  # Trim extra values

                # Create DataFrame
                df = pd.DataFrame(data, columns=header)

                # Convert numeric columns
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
        else:
            raise ValueError("Unsupported file type.")

        print(f"Successfully processed {filename}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def process_excel_csv(filename):
    """
    Process the old Excel/CSV files with comma-delimited data.
    """
    try:
        # Read the Excel file
        excel_file = pd.ExcelFile(filename)
        df_raw = excel_file.parse(sheet_name=0)

        # Extract the column name to get the header
        header = df_raw.columns[0]
        column_names = header.split(',')

        # Create a new DataFrame by splitting the values in each row
        data_rows = []
        for _, row in df_raw.iterrows():
            # Split the string by commas and convert to appropriate types
            values = row[0].split(',')
            data_rows.append(values)

        # Create the properly structured DataFrame
        df = pd.DataFrame(data_rows, columns=column_names)

        # Convert columns to numeric (this will handle strings with decimal points)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        print(f"Successfully processed {filename}")
        return df

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


def convert_angle_to_original(transformed_angle):
    """
    Convert the transformed angle back to the original format.
    """
    if transformed_angle > 0:
        original_angle = transformed_angle + 180
    else:
        original_angle = transformed_angle - 180

    # Ensure angle stays in the range [-180, 180]
    if original_angle > 180:
        original_angle -= 360
    elif original_angle <= -180:
        original_angle += 360

    return original_angle


def process_pendulum_data(df):
    """
    Process the pendulum data to add the original angle column.
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Convert the angle column back to the original reading
    processed_df['pendulum_angle'] = processed_df['angle'].apply(convert_angle_to_original)

    return processed_df


def process_all_dataframes(df_dict):
    """
    Process multiple dataframes at once.
    """
    result_dict = {}
    for name, df in df_dict.items():
        result_dict[name] = process_pendulum_data(df)
        print(f"Processed {name}: Added original_angle column")
    return result_dict


def convert_new_format_to_old(df_new):
    """
    Convert the new data format to the old format for compatibility with existing functions.
    """
    df_old = pd.DataFrame()

    # Map the columns from new format to old format
    df_old['time'] = df_new['Time(s)'].astype(float)
    df_old['position'] = df_new['Motor_Angle(deg)'].astype(float)
    df_old['angle'] = df_new['Pendulum_Angle_From_Upright(deg)'].astype(float)
    df_old['motor_velocity'] = df_new['Motor_Velocity(rad/s)'].astype(float)
    df_old['pendulum_velocity'] = df_new['Pendulum_Velocity(rad/s)'].astype(float)

    # *** IMPORTANT: Ensure voltage is correctly mapped and has proper values ***
    df_old['voltage'] = df_new['Motor_Voltage(V)'].astype(float)

    # Add mode column if Controller_Mode exists
    if 'Controller_Mode' in df_new.columns:
        # Map the mode values or use as is
        df_old['mode'] = df_new['Controller_Mode'].apply(lambda x: 0 if x == 'Emergency' else 1)
    else:
        df_old['mode'] = 0  # Default mode

    # Add energy column (if needed)
    df_old['energy'] = 0  # Default value

    print("Column data types after conversion:")
    print(df_old.dtypes)

    # Verify voltage values
    print(
        f"Voltage statistics - Min: {df_old['voltage'].min()}, Max: {df_old['voltage'].max()}, Mean: {df_old['voltage'].mean()}")

    return df_old


def process_and_save_angles(df, filename="processed_angles.csv"):
    """
    Process and save the angles data, adding an unwrapped version.
    Ensures voltage column is correctly included.
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Make sure pendulum_angle is calculated if needed
    if 'angle' in processed_df.columns and 'pendulum_angle' not in processed_df.columns:
        processed_df['pendulum_angle'] = processed_df['angle'].apply(convert_angle_to_original)

    # Create an unwrapped version of the angle to avoid discontinuities
    # Convert to radians, unwrap, then convert back to degrees
    if 'pendulum_angle' in processed_df.columns:
        angle_rad = np.radians(processed_df['pendulum_angle'])
        unwrapped_angle_rad = np.unwrap(angle_rad)
        processed_df['unwrapped_pendulum_angle'] = np.degrees(unwrapped_angle_rad)

    # Make sure all essential columns are included (especially voltage)
    essential_columns = ['time', 'position', 'voltage']
    if all(col in processed_df.columns for col in essential_columns):
        # Keep the columns we want, ensuring voltage is included
        result_columns = ['time', 'position', 'voltage']

        if 'angle' in processed_df.columns:
            result_columns.append('angle')

        if 'pendulum_angle' in processed_df.columns:
            result_columns.append('pendulum_angle')

        if 'unwrapped_pendulum_angle' in processed_df.columns:
            result_columns.append('unwrapped_pendulum_angle')

        result_df = processed_df[result_columns]

        # Save to CSV
        result_df.to_csv(filename, index=False)
        print(f"Saved processed data to {filename}")
        print(f"Columns in saved file: {result_df.columns.tolist()}")

        # Verify voltage values in processed file
        print(f"Voltage in processed file - Min: {result_df['voltage'].min()}, Max: {result_df['voltage'].max()}")

        return result_df
    else:
        missing_columns = [col for col in essential_columns if col not in processed_df.columns]
        print(f"Warning: Missing essential columns: {missing_columns}")
        print(f"Available columns: {processed_df.columns.tolist()}")
        return processed_df


def process_and_plot_new_data(filename):
    """
    Process and plot data in the new format.
    """
    # Read the new format data
    df_new = read_new_data(filename)

    if df_new is not None:
        # Check if we have the expected columns
        expected_columns = ['Time(s)', 'Motor_Angle(deg)', 'Pendulum_Angle_From_Upright(deg)', 'Motor_Voltage(V)']
        missing_columns = [col for col in expected_columns if col not in df_new.columns]

        if missing_columns:
            print(f"Warning: Missing expected columns: {missing_columns}")
            print(f"Available columns: {df_new.columns.tolist()}")
            return None, None, None

        # Plot directly with the new format
        print("Plotting data in new format...")
        plot_new(df_new)

        # Debug: Check voltage values in the new data
        print(f"Voltage in new data - Min: {df_new['Motor_Voltage(V)'].min()}, "
              f"Max: {df_new['Motor_Voltage(V)'].max()}, "
              f"Mean: {df_new['Motor_Voltage(V)'].mean()}")

        # Convert to old format for compatibility with existing functions
        print("Converting to old format for compatibility...")
        df_old = convert_new_format_to_old(df_new)

        # Process the converted data
        processed_df = process_and_save_angles(df_old, f"processed_{os.path.basename(filename)}")

        # Plot with the processed data
        print("Plotting processed data...")
        plot_p_d(processed_df)

        return df_new, df_old, processed_df
    else:
        print(f"Unable to process {filename}")
        return None, None, None


def main():
    # Dictionary to hold all dataframes
    df_dict = {}
    missing_files = []  # List to track missing files

    # Process each of the new format files
    new_filenames = [
        "qube_data_20250329_111906.csv",
        "qube_data_20250329_113057.csv",
        "qube_data_20250329_122651.csv",
        "qube_data_20250329_132333.csv",
        "qube_data_20250329_132635.csv",
        "qube_data_20250329_145238.csv"
    ]

    for i, filename in enumerate(new_filenames):
        if os.path.exists(filename):
            print(f"\nProcessing file: {filename}")
            # Process and plot the new format file
            df_new, df_old, processed_df = process_and_plot_new_data(filename)

            if df_new is not None and df_old is not None and processed_df is not None:
                # Add to dictionary
                df_dict[f'new_df{i + 1}'] = df_new
                df_dict[f'old_df{i + 1}'] = df_old
                df_dict[f'processed_df{i + 1}'] = processed_df

                # Print information about the processed file that will be used for simulation
                print(f"Created processed file 'processed_{filename}' for simulation")
        else:
            missing_files.append(filename)
            print(f"File {filename} not found")

    # Extract dataframes to individual variables (df1, df2, etc.)
    for name, df in df_dict.items():
        globals()[name] = df

    print(f"\nProcessed {len(df_dict) / 3} files")
    print(f"Missing files: {missing_files}")

    print("\nNOTE: The processed CSV files have been created with the correct voltage values.")
    print("You can now run the simulation with these files.")
    print("Make sure the simulation code correctly reads the 'voltage' column.")


if __name__ == "__main__":
    main()