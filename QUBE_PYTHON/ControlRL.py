import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog, messagebox
from QUBE import QUBE
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import os
import csv

# Update with your COM port
COM_PORT = "COM3"


# Low Pass Filter class (similar to the Arduino implementation)
class LowPassFilter:
    def __init__(self, cutoff_freq=63.0):
        """Initialize a low-pass filter with specified cutoff frequency in Hz"""
        self.twopi = 2.0 * np.pi
        self.wc = cutoff_freq * self.twopi  # Cutoff frequency parameter
        self.y_last = 0.0  # Last output value

    def filter(self, x, dt):
        """Apply filter to input x with timestep dt"""
        # Same equation as in the Arduino code:
        # y_k = y_k_last + wc*dt*(x - y_k_last)
        y_k = self.y_last + self.wc * dt * (x - self.y_last)
        self.y_last = y_k  # Save for next iteration
        return y_k

    def reset(self, initial_value=0.0):
        """Reset the filter state"""
        self.y_last = initial_value


# Actor Network
class Actor(nn.Module):
    """Policy network that outputs action distribution."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log std for continuous action
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.network(state)

        # Get mean and constrain it to [-1, 1]
        action_mean = torch.tanh(self.mean(features))

        # Get log standard deviation and clamp it
        action_log_std = self.log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)

        return action_mean, action_log_std


# Helper functions
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


class QUBEControllerWithRL:
    def __init__(self, master, qube):
        self.master = master
        self.qube = qube
        master.title("QUBE Controller with Reinforcement Learning")

        # Control state
        self.motor_voltage = 0.0
        self.target_position = 0.0
        self.calibrating = False
        self.rl_mode = False
        self.moving_to_position = False
        self.rl_model = None
        self.system_max_voltage = 18.0  # Increased maximum hardware voltage to 18.0V
        self.max_voltage = 5.0  # Default max voltage for RL control (replaces scaling factor)

        # Performance optimization settings - MODIFIED: Optimized UI updates
        self.ui_update_interval = 15  # Increased UI update interval (higher = less UI overhead)
        self.ui_counter = 0
        self.last_loop_time = time.time()
        self.actual_frequency = 0

        # Cache for UI values to avoid unnecessary updates
        self.ui_cache = {
            'motor_angle': 0.0,
            'pendulum_angle': 0.0,
            'rpm': 0.0,
            'voltage': 0.0,
            'frequency': 0.0
        }

        # Performance tracking
        self.iteration_count = 0
        self.start_time = time.time()

        # Data logging variables
        self.logging = False
        self.log_file = None
        self.log_writer = None
        self.log_counter = 0
        self.log_interval = 20  # Log every 20 steps
        self.log_step = 0
        self.log_start_time = 0

        # Pendulum velocity tracking for logging
        self.prev_pendulum_angle = 0
        self.prev_motor_pos = 0
        self.prev_logging_time = time.time()

        # Low-pass filters
        self.filter_cutoff = 500.0  # Default cutoff frequency
        self.pendulum_velocity_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
        self.motor_velocity_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
        self.voltage_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
        self.filtered_voltage = 0.0

        # Initialize the RL model (but don't load weights yet)
        self.initialize_rl_model()

        # Create GUI elements
        self.create_gui()

    def initialize_rl_model(self):
        """Initialize the RL model architecture"""
        state_dim = 6  # Our observation space (same as in training)
        action_dim = 1  # Motor voltage (normalized)

        # Initialize model
        self.actor = Actor(state_dim, action_dim)

        # Set device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor.eval()  # Set to evaluation mode

    def load_rl_model(self):
        """Open file dialog to select the model file"""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select RL Model File",
            filetypes=(("PyTorch Models", "*.pth"), ("All files", "*.*"))
        )

        if filename:
            try:
                # Load the model state dict
                state_dict = torch.load(filename, map_location=self.device)

                # Try loading the model
                try:
                    self.actor.load_state_dict(state_dict)
                    self.actor.to(self.device)
                    self.actor.eval()
                except Exception as e:
                    raise ValueError(f"Failed to load model: {str(e)}")

                # Confirm model loaded
                self.status_label.config(text=f"Model loaded: {os.path.basename(filename)}")
                self.rl_model = filename

                # Update model loaded indication
                self.model_type_label.config(text="Model Status: Loaded")
                self.architecture_label.config(text="Model ready for control")

                # Enable RL control button
                self.rl_control_btn.config(state=tk.NORMAL)

                # Set blue LED to indicate ready
                self.r_slider.set(0)
                self.g_slider.set(0)
                self.b_slider.set(999)

            except Exception as e:
                self.status_label.config(text=f"Error loading model: {str(e)}")
                messagebox.showerror("Load Error", f"Could not load model: {str(e)}")

    def create_gui(self):
        # Main control frame
        control_frame = Frame(self.master, padx=10, pady=10)
        control_frame.pack()

        # Calibrate button
        self.calibrate_btn = Button(control_frame, text="Calibrate (Set Corner as Zero)",
                                    command=self.calibrate,
                                    width=25, height=2)
        self.calibrate_btn.grid(row=0, column=0, padx=5, pady=5)

        # RL Model buttons
        rl_frame = Frame(control_frame)
        rl_frame.grid(row=1, column=0, pady=10)

        self.load_model_btn = Button(rl_frame, text="Load RL Model",
                                     command=self.load_rl_model,
                                     width=15)
        self.load_model_btn.grid(row=0, column=0, padx=5)

        self.rl_control_btn = Button(rl_frame, text="Start RL Control",
                                     command=self.toggle_rl_control,
                                     width=15, state=tk.DISABLED)
        self.rl_control_btn.grid(row=0, column=1, padx=5)

        # Model Status Indicator
        self.model_type_label = Label(rl_frame, text="Model Status: Not Loaded", width=20)
        self.model_type_label.grid(row=0, column=2, padx=5)

        # Max voltage slider
        max_voltage_frame = Frame(control_frame)
        max_voltage_frame.grid(row=2, column=0, pady=5)

        self.max_voltage_slider = Scale(
            max_voltage_frame,
            from_=0.5,
            to=self.system_max_voltage,
            orient=tk.HORIZONTAL,
            label="RL Max Voltage",
            length=300,
            resolution=0.1,
            command=self.set_max_voltage
        )
        self.max_voltage_slider.set(self.max_voltage)
        self.max_voltage_slider.pack(padx=5)

        # Move to position input and button
        position_frame = Frame(control_frame)
        position_frame.grid(row=3, column=0, pady=10)

        Label(position_frame, text="Target Position (degrees):").grid(row=0, column=0, padx=5)
        self.position_entry = Entry(position_frame, width=10)
        self.position_entry.grid(row=0, column=1, padx=5)
        self.position_entry.insert(0, "0.0")

        self.move_btn = Button(position_frame, text="Move to Position",
                               command=self.start_move_to_position, width=15)
        self.move_btn.grid(row=0, column=2, padx=5)

        # DATA LOGGING SECTION
        logging_frame = Frame(control_frame)
        logging_frame.grid(row=4, column=0, pady=10)

        # Logging interval entry
        Label(logging_frame, text="Log Interval (steps):").grid(row=0, column=0, padx=5)
        self.log_interval_entry = Entry(logging_frame, width=5)
        self.log_interval_entry.grid(row=0, column=1, padx=5)
        self.log_interval_entry.insert(0, "20")  # Default to 20 steps

        # Log button
        self.log_btn = Button(logging_frame, text="Start Logging",
                              command=self.toggle_logging, width=15)
        self.log_btn.grid(row=0, column=2, padx=5)

        # Log status
        self.log_status_label = Label(logging_frame, text="Logging: OFF")
        self.log_status_label.grid(row=0, column=3, padx=5)

        # MODIFIED: Replace filter text entry with slider (previously control frequency slider)
        filter_frame = Frame(control_frame)
        filter_frame.grid(row=5, column=0, pady=5)

        # New filter cutoff slider (replacing control frequency slider)
        self.filter_cutoff_slider = Scale(
            filter_frame,
            from_=0,
            to=6000,
            orient=tk.HORIZONTAL,
            label="Filter Cutoff Frequency (Hz)",
            length=300,
            resolution=100,
            command=self.set_filter_cutoff
        )
        self.filter_cutoff_slider.set(self.filter_cutoff)
        self.filter_cutoff_slider.grid(row=0, column=0, padx=5)

        # Filter status
        self.filter_status_label = Label(filter_frame, text=f"Cutoff: {self.filter_cutoff} Hz")
        self.filter_status_label.grid(row=0, column=1, padx=5)

        # Stop button
        self.stop_btn = Button(control_frame, text="STOP MOTOR",
                               command=self.stop_motor,
                               width=20, height=2,
                               bg="red", fg="white")
        self.stop_btn.grid(row=6, column=0, pady=10)

        # Manual voltage control
        self.voltage_slider = Scale(
            control_frame,
            from_=-self.system_max_voltage,
            to=self.system_max_voltage,
            orient=tk.HORIZONTAL,
            label="Manual Voltage",
            length=400,
            resolution=0.1,
            command=self.set_manual_voltage
        )
        self.voltage_slider.set(0)
        self.voltage_slider.grid(row=7, column=0, padx=5, pady=10)

        # Performance settings frame - MODIFIED: Only UI update interval
        perf_frame = Frame(control_frame)
        perf_frame.grid(row=8, column=0, pady=5)

        # Add UI update interval slider with increased range
        self.ui_slider = Scale(
            perf_frame,
            from_=5,
            to=50,
            orient=tk.HORIZONTAL,
            label="UI Update Every N Iterations",
            length=300,
            resolution=5,
            command=self.set_ui_update_interval
        )
        self.ui_slider.set(self.ui_update_interval)
        self.ui_slider.grid(row=0, column=0, padx=5)

        # Status display
        status_frame = Frame(self.master, padx=10, pady=10)
        status_frame.pack()

        Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = Label(status_frame, text="Ready - Please calibrate", width=40)
        self.status_label.grid(row=0, column=1, sticky=tk.W)

        Label(status_frame, text="Model:").grid(row=1, column=0, sticky=tk.W)
        self.model_label = Label(status_frame, text="No model loaded", width=40)
        self.model_label.grid(row=1, column=1, sticky=tk.W)

        Label(status_frame, text="Architecture:").grid(row=2, column=0, sticky=tk.W)
        self.architecture_label = Label(status_frame, text="Not loaded", width=40)
        self.architecture_label.grid(row=2, column=1, sticky=tk.W)

        Label(status_frame, text="Motor Angle:").grid(row=3, column=0, sticky=tk.W)
        self.angle_label = Label(status_frame, text="0.0°")
        self.angle_label.grid(row=3, column=1, sticky=tk.W)

        Label(status_frame, text="Pendulum Angle:").grid(row=4, column=0, sticky=tk.W)
        self.pendulum_label = Label(status_frame, text="0.0°")
        self.pendulum_label.grid(row=4, column=1, sticky=tk.W)

        Label(status_frame, text="Motor RPM:").grid(row=5, column=0, sticky=tk.W)
        self.rpm_label = Label(status_frame, text="0.0")
        self.rpm_label.grid(row=5, column=1, sticky=tk.W)

        Label(status_frame, text="Current Voltage:").grid(row=6, column=0, sticky=tk.W)
        self.voltage_label = Label(status_frame, text="0.0 V")
        self.voltage_label.grid(row=6, column=1, sticky=tk.W)

        Label(status_frame, text="RL Max Voltage:").grid(row=7, column=0, sticky=tk.W)
        self.max_voltage_label = Label(status_frame, text=f"{self.max_voltage:.1f} V")
        self.max_voltage_label.grid(row=7, column=1, sticky=tk.W)

        # Add actual frequency display
        Label(status_frame, text="Control Frequency:").grid(row=8, column=0, sticky=tk.W)
        self.freq_label = Label(status_frame, text="0.0 Hz")
        self.freq_label.grid(row=8, column=1, sticky=tk.W)

        # Add data logging info
        Label(status_frame, text="Log Status:").grid(row=9, column=0, sticky=tk.W)
        self.log_info_label = Label(status_frame, text="Not logging")
        self.log_info_label.grid(row=9, column=1, sticky=tk.W)

        # RGB Control
        rgb_frame = Frame(self.master, padx=10, pady=10)
        rgb_frame.pack()

        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

    # MODIFIED: New method for filter cutoff slider
    def set_filter_cutoff(self, value):
        """Set the cutoff frequency of all filters from slider"""
        try:
            cutoff_freq = float(value)
            if cutoff_freq <= 0:
                cutoff_freq = 100.0  # Minimum value for safety

            # Store current filter values to preserve states
            pendulum_velocity_last = self.pendulum_velocity_filter.y_last
            motor_velocity_last = self.motor_velocity_filter.y_last
            voltage_last = self.voltage_filter.y_last

            # Update internal value
            self.filter_cutoff = cutoff_freq

            # Create new filters with updated cutoff frequency
            self.pendulum_velocity_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
            self.motor_velocity_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)
            self.voltage_filter = LowPassFilter(cutoff_freq=self.filter_cutoff)

            # Transfer the previous states to maintain continuity
            self.pendulum_velocity_filter.reset(pendulum_velocity_last)
            self.motor_velocity_filter.reset(motor_velocity_last)
            self.voltage_filter.reset(voltage_last)

            # Update status label
            self.filter_status_label.config(text=f"Cutoff: {self.filter_cutoff} Hz")

        except ValueError:
            # Handle invalid input
            self.filter_status_label.config(text="Invalid cutoff value!")

    def toggle_logging(self):
        """Start or stop data logging"""
        if not self.logging:
            # Start logging
            try:
                # Get log interval from entry field
                self.log_interval = int(self.log_interval_entry.get())
                if self.log_interval < 1:
                    self.log_interval = 1
            except ValueError:
                self.log_interval = 20  # Default value

            # Open a file dialog to select save location
            log_filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Log Data As"
            )

            if not log_filename:  # User canceled
                return

            # Open the file for writing
            try:
                self.log_file = open(log_filename, 'w', newline='')
                self.log_writer = csv.writer(self.log_file)

                # Write header
                self.log_writer.writerow([
                    "Step", "Mode", "PendulumAngle", "PendulumVelocity",
                    "MotorPosition", "MotorVelocity", "dt", "Voltage"
                ])

                # Set logging state
                self.logging = True
                self.log_counter = 0
                self.log_step = 0
                self.log_start_time = time.time()
                self.prev_logging_time = self.log_start_time
                self.prev_pendulum_angle = self.qube.getPendulumAngle()
                self.prev_motor_pos = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero

                # Save the current cutoff frequency
                self.log_cutoff_freq = self.filter_cutoff

                # Update UI
                self.log_btn.config(text="Stop Logging")
                self.log_status_label.config(text="Logging: ON")
                self.log_info_label.config(text=f"Logging to: {os.path.basename(log_filename)}")

                # Set cyan LED during logging
                self.r_slider.set(0)
                self.g_slider.set(500)
                self.b_slider.set(500)

            except Exception as e:
                messagebox.showerror("Logging Error", f"Could not start logging: {str(e)}")
        else:
            # Stop logging
            self.stop_logging()

    def stop_logging(self):
        """Stop data logging and close file"""
        if self.logging:
            try:
                if self.log_file:
                    self.log_file.close()
                    self.log_file = None
                    self.log_writer = None
            except Exception as e:
                print(f"Error closing log file: {str(e)}")

            # Update state
            self.logging = False

            # Update UI
            self.log_btn.config(text="Start Logging")
            self.log_status_label.config(text="Logging: OFF")
            self.log_info_label.config(text="Not logging")

            # Reset LED to blue
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def log_data(self):
        """Log current data to CSV file with low-pass filtered velocities"""
        if not self.logging or not self.log_file or not self.log_writer:
            return

        try:
            # Increment log step counter
            self.log_step += 1

            # Get current time and calculate dt (just for logging purposes)
            current_time = time.time()
            dt = current_time - self.prev_logging_time
            self.prev_logging_time = current_time

            # Get pendulum angle and motor position
            pendulum_angle = self.qube.getPendulumAngle()
            motor_position = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero

            # For motor velocity, use the RPM directly from QUBE
            motor_rpm = self.qube.getMotorRPM()
            motor_velocity = motor_rpm * (2 * np.pi / 60)  # Convert RPM to rad/s

            # For pendulum velocity, use the same filtered value used in control
            # This requires recomputing it
            current_pendulum_angle_rad = np.radians(pendulum_angle)
            if hasattr(self, 'prev_pendulum_angle_rl') and hasattr(self, 'prev_time_rl'):
                time_diff = current_time - self.prev_time_rl
                if time_diff > 0:
                    # Get the same filtered velocity used in RL control
                    pendulum_velocity = self.pendulum_velocity_filter.y_last
                else:
                    pendulum_velocity = 0.0
            else:
                pendulum_velocity = 0.0

            # Get the filtered voltage being used
            filtered_voltage = self.motor_voltage  # Use the current commanded voltage

            # Determine mode
            mode = "Balance" if self.rl_mode else "Swingup"
            if self.moving_to_position:
                mode = "Position"
            elif self.calibrating:
                mode = "Calibrate"

            # Write the row to CSV
            self.log_writer.writerow([
                self.log_step,
                mode,
                f"{pendulum_angle:.2f}",
                f"{pendulum_velocity:.2f}",
                f"{motor_position:.2f}",
                f"{motor_velocity:.2f}",
                f"{dt:.6f}",
                f"{filtered_voltage:.2f}"
            ])

            # Flush the file to make sure data is written
            self.log_file.flush()

            # Update log status every 100 records
            if self.log_step % 100 == 0:
                elapsed = time.time() - self.log_start_time
                rate = self.log_step / max(elapsed, 0.001)
                self.log_info_label.config(text=f"Logging: {self.log_step} rows at {rate:.1f} rows/sec")

        except Exception as e:
            print(f"Error logging data: {str(e)}")
            # Try to stop logging if there's an error
            self.stop_logging()

    def set_ui_update_interval(self, value):
        """Set UI update interval from slider"""
        self.ui_update_interval = int(value)

    def set_max_voltage(self, value):
        """Set the maximum voltage for RL control from slider"""
        self.max_voltage = float(value)
        self.max_voltage_label.config(text=f"{self.max_voltage:.1f} V")

    def set_manual_voltage(self, value):
        """Set manual voltage from slider"""
        self.motor_voltage = float(value)
        # Reset any automatic control modes
        self.calibrating = False
        self.moving_to_position = False
        self.rl_mode = False

        if self.rl_model:
            self.rl_control_btn.config(text="Start RL Control")

    def calibrate(self):
        """Simple calibration - move to corner and set as zero"""
        self.calibrating = True
        self.moving_to_position = False
        self.rl_mode = False
        self.voltage_slider.set(0)  # Reset slider

        # Set calibration start time
        self.calibration_start_time = time.time()
        self.status_label.config(text="Calibrating - Finding corner...")

        # Set yellow LED during calibration
        self.r_slider.set(999)
        self.g_slider.set(999)
        self.b_slider.set(0)

    def update_calibration(self):
        """Update calibration process"""
        elapsed = time.time() - self.calibration_start_time

        if elapsed < 3.0:
            # Apply voltage to move to corner
            self.motor_voltage = 1.5
            if self.ui_counter == 0:  # Only update UI when counter is 0
                self.status_label.config(text=f"Finding corner... ({3.0 - elapsed:.1f}s)")
        else:
            # At corner - set as zero
            self.motor_voltage = 0.0

            # Reset encoders at the corner position
            self.qube.resetMotorEncoder()
            self.qube.resetPendulumEncoder()

            # Set blue LED to indicate ready
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

            # End calibration
            self.calibrating = False
            self.status_label.config(text="Calibration complete - Corner is now zero")

    def toggle_rl_control(self):
        """Toggle RL control mode on/off"""
        if not self.rl_mode:
            # Start RL control
            self.rl_mode = True
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.rl_control_btn.config(text="Stop RL Control")
            self.status_label.config(text="RL control active")

            # Set purple LED during RL control
            self.r_slider.set(500)
            self.g_slider.set(0)
            self.b_slider.set(999)

        else:
            # Stop RL control
            self.rl_mode = False
            self.motor_voltage = 0.0
            self.rl_control_btn.config(text="Start RL Control")
            self.status_label.config(text="RL control stopped")

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def start_move_to_position(self):
        """Start moving to target position"""
        if not self.calibrating and not self.rl_mode:
            try:
                # Get target position from entry field
                self.target_position = float(self.position_entry.get())

                self.moving_to_position = True
                self.voltage_slider.set(0)  # Reset slider
                self.status_label.config(text=f"Moving to {self.target_position:.1f}°...")

                # Set green LED during movement
                self.r_slider.set(0)
                self.g_slider.set(999)
                self.b_slider.set(0)
            except ValueError:
                self.status_label.config(text="Invalid position value")

    def update_position_control(self):
        """Update position control"""
        # Get current angle
        current_angle = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero

        # Calculate position error (difference from target)
        position_error = self.target_position - current_angle

        # Check if we're close enough
        if abs(position_error) < 0.5:
            # We've reached the target
            self.moving_to_position = False
            self.motor_voltage = 0.0

            # Set blue LED when done
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

            if self.ui_counter == 0:  # Only update UI when counter is 0
                self.status_label.config(text="Position reached")
            return

        # Simple proportional control
        kp = 0.02  # Low gain
        self.motor_voltage = kp * position_error

        # Limit voltage for safety
        self.motor_voltage = max(-self.system_max_voltage, min(self.system_max_voltage, self.motor_voltage))

    def get_observation(self):
        """Get the current observation vector"""
        # Get current state
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert all angles to radians for the RL model
        motor_angle = np.radians(motor_angle_deg)

        # For pendulum angle, convert to radians and flip convention:
        # QUBE: 0 degrees = down position
        # Training: 0 radians = upright position, π radians = down position
        pendulum_angle = np.radians(pendulum_angle_deg)
        # Adjust so that upright is 0
        pendulum_angle_norm = normalize_angle(pendulum_angle + np.pi)

        # Estimate motor velocity - convert from RPM to rad/s
        motor_velocity = self.qube.getMotorRPM() * (2 * np.pi / 60)  # Convert RPM to rad/s

        # Estimate pendulum velocity by finite difference (in radians)
        current_pendulum_angle_rad = np.radians(pendulum_angle_deg)
        current_time = time.time()

        if not hasattr(self, 'prev_pendulum_angle_rl'):
            self.prev_pendulum_angle_rl = current_pendulum_angle_rad
            self.prev_time_rl = current_time
            pendulum_velocity = 0.0
        else:
            dt = current_time - self.prev_time_rl
            if dt > 0:
                raw_velocity = (current_pendulum_angle_rad - self.prev_pendulum_angle_rl) / dt
                # Apply low-pass filter to pendulum velocity for RL
                pendulum_velocity = self.pendulum_velocity_filter.filter(raw_velocity, dt)
            else:
                pendulum_velocity = 0.0
            self.prev_pendulum_angle_rl = current_pendulum_angle_rad
            self.prev_time_rl = current_time

        # Create observation vector
        obs = np.array([
            np.sin(motor_angle), np.cos(motor_angle),
            np.sin(pendulum_angle_norm), np.cos(pendulum_angle_norm),
            motor_velocity / 10.0,  # Scale velocities as in training
            pendulum_velocity / 10.0
        ])

        # Return both the observation and pendulum angle for status updates
        return obs, pendulum_angle_norm

    def update_rl_control(self):
        """Update RL control logic"""
        # Get current state
        obs, pendulum_angle_norm = self.get_observation()

        # Get action from RL model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            action_mean, _ = self.actor(state_tensor)
            action = action_mean.cpu().numpy()[0][0]  # Get action as scalar

        # Apply max_voltage directly
        raw_voltage = float(action) * self.max_voltage

        # Apply low-pass filter to voltage
        current_time = time.time()
        dt = current_time - self.last_loop_time if hasattr(self, 'last_loop_time') else 0.001
        self.motor_voltage = self.voltage_filter.filter(raw_voltage, dt)

        # Update status - but only during UI updates
        if self.ui_counter == 0:
            upright_angle = abs(pendulum_angle_norm) * 180 / np.pi
            if upright_angle < 30:
                self.status_label.config(text=f"Control active - Near balance ({upright_angle:.1f}° from upright)")
            else:
                self.status_label.config(text=f"Control active - Swinging ({upright_angle:.1f}° from upright)")

    def stop_motor(self):
        """Stop the motor"""
        self.calibrating = False
        self.moving_to_position = False
        self.rl_mode = False
        self.motor_voltage = 0.0
        self.voltage_slider.set(0)

        if self.rl_model:
            self.rl_control_btn.config(text="Start RL Control")

        # Set blue LED when stopped
        self.r_slider.set(0)
        self.g_slider.set(0)
        self.b_slider.set(999)

        self.status_label.config(text="Motor stopped")

    def update_gui(self):
        """Update the GUI and control the hardware - called continuously"""
        # Increment iteration counter
        self.iteration_count += 1

        # Calculate current control frequency (less frequently)
        now = time.time()
        elapsed = now - self.last_loop_time
        self.last_loop_time = now

        # Smooth frequency calculation (moving average)
        alpha = 0.05  # Reduced smoothing factor for less processing
        self.actual_frequency = (1 - alpha) * self.actual_frequency + alpha * (1.0 / max(elapsed, 0.001))

        # Update UI counter
        self.ui_counter = (self.ui_counter + 1) % self.ui_update_interval

        # CRITICAL CONTROL OPERATIONS - These must happen on every loop
        # ---------------------------------------------------------------

        # Update automatic control modes if active (core functionality)
        if self.calibrating:
            self.update_calibration()
        elif self.moving_to_position:
            self.update_position_control()
        elif self.rl_mode:
            self.update_rl_control()

        # Apply the current motor voltage - CRITICAL TO DO ON EVERY LOOP!
        self.qube.setMotorVoltage(self.motor_voltage)

        # Handle data logging counter increment (minimal overhead)
        if self.logging:
            self.log_counter = (self.log_counter + 1) % self.log_interval
            if self.log_counter == 0:
                self.log_data()

        # Only update RGB values periodically to reduce hardware communication overhead
        if self.ui_counter % 3 == 0:  # Update RGB less frequently than UI
            self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())

        # NON-CRITICAL UI UPDATES - Only do these periodically
        # ---------------------------------------------------------------
        if self.ui_counter == 0:
            # Get current hardware values
            motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero
            pendulum_angle_deg = self.qube.getPendulumAngle()
            pendulum_angle_rad = np.radians(pendulum_angle_deg)
            pendulum_angle_norm = normalize_angle(pendulum_angle_rad + np.pi)
            rpm = self.qube.getMotorRPM()

            # Check if values have changed enough to warrant a UI update
            # This avoids unnecessary tkinter operations which are expensive
            if (abs(motor_angle_deg - self.ui_cache['motor_angle']) > 0.5 or
                    abs(pendulum_angle_deg - self.ui_cache['pendulum_angle']) > 0.5):
                # Update angle displays
                self.angle_label.config(text=f"{motor_angle_deg:.1f}°")
                self.pendulum_label.config(
                    text=f"{pendulum_angle_deg:.1f}° ({abs(pendulum_angle_norm) * 180 / np.pi:.1f}° from upright)")
                # Update cache
                self.ui_cache['motor_angle'] = motor_angle_deg
                self.ui_cache['pendulum_angle'] = pendulum_angle_deg

            # Update other displays only if they've changed significantly
            if abs(rpm - self.ui_cache['rpm']) > 1.0:
                self.rpm_label.config(text=f"{rpm:.1f}")
                self.ui_cache['rpm'] = rpm

            if abs(self.motor_voltage - self.ui_cache['voltage']) > 0.1:
                self.voltage_label.config(text=f"{self.motor_voltage:.1f} V")
                self.ui_cache['voltage'] = self.motor_voltage

            # Update frequency display less often (every 3rd UI update)
            if self.iteration_count % (self.ui_update_interval * 3) == 0:
                self.freq_label.config(text=f"{self.actual_frequency:.1f} Hz")
                self.ui_cache['frequency'] = self.actual_frequency

            # Log performance stats every 1000 iterations instead of 500
            if self.iteration_count % 1000 == 0:
                avg_freq = self.iteration_count / (now - self.start_time)
                print(
                    f"Performance: {self.iteration_count} iterations, Avg frequency: {avg_freq:.1f} Hz, Current: {self.actual_frequency:.1f} Hz")


def main():
    print("Starting QUBE Controller with RL, Data Logging, and Low-Pass Filtering...")
    print("Will set corner position as zero")
    print("OPTIMIZED VERSION - Maximum speed control, improved filter control, optimized UI")

    try:
        # Initialize QUBE
        qube = QUBE(COM_PORT, 115200)

        # Initial reset
        qube.resetMotorEncoder()
        qube.resetPendulumEncoder()
        qube.setMotorVoltage(0.0)
        qube.setRGB(0, 0, 999)  # Blue LED to start
        qube.update()

        # Configure Tkinter for higher performance
        root = tk.Tk()

        # Optimize Tkinter settings
        root.update_idletasks()  # Process any pending UI events before starting main loop
        root.protocol("WM_DELETE_WINDOW", root.destroy)  # Ensure clean shutdown

        # Create application
        app = QUBEControllerWithRL(root, qube)

        # Set initial values
        app.filter_cutoff_slider.set(500)  # Default filter cutoff to 500 Hz
        app.max_voltage_slider.set(4.0)  # Set default max voltage
        app.set_max_voltage(4.0)  # Update the internal value

        # Preallocate variables used in main loop
        ui_counter = 0
        update_count = 0

        # Display initial performance advice
        print("Performance tip: Increase 'UI Update Every N Iterations' slider for higher control frequency")

        # Main loop - MODIFIED: High performance optimized loop
        while True:
            try:
                # Update hardware - critical control operation
                qube.update()

                # Update controller - critical control operation
                app.update_gui()

                # Update Tkinter less frequently - major performance improvement
                if app.ui_counter == 0:
                    # Process Tkinter events without full redraw
                    root.update_idletasks()  # Process pending events

                    # Full update only every few UI cycles for better performance
                    update_count += 1
                    if update_count % 3 == 0:  # Reduce full UI updates further
                        root.update()  # Complete update including redraw

            except RuntimeError as e:
                # Handle transient Tkinter errors (can happen during heavy load)
                if "main thread is not in main loop" in str(e):
                    print("Handled UI threading issue, continuing...")
                    pass
                else:
                    raise  # Re-raise if it's a different error

    except tk.TclError:
        # Window closed
        print("Window closed, shutting down")
    except KeyboardInterrupt:
        # User pressed Ctrl+C
        print("Keyboard interrupt, shutting down")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Final stop attempt with more robust shutdown
        try:
            print("Shutting down motor and hardware...")
            qube.setMotorVoltage(0.0)
            qube.setRGB(0, 0, 0)  # Turn off LEDs
            qube.update()
            time.sleep(0.1)  # Brief pause to ensure commands are sent
            print("Motor stopped")
        except Exception as e:
            print(f"Warning: Could not cleanly shut down hardware: {str(e)}")


if __name__ == "__main__":
    main()
