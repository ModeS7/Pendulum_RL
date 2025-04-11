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
    def __init__(self, cutoff_freq=500.0):
        """Initialize a low-pass filter with specified cutoff frequency in Hz"""
        self.twopi = 2.0 * np.pi
        self.wc = cutoff_freq / self.twopi  # Cutoff frequency parameter
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


# Actor Networks - supporting both original and modern versions
class ActorReLU(nn.Module):
    """Original policy network that outputs action distribution with ReLU activation."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorReLU, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log std for continuous action
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Identify model type
        self.model_type = "relu"

    def forward(self, state):
        features = self.network(state)

        # Get mean and constrain it to [-1, 1]
        action_mean = torch.tanh(self.mean(features))

        # Get log standard deviation and clamp it
        action_log_std = self.log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)

        return action_mean, action_log_std


class ActorSiLU(nn.Module):
    """Modern policy network that outputs action distribution with SiLU activation."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorSiLU, self).__init__()

        # Improved network with SiLU activation
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Mean and log std for continuous action
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Identify model type
        self.model_type = "silu"

    def forward(self, state):
        """Forward pass to get action mean and log std."""
        features = self.network(state)

        # Get mean and constrain it to [-1, 1]
        action_mean = torch.tanh(self.mean(features))

        # Get log standard deviation and clamp it
        action_log_std = self.log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)

        return action_mean, action_log_std

    def sample(self, state):
        """Sample action from the distribution and compute log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Sample from normal distribution with reparameterization trick
        normal = Normal(mean, std)
        x = normal.rsample()

        # Constrain to [-1, 1]
        action = torch.tanh(x)

        # Calculate log probability with squashing correction
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


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

        # Performance optimization settings
        self.target_frequency = 1000  # Increased target control frequency to 1000 Hz (max)
        self.ui_update_interval = 5  # Update UI every N iterations (was every iteration)
        self.ui_counter = 0
        self.last_loop_time = time.time()
        self.actual_frequency = 0

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

        # Low-pass filters (same cutoff freq as Arduino code)
        self.dt = 1.0 / self.target_frequency  # Default timestep
        self.pendulum_velocity_filter = LowPassFilter(cutoff_freq=500.0)
        self.motor_velocity_filter = LowPassFilter(cutoff_freq=500.0)
        self.voltage_filter = LowPassFilter(cutoff_freq=500.0)
        self.filtered_voltage = 0.0

        # Initialize the RL model (but don't load weights yet)
        self.initialize_rl_model()

        # Create GUI elements
        self.create_gui()

    def initialize_rl_model(self):
        """Initialize the RL model architecture"""
        state_dim = 6  # Our observation space (same as in training)
        action_dim = 1  # Motor voltage (normalized)

        # Initialize both model types
        self.actor_relu = ActorReLU(state_dim, action_dim)
        self.actor_silu = ActorSiLU(state_dim, action_dim)

        # Default to the original ReLU model initially
        self.actor = self.actor_relu
        self.model_type = "relu"

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

                # Try to determine model type automatically
                # First attempt: Try loading into both models and check which one works
                try_relu = True
                try_silu = True

                # Try loading into ReLU model
                try:
                    self.actor_relu.load_state_dict(state_dict)
                    self.actor_relu.to(self.device)
                    self.actor_relu.eval()
                except Exception:
                    try_relu = False

                # Try loading into SiLU model
                try:
                    self.actor_silu.load_state_dict(state_dict)
                    self.actor_silu.to(self.device)
                    self.actor_silu.eval()
                except Exception:
                    try_silu = False

                # Decide which model to use based on auto-detection
                if try_relu and try_silu:
                    # If both succeed, show dialog to select
                    model_choice = messagebox.askquestion(
                        "Model Type",
                        "Model can load with either architecture. Use SiLU model (newer, recommended)? Select 'No' for ReLU model."
                    )
                    if model_choice == 'yes':
                        self.actor = self.actor_silu
                        self.model_type = "silu"
                    else:
                        self.actor = self.actor_relu
                        self.model_type = "relu"
                elif try_relu:
                    self.actor = self.actor_relu
                    self.model_type = "relu"
                elif try_silu:
                    self.actor = self.actor_silu
                    self.model_type = "silu"
                else:
                    raise ValueError("Model is not compatible with either architecture")

                # Confirm model loaded
                model_type_str = "SiLU" if self.model_type == "silu" else "ReLU"
                self.status_label.config(text=f"Model loaded: {os.path.basename(filename)}")
                self.rl_model = filename

                # Update model type displays
                self.model_type_label.config(text=f"Model Type: {model_type_str}")
                self.architecture_label.config(text=f"Using {model_type_str} activation network")

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

        # Model Type Indicator (added to show current model type)
        self.model_type_label = Label(rl_frame, text="Model Type: Not Loaded", width=20)
        self.model_type_label.grid(row=0, column=2, padx=5)

        # CHANGED: Replace RL scaling factor slider with max voltage slider
        max_voltage_frame = Frame(control_frame)
        max_voltage_frame.grid(row=2, column=0, pady=5)

        self.max_voltage_slider = Scale(
            max_voltage_frame,
            from_=0.5,
            to=self.system_max_voltage,  # Now goes up to 18V
            orient=tk.HORIZONTAL,
            label="RL Max Voltage",
            length=300,
            resolution=0.1,
            command=self.set_max_voltage
        )
        self.max_voltage_slider.set(self.max_voltage)  # Start with default value
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

        # DATA LOGGING SECTION - NEW
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

        # Filter Controls - NEW
        filter_frame = Frame(control_frame)
        filter_frame.grid(row=5, column=0, pady=5)

        Label(filter_frame, text="Filter Cutoff (Hz):").grid(row=0, column=0, padx=5)
        self.filter_entry = Entry(filter_frame, width=5)
        self.filter_entry.grid(row=0, column=1, padx=5)
        self.filter_entry.insert(0, "500")  # Default to 500 Hz

        self.filter_btn = Button(filter_frame, text="Update Filter",
                                 command=self.update_filter_cutoff, width=15)
        self.filter_btn.grid(row=0, column=2, padx=5)

        # Filter status
        self.filter_status_label = Label(filter_frame, text="Cutoff: 500 Hz")
        self.filter_status_label.grid(row=0, column=3, padx=5)

        # Stop button - moved to row 6
        self.stop_btn = Button(control_frame, text="STOP MOTOR",
                               command=self.stop_motor,
                               width=20, height=2,
                               bg="red", fg="white")
        self.stop_btn.grid(row=6, column=0, pady=10)

        # Manual voltage control - moved to row 7
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

        # Performance settings frame - moved to row 8
        perf_frame = Frame(control_frame)
        perf_frame.grid(row=8, column=0, pady=5)

        # Add frequency slider
        self.freq_slider = Scale(
            perf_frame,
            from_=60,
            to=1000,  # Increased maximum frequency to 1000 Hz
            orient=tk.HORIZONTAL,
            label="Target Control Frequency (Hz)",
            length=300,
            resolution=10,
            command=self.set_target_frequency
        )
        self.freq_slider.set(self.target_frequency)
        self.freq_slider.grid(row=0, column=0, padx=5)

        # Add UI update interval slider
        self.ui_slider = Scale(
            perf_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            label="UI Update Every N Iterations",
            length=300,
            resolution=1,
            command=self.set_ui_update_interval
        )
        self.ui_slider.set(self.ui_update_interval)
        self.ui_slider.grid(row=1, column=0, padx=5)

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

        # CHANGED: Update label from scaling factor to max voltage
        Label(status_frame, text="RL Max Voltage:").grid(row=7, column=0, sticky=tk.W)
        self.max_voltage_label = Label(status_frame, text=f"{self.max_voltage:.1f} V")
        self.max_voltage_label.grid(row=7, column=1, sticky=tk.W)

        # Add actual frequency display
        Label(status_frame, text="Control Frequency:").grid(row=8, column=0, sticky=tk.W)
        self.freq_label = Label(status_frame, text=f"{self.actual_frequency:.1f} Hz")
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

    def update_filter_cutoff(self):
        """Update the cutoff frequency of the low-pass filters"""
        try:
            # Get cutoff frequency from entry field
            cutoff_freq = float(self.filter_entry.get())
            if cutoff_freq <= 0:
                cutoff_freq = 500.0  # Default if invalid

            # Store current filter values to preserve states
            pendulum_velocity_last = self.pendulum_velocity_filter.y_last
            motor_velocity_last = self.motor_velocity_filter.y_last
            voltage_last = self.voltage_filter.y_last

            # Create new filters with updated cutoff frequency
            self.pendulum_velocity_filter = LowPassFilter(cutoff_freq=cutoff_freq)
            self.motor_velocity_filter = LowPassFilter(cutoff_freq=cutoff_freq)
            self.voltage_filter = LowPassFilter(cutoff_freq=cutoff_freq)

            # Transfer the previous states to maintain continuity
            self.pendulum_velocity_filter.reset(pendulum_velocity_last)
            self.motor_velocity_filter.reset(motor_velocity_last)
            self.voltage_filter.reset(voltage_last)

            # Update status
            self.filter_status_label.config(text=f"Cutoff: {cutoff_freq} Hz")

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

                # Don't reset filters here anymore - instead keep using the existing ones
                # Only save the current cutoff frequency in case we need it
                try:
                    self.log_cutoff_freq = float(self.filter_entry.get())
                except ValueError:
                    self.log_cutoff_freq = 500.0  # Default value

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

            # Get current velocities from the main control loop's filters
            # instead of calculating them separately
            # This is to ensure consistency between control and logging

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

        except Exception as e:
            print(f"Error logging data: {str(e)}")
            # Try to stop logging if there's an error
            self.stop_logging()

    def set_target_frequency(self, value):
        """Set target control frequency from slider"""
        self.target_frequency = float(value)

    def set_ui_update_interval(self, value):
        """Set UI update interval from slider"""
        self.ui_update_interval = int(value)

    # CHANGED: Replaced RL scaling factor setter with max voltage setter
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
        self.rl_started = False

        if self.rl_model:
            self.rl_control_btn.config(text="Start RL Control")

    def calibrate(self):
        """Simple calibration - move to corner and set as zero"""
        self.calibrating = True
        self.moving_to_position = False
        self.rl_mode = False
        self.rl_started = False
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
        """Get the current observation vector (shared between ReLU and SiLU models)"""
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
        """Update RL control logic using ReLU model"""
        # Get current state
        obs, pendulum_angle_norm = self.get_observation()

        # Get action from RL model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            action_mean, _ = self.actor(state_tensor)
            action = action_mean.cpu().numpy()[0][0]  # Get action as scalar

        # Apply max_voltage directly
        raw_voltage = float(action) * self.max_voltage

        # Apply low-pass filter to voltage (just like in Arduino code)
        current_time = time.time()
        dt = current_time - self.last_loop_time if hasattr(self, 'last_loop_time') else 0.001
        self.motor_voltage = self.voltage_filter.filter(raw_voltage, dt)

        # Update status - but only during UI updates
        if self.ui_counter == 0:
            upright_angle = abs(pendulum_angle_norm) * 180 / np.pi
            if upright_angle < 30:
                self.status_label.config(text=f"RL control active - Near balance ({upright_angle:.1f}° from upright)")
            else:
                self.status_label.config(text=f"RL control active - Swinging ({upright_angle:.1f}° from upright)")

    def update_rl_control_silu(self):
        """Update RL control logic using SiLU model with sampling if available"""
        # Get current state
        obs, pendulum_angle_norm = self.get_observation()

        # Get action from RL model - using sample method if available
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)

            # Try using the sample method if available (for exploration)
            try:
                action, _ = self.actor.sample(state_tensor)
                action = action.cpu().numpy()[0][0]  # Get action as scalar
            except (AttributeError, NotImplementedError):
                # Fall back to standard forward if sample is not implemented
                action_mean, _ = self.actor(state_tensor)
                action = action_mean.cpu().numpy()[0][0]  # Get action as scalar

        # Apply max_voltage directly
        raw_voltage = float(action) * self.max_voltage

        # Apply low-pass filter to voltage (just like in Arduino code)
        current_time = time.time()
        dt = current_time - self.last_loop_time if hasattr(self, 'last_loop_time') else 0.001
        self.motor_voltage = self.voltage_filter.filter(raw_voltage, dt)

        # Update status - but only during UI updates
        if self.ui_counter == 0:
            upright_angle = abs(pendulum_angle_norm) * 180 / np.pi
            if upright_angle < 30:
                self.status_label.config(
                    text=f"RL control active (SiLU) - Near balance ({upright_angle:.1f}° from upright)")
            else:
                self.status_label.config(
                    text=f"RL control active (SiLU) - Swinging ({upright_angle:.1f}° from upright)")

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

        # Calculate current control frequency (every 50 iterations)
        now = time.time()
        elapsed = now - self.last_loop_time
        self.last_loop_time = now

        # Smooth frequency calculation (moving average)
        alpha = 0.1  # Smoothing factor
        self.actual_frequency = (1 - alpha) * self.actual_frequency + alpha * (1.0 / max(elapsed, 0.001))

        # Update UI counter
        self.ui_counter = (self.ui_counter + 1) % self.ui_update_interval

        # Update automatic control modes if active
        if self.calibrating:
            self.update_calibration()
        elif self.moving_to_position:
            self.update_position_control()
        elif self.rl_mode:
            if self.model_type == "silu":
                self.update_rl_control_silu()  # Use SiLU-specific control if applicable
            else:
                self.update_rl_control()

        # Apply the current motor voltage - THIS IS CRITICAL TO DO ON EVERY LOOP!
        self.qube.setMotorVoltage(self.motor_voltage)

        # Apply RGB values
        self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())

        # Handle data logging - log data every log_interval iterations
        if self.logging:
            self.log_counter = (self.log_counter + 1) % self.log_interval
            if self.log_counter == 0:
                self.log_data()

        # Only update display information when UI counter is 0
        if self.ui_counter == 0:
            # Update display information
            motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero
            pendulum_angle_deg = self.qube.getPendulumAngle()

            # Convert to radians for normalized angle calculation
            pendulum_angle_rad = np.radians(pendulum_angle_deg)
            pendulum_angle_norm = normalize_angle(pendulum_angle_rad + np.pi)  # For display

            rpm = self.qube.getMotorRPM()

            self.angle_label.config(text=f"{motor_angle_deg:.1f}°")
            self.pendulum_label.config(
                text=f"{pendulum_angle_deg:.1f}° ({abs(pendulum_angle_norm) * 180 / np.pi:.1f}° from upright)")
            self.rpm_label.config(text=f"{rpm:.1f}")
            self.voltage_label.config(text=f"{self.motor_voltage:.1f} V")
            self.freq_label.config(text=f"{self.actual_frequency:.1f} Hz")

            if self.rl_model:
                self.model_label.config(text=f"Using: {os.path.basename(self.rl_model)}")

            # Log performance stats every 500 iterations
            if self.iteration_count % 500 == 0:
                avg_freq = self.iteration_count / (now - self.start_time)
                print(
                    f"Performance: {self.iteration_count} iterations, Avg frequency: {avg_freq:.1f} Hz, Current: {self.actual_frequency:.1f} Hz")


def main():
    print("Starting QUBE Controller with RL, Data Logging, and Low-Pass Filtering...")
    print("Will set corner position as zero")
    print("OPTIMIZED VERSION - High-frequency control, extended voltage range")

    try:
        # Initialize QUBE
        qube = QUBE(COM_PORT, 115200)

        # Initial reset
        qube.resetMotorEncoder()
        qube.resetPendulumEncoder()
        qube.setMotorVoltage(0.0)
        qube.setRGB(0, 0, 999)  # Blue LED to start
        qube.update()

        # Create GUI
        root = tk.Tk()
        app = QUBEControllerWithRL(root, qube)

        # Set initial high-performance values
        app.freq_slider.set(500)  # Set to max frequency
        app.max_voltage_slider.set(4.0)  # Set to 18V
        app.set_max_voltage(4.0)  # Update the internal value

        # For frequency calculation
        target_period = 1.0 / app.target_frequency
        last_time = time.time()

        # Main loop with dynamic timing
        while True:
            loop_start = time.time()

            # Update hardware
            qube.update()

            # Update controller
            app.update_gui()

            # Update Tkinter - reduced frequency based on UI counter
            if app.ui_counter == 0:
                root.update()

            # Dynamic sleep calculation to maintain target frequency
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, target_period - elapsed)

            if sleep_time > 0:
                time.sleep(sleep_time)

            # Update target period based on current slider value
            target_period = 1.0 / app.target_frequency

    except tk.TclError:
        # Window closed
        pass
    except KeyboardInterrupt:
        # User pressed Ctrl+C
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Final stop attempt
        try:
            qube.setMotorVoltage(0.0)
            qube.update()
            print("Motor stopped")
        except:
            pass


if __name__ == "__main__":
    main()