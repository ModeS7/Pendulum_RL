import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog, Checkbutton, IntVar
from QUBE import QUBE
import time
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import threading
from pendulum_kalman_filter import PendulumKalmanFilter  # Save the previous code as this file

# Update with your COM port
COM_PORT = "COM10"

# System parameters (copied from your training code)
# Assuming you've defined these in a separate file
Rm = 8.94
Km = 0.0431
Jm = 6e-5
bm = 3e-4
DA = 3e-4
DL = 5e-4
mA = 0.053
mL = 0.024
LA = 0.086
LL = 0.128
JA = 5.72e-5
JL = 1.31e-4
g = 9.81
max_voltage = 10.0
THETA_MIN = -2.2
THETA_MAX = 2.2


# Actor Network - same as before
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.network(state)
        action_mean = torch.tanh(self.mean(features))
        action_log_std = self.log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        return action_mean, action_log_std


# Helper functions - copied from your training code
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


def dynamics_step(state, t, vm):
    """Simplified dynamics calculation - pulled from your training code"""
    theta_m, theta_L, theta_m_dot, theta_L_dot = state[0], state[1], state[2], state[3]

    # Check theta limits
    if (theta_m >= THETA_MAX and theta_m_dot > 0) or (theta_m <= THETA_MIN and theta_m_dot < 0):
        theta_m_dot = 0.0

    # Apply dead zone and calculate motor torque
    if -0.2 <= vm <= 0.2:
        vm = 0.0

    # Motor torque calculation
    im = (vm - Km * theta_m_dot) / Rm
    Tm = Km * im

    # Equations of motion coefficients
    half_mL_LL_g = 0.5 * mL * LL * g
    half_mL_LL_LA = 0.5 * mL * LL * LA
    quarter_mL_LL_squared = 0.25 * mL * LL ** 2

    # For theta_m equation:
    M11 = mL * LA ** 2 + quarter_mL_LL_squared - quarter_mL_LL_squared * np.cos(theta_L) ** 2 + JA
    M12 = -half_mL_LL_LA * np.cos(theta_L)
    C1 = 0.5 * mL * LL ** 2 * np.sin(theta_L) * np.cos(theta_L) * theta_m_dot * theta_L_dot
    C2 = half_mL_LL_LA * np.sin(theta_L) * theta_L_dot ** 2

    # For theta_L equation:
    M21 = half_mL_LL_LA * np.cos(theta_L)
    M22 = JL + quarter_mL_LL_squared
    C3 = -quarter_mL_LL_squared * np.cos(theta_L) * np.sin(theta_L) * theta_m_dot ** 2
    G = half_mL_LL_g * np.sin(theta_L)

    # Calculate determinant for matrix inversion
    det_M = M11 * M22 - M12 * M21

    # Handle near-singular matrix
    if abs(det_M) < 1e-10:
        theta_m_ddot = 0
        theta_L_ddot = 0
    else:
        # Right-hand side of equations
        RHS1 = Tm - C1 - C2 - DA * theta_m_dot
        RHS2 = -G - DL * theta_L_dot - C3

        # Solve for accelerations
        theta_m_ddot = (M22 * RHS1 - M12 * RHS2) / det_M
        theta_L_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M

    return np.array([theta_m_dot, theta_L_dot, theta_m_ddot, theta_L_ddot])


def simulation_model(state, action):
    """Wrapper for dynamics_step to be used with Kalman filter"""
    dt = 0.01  # Time step

    # Use RK4 integration as in your original code
    k1 = dynamics_step(state, 0, action)
    k2 = dynamics_step(state + 0.5 * dt * k1, 0, action)
    k3 = dynamics_step(state + 0.5 * dt * k2, 0, action)
    k4 = dynamics_step(state + dt * k3, 0, action)

    new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Apply theta limits
    if new_state[0] > THETA_MAX:
        new_state[0] = THETA_MAX
        if new_state[2] > 0:
            new_state[2] = 0.0
    elif new_state[0] < THETA_MIN:
        new_state[0] = THETA_MIN
        if new_state[2] < 0:
            new_state[2] = 0.0

    return new_state


class QUBEControllerWithRL:
    def __init__(self, master, qube):
        self.master = master
        self.qube = qube
        master.title("QUBE Controller with RL and Kalman Filter")

        # Control state
        self.motor_voltage = 0.0
        self.target_position = 0.0
        self.calibrating = False
        self.rl_mode = False
        self.moving_to_position = False
        self.rl_model = None
        self.max_voltage = 10.0

        # Initialize Kalman filter
        self.kf = PendulumKalmanFilter(dt=0.01)
        self.use_kalman = True  # Default to using Kalman filter
        self.prev_pendulum_angle_rad = 0
        self.prev_time = time.time()

        # Initialize filtered state with zeros
        self.filtered_state = np.zeros(4)

        # Data collection for plotting
        self.max_data_points = 1000  # Store last 10 seconds at 100Hz
        self.time_history = deque(maxlen=self.max_data_points)
        self.raw_theta_history = deque(maxlen=self.max_data_points)
        self.raw_alpha_history = deque(maxlen=self.max_data_points)
        self.model_theta_history = deque(maxlen=self.max_data_points)
        self.model_alpha_history = deque(maxlen=self.max_data_points)
        self.filtered_theta_history = deque(maxlen=self.max_data_points)
        self.filtered_alpha_history = deque(maxlen=self.max_data_points)
        self.voltage_history = deque(maxlen=self.max_data_points)
        self.is_recording = False
        self.record_start_time = 0

        # Model state for comparison
        self.model_state = np.zeros(4)

        # Initialize the RL model
        self.initialize_rl_model()

        # Create GUI elements
        self.create_gui()

    def initialize_rl_model(self):
        """Initialize the RL model architecture"""
        state_dim = 6  # Our observation space
        action_dim = 1  # Motor voltage (normalized)
        self.actor = Actor(state_dim, action_dim)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor.eval()

    def load_rl_model(self):
        """Open file dialog to select the model file"""
        filename = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select RL Model File",
            filetypes=(("PyTorch Models", "*.pth"), ("All files", "*.*"))
        )

        if filename:
            try:
                # Load the model
                self.actor.load_state_dict(torch.load(filename, map_location=self.device))
                self.status_label.config(text=f"Model loaded: {os.path.basename(filename)}")
                self.rl_model = filename

                # Enable RL control button
                self.rl_control_btn.config(state=tk.NORMAL)

                # Set blue LED to indicate ready
                self.r_slider.set(0)
                self.g_slider.set(0)
                self.b_slider.set(999)

            except Exception as e:
                self.status_label.config(text=f"Error loading model: {str(e)}")

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

        # Kalman filter toggle
        self.kalman_var = IntVar(value=1)  # Default to ON
        self.kalman_check = Checkbutton(rl_frame, text="Use Kalman Filter",
                                        variable=self.kalman_var,
                                        command=self.toggle_kalman)
        self.kalman_check.grid(row=0, column=2, padx=5)

        # Position control (same as before)
        position_frame = Frame(control_frame)
        position_frame.grid(row=2, column=0, pady=10)

        Label(position_frame, text="Target Position (degrees):").grid(row=0, column=0, padx=5)
        self.position_entry = Entry(position_frame, width=10)
        self.position_entry.grid(row=0, column=1, padx=5)
        self.position_entry.insert(0, "0.0")

        self.move_btn = Button(position_frame, text="Move to Position",
                               command=self.start_move_to_position, width=15)
        self.move_btn.grid(row=0, column=2, padx=5)

        # Stop button
        self.stop_btn = Button(control_frame, text="STOP MOTOR",
                               command=self.stop_motor,
                               width=20, height=2,
                               bg="red", fg="white")
        self.stop_btn.grid(row=3, column=0, pady=10)

        # Manual voltage control
        self.voltage_slider = Scale(
            control_frame,
            from_=-self.max_voltage,
            to=self.max_voltage,
            orient=tk.HORIZONTAL,
            label="Manual Voltage",
            length=400,
            resolution=0.1,
            command=self.set_manual_voltage
        )
        self.voltage_slider.set(0)
        self.voltage_slider.grid(row=4, column=0, padx=5, pady=10)

        # Status display
        status_frame = Frame(self.master, padx=10, pady=10)
        status_frame.pack()

        Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = Label(status_frame, text="Ready - Please calibrate", width=40)
        self.status_label.grid(row=0, column=1, sticky=tk.W)

        Label(status_frame, text="Model:").grid(row=1, column=0, sticky=tk.W)
        self.model_label = Label(status_frame, text="No model loaded", width=40)
        self.model_label.grid(row=1, column=1, sticky=tk.W)

        Label(status_frame, text="Motor Angle:").grid(row=2, column=0, sticky=tk.W)
        self.angle_label = Label(status_frame, text="0.0°")
        self.angle_label.grid(row=2, column=1, sticky=tk.W)

        Label(status_frame, text="Pendulum Angle:").grid(row=3, column=0, sticky=tk.W)
        self.pendulum_label = Label(status_frame, text="0.0°")
        self.pendulum_label.grid(row=3, column=1, sticky=tk.W)

        # Add Kalman filter status display
        Label(status_frame, text="Filtered Angle:").grid(row=4, column=0, sticky=tk.W)
        self.filtered_angle_label = Label(status_frame, text="0.0°")
        self.filtered_angle_label.grid(row=4, column=1, sticky=tk.W)

        Label(status_frame, text="Motor RPM:").grid(row=5, column=0, sticky=tk.W)
        self.rpm_label = Label(status_frame, text="0.0")
        self.rpm_label.grid(row=5, column=1, sticky=tk.W)

        Label(status_frame, text="Current Voltage:").grid(row=6, column=0, sticky=tk.W)
        self.voltage_label = Label(status_frame, text="0.0 V")
        self.voltage_label.grid(row=6, column=1, sticky=tk.W)

        # RGB Control
        rgb_frame = Frame(self.master, padx=10, pady=10)
        rgb_frame.pack()

        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

        # Data collection and plotting controls
        plot_frame = Frame(self.master, padx=10, pady=10)
        plot_frame.pack()

        self.record_btn = Button(plot_frame, text="Start Recording Data",
                                 command=self.toggle_recording, width=20)
        self.record_btn.grid(row=0, column=0, padx=5, pady=5)

        self.plot_btn = Button(plot_frame, text="Plot Data",
                               command=self.plot_data, width=20,
                               state=tk.DISABLED)
        self.plot_btn.grid(row=0, column=1, padx=5, pady=5)

        self.save_data_btn = Button(plot_frame, text="Save Data to CSV",
                                    command=self.save_data, width=20,
                                    state=tk.DISABLED)
        self.save_data_btn.grid(row=0, column=2, padx=5, pady=5)

    def toggle_kalman(self):
        """Toggle Kalman filter on/off"""
        self.use_kalman = bool(self.kalman_var.get())
        if self.use_kalman:
            self.status_label.config(text="Kalman filter enabled")
        else:
            self.status_label.config(text="Kalman filter disabled")

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
            self.status_label.config(text=f"Finding corner... ({3.0 - elapsed:.1f}s)")
        else:
            # At corner - set as zero
            self.motor_voltage = 0.0

            # Reset encoders at the corner position
            self.qube.resetMotorEncoder()
            self.qube.resetPendulumEncoder()

            # Reset Kalman filter with new zero
            self.kf = PendulumKalmanFilter(dt=0.01)

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

            if self.use_kalman:
                self.status_label.config(text="RL control active with Kalman filter")
            else:
                self.status_label.config(text="RL control active (without Kalman filter)")

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
        # Get current angle (either filtered or raw)
        if self.use_kalman and hasattr(self, 'filtered_state'):
            current_angle = np.degrees(self.filtered_state[0]) + 136.0
        else:
            current_angle = self.qube.getMotorAngle() + 136.0

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

            self.status_label.config(text="Position reached")
            return

        # Simple proportional control
        kp = 0.02  # Low gain
        self.motor_voltage = kp * position_error

        # Limit voltage for safety
        self.motor_voltage = max(-self.max_voltage, min(self.max_voltage, self.motor_voltage))

    def get_pendulum_velocity(self):
        """Calculate pendulum angular velocity using finite differences"""
        current_pendulum_angle_rad = np.radians(self.qube.getPendulumAngle())
        current_time = time.time()

        dt = current_time - self.prev_time
        if dt > 0:
            pendulum_velocity = (current_pendulum_angle_rad - self.prev_pendulum_angle_rad) / dt
        else:
            pendulum_velocity = 0.0

        self.prev_pendulum_angle_rad = current_pendulum_angle_rad
        self.prev_time = current_time

        return pendulum_velocity

    def update_rl_control(self):
        """Update RL control logic with optional Kalman filtering"""
        # Get raw measurements
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert angles to radians
        motor_angle_rad = np.radians(motor_angle_deg)
        pendulum_angle_rad = np.radians(pendulum_angle_deg)

        # Get velocities
        motor_velocity = self.qube.getMotorRPM() * (2 * np.pi / 60)  # RPM to rad/s
        pendulum_velocity = self.get_pendulum_velocity()

        # Create raw state vector
        raw_state = np.array([
            motor_angle_rad,
            pendulum_angle_rad,
            motor_velocity,
            pendulum_velocity
        ])

        # Update model prediction (simulation)
        if hasattr(self, 'model_state'):
            # Use same dynamics as in training
            self.model_state = simulation_model(self.model_state, self.motor_voltage)

        if self.use_kalman:
            # Use Kalman filter for state estimation
            # First predict using previous action
            self.kf.predict(simulation_model, self.motor_voltage)

            # Then update with measurements
            self.filtered_state = self.kf.update(raw_state)

            # Use filtered state for control
            motor_angle = self.filtered_state[0]
            pendulum_angle = self.filtered_state[1]
            motor_velocity = self.filtered_state[2]
            pendulum_velocity = self.filtered_state[3]
        else:
            # Use raw measurements directly
            motor_angle = motor_angle_rad
            pendulum_angle = pendulum_angle_rad
            motor_velocity = motor_velocity
            pendulum_velocity = pendulum_velocity

            # Store for display
            self.filtered_state = raw_state

        # Normalize pendulum angle for RL model (as in training)
        pendulum_angle_norm = normalize_angle(pendulum_angle + np.pi)

        # Create observation vector for RL policy
        obs = np.array([
            np.sin(motor_angle), np.cos(motor_angle),
            np.sin(pendulum_angle_norm), np.cos(pendulum_angle_norm),
            motor_velocity / 10.0,  # Scale velocities as in training
            pendulum_velocity / 10.0
        ])

        # Get action from RL model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            action_mean, _ = self.actor(state_tensor)
            action = action_mean.cpu().numpy()[0][0]  # Get action as scalar

        # Convert normalized action [-1, 1] to voltage
        self.motor_voltage = float(action) * self.max_voltage

        # Update status
        upright_angle = abs(pendulum_angle_norm) * 180 / np.pi
        filter_status = "with Kalman filter" if self.use_kalman else "without Kalman filter"

        if upright_angle < 30:
            self.status_label.config(
                text=f"RL control ({filter_status}) - Near balance ({upright_angle:.1f}° from upright)")
        else:
            self.status_label.config(
                text=f"RL control ({filter_status}) - Swinging ({upright_angle:.1f}° from upright)")

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

    def toggle_recording(self):
        """Toggle data recording for comparison plots"""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
            self.record_start_time = time.time()

            # Clear previous data
            self.time_history.clear()
            self.raw_theta_history.clear()
            self.raw_alpha_history.clear()
            self.model_theta_history.clear()
            self.model_alpha_history.clear()
            self.filtered_theta_history.clear()
            self.filtered_alpha_history.clear()
            self.voltage_history.clear()

            # Reset model state to match current real state
            motor_angle_rad = np.radians(self.qube.getMotorAngle() + 136.0)
            pendulum_angle_rad = np.radians(self.qube.getPendulumAngle())
            motor_velocity = self.qube.getMotorRPM() * (2 * np.pi / 60)
            pendulum_velocity = self.get_pendulum_velocity()

            self.model_state = np.array([
                motor_angle_rad,
                pendulum_angle_rad,
                motor_velocity,
                pendulum_velocity
            ])

            self.status_label.config(text="Recording data...")

            # Set recording LED to yellow
            self.r_slider.set(999)
            self.g_slider.set(999)
            self.b_slider.set(0)

        else:
            # Stop recording
            self.is_recording = False
            self.record_btn.config(text="Start Recording Data")

            # Enable plot button if we have data
            if len(self.time_history) > 0:
                self.plot_btn.config(state=tk.NORMAL)
                self.save_data_btn.config(state=tk.NORMAL)

            self.status_label.config(text=f"Recording stopped. {len(self.time_history)} data points collected.")

            # Set LED back to default (blue)
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def save_data(self):
        """Save recorded data to CSV file"""
        if len(self.time_history) == 0:
            return

        try:
            filename = filedialog.asksaveasfilename(
                initialdir=os.getcwd(),
                title="Save Data As CSV",
                defaultextension=".csv",
                filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
            )

            if filename:
                with open(filename, 'w') as f:
                    # Write header
                    f.write("time,raw_theta,raw_alpha,model_theta,model_alpha,filtered_theta,filtered_alpha,voltage\n")

                    # Write data
                    for i in range(len(self.time_history)):
                        f.write(
                            f"{self.time_history[i]:.3f},{self.raw_theta_history[i]:.6f},{self.raw_alpha_history[i]:.6f},"
                            f"{self.model_theta_history[i]:.6f},{self.model_alpha_history[i]:.6f},"
                            f"{self.filtered_theta_history[i]:.6f},{self.filtered_alpha_history[i]:.6f},"
                            f"{self.voltage_history[i]:.6f}\n")

                self.status_label.config(text=f"Data saved to {os.path.basename(filename)}")
        except Exception as e:
            self.status_label.config(text=f"Error saving data: {str(e)}")

    def plot_data(self):
        """Plot the recorded data"""
        if len(self.time_history) == 0:
            return

        # Create a new popup window for plots
        plot_window = tk.Toplevel(self.master)
        plot_window.title("System vs Model Comparison")
        plot_window.geometry("1000x800")

        # Convert deques to lists for plotting
        times = list(self.time_history)
        raw_theta = list(self.raw_theta_history)
        raw_alpha = list(self.raw_alpha_history)
        model_theta = list(self.model_theta_history)
        model_alpha = list(self.model_alpha_history)
        filtered_theta = list(self.filtered_theta_history)
        filtered_alpha = list(self.filtered_alpha_history)
        voltages = list(self.voltage_history)

        # Create figure with subplots
        fig = plt.Figure(figsize=(10, 8))

        # Motor angle plot
        ax1 = fig.add_subplot(311)
        ax1.plot(times, raw_theta, 'b-', label='Real System')
        ax1.plot(times, model_theta, 'r--', label='Model Prediction')
        ax1.plot(times, filtered_theta, 'g-', label='Kalman Filter')
        ax1.set_ylabel('Motor Angle (rad)')
        ax1.set_title('Motor Angle Comparison')
        ax1.legend()
        ax1.grid(True)

        # Pendulum angle plot
        ax2 = fig.add_subplot(312)
        ax2.plot(times, raw_alpha, 'b-', label='Real System')
        ax2.plot(times, model_alpha, 'r--', label='Model Prediction')
        ax2.plot(times, filtered_alpha, 'g-', label='Kalman Filter')
        ax2.set_ylabel('Pendulum Angle (rad)')
        ax2.set_title('Pendulum Angle Comparison')
        ax2.legend()
        ax2.grid(True)

        # Voltage plot
        ax3 = fig.add_subplot(313)
        ax3.plot(times, voltages, 'k-')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Voltage (V)')
        ax3.set_title('Control Voltage')
        ax3.grid(True)

        # Add canvas to window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add close button
        close_btn = Button(plot_window, text="Close", command=plot_window.destroy)
        close_btn.pack(pady=10)

    def update_gui(self):
        """Update the GUI and control the hardware"""
        # Update automatic control modes if active
        if self.calibrating:
            self.update_calibration()
        elif self.moving_to_position:
            self.update_position_control()
        elif self.rl_mode:
            self.update_rl_control()

        # Apply the current motor voltage
        self.qube.setMotorVoltage(self.motor_voltage)

        # Apply RGB values
        self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())

        # Update display information
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert to radians for normalized angle calculation
        motor_angle_rad = np.radians(motor_angle_deg)
        pendulum_angle_rad = np.radians(pendulum_angle_deg)
        pendulum_angle_norm = normalize_angle(pendulum_angle_rad + np.pi)  # For display

        # Get velocities
        motor_velocity = self.qube.getMotorRPM() * (2 * np.pi / 60)  # RPM to rad/s
        pendulum_velocity = self.get_pendulum_velocity()

        # Show filtered angles if Kalman is active
        if self.use_kalman:
            filtered_motor_deg = np.degrees(self.filtered_state[0]) + 136.0
            filtered_pendulum_deg = np.degrees(self.filtered_state[1])
            filtered_pendulum_norm = normalize_angle(self.filtered_state[1] + np.pi)

            self.filtered_angle_label.config(
                text=f"Motor: {filtered_motor_deg:.1f}°, Pend: {filtered_pendulum_deg:.1f}° ({abs(filtered_pendulum_norm) * 180 / np.pi:.1f}° from up)")
        else:
            self.filtered_angle_label.config(text="Kalman filter disabled")

        rpm = self.qube.getMotorRPM()

        self.angle_label.config(text=f"{motor_angle_deg:.1f}°")
        self.pendulum_label.config(
            text=f"{pendulum_angle_deg:.1f}° ({abs(pendulum_angle_norm) * 180 / np.pi:.1f}° from upright)")
        self.rpm_label.config(text=f"{rpm:.1f}")
        self.voltage_label.config(text=f"{self.motor_voltage:.1f} V")

        if self.rl_model:
            self.model_label.config(text=f"Using: {os.path.basename(self.rl_model)}")

        # Record data if in recording mode
        if self.is_recording:
            current_time = time.time() - self.record_start_time

            # Raw sensor data
            self.time_history.append(current_time)
            self.raw_theta_history.append(motor_angle_rad)
            self.raw_alpha_history.append(pendulum_angle_rad)
            self.voltage_history.append(self.motor_voltage)

            # Model predictions
            if hasattr(self, 'model_state'):
                self.model_theta_history.append(self.model_state[0])
                self.model_alpha_history.append(self.model_state[1])
            else:
                self.model_theta_history.append(0)
                self.model_alpha_history.append(0)

            # Filtered data
            self.filtered_theta_history.append(self.filtered_state[0])
            self.filtered_alpha_history.append(self.filtered_state[1])


def main():
    print("Starting QUBE Controller with RL and Kalman Filter...")

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

        # Main loop
        while True:
            qube.update()
            app.update_gui()
            root.update()
            time.sleep(0.01)

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