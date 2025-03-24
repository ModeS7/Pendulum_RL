import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog
from QUBE import QUBE
import time
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
# Update with your COM port
COM_PORT = "COM10"


class SimpleSimulator:
    """Simple simulator for QUBE-Servo 2 system to use for comparison with real hardware"""

    def __init__(self, dt=0.0115):
        # Base parameters
        self.dt = dt  # Default time step, but will use actual time steps from real data

        # Motor and pendulum parameters - fixed values for simulation
        self.Rm = 8.94  # Motor resistance (Ohm)
        self.Km = 0.0431  # Motor back-emf constant
        self.Jm = 6e-5  # Motor inertia (kg·m^2)
        self.bm = 3e-4  # Motor damping coefficient (Nm/rad/s)
        self.mA = 0.053  # Weight of pendulum arm (kg)
        self.mL = 0.024  # Weight of pendulum link (kg)
        self.LA = 0.086  # Length of pendulum arm (m)
        self.LL = 0.128  # Length of pendulum link (m)
        self.g = 9.81  # Gravity constant (m/s^2)

        # Derived parameters
        self.half_mL_LL_g = 0.5 * self.mL * self.LL * self.g
        self.half_mL_LL_LA = 0.5 * self.mL * self.LL * self.LA
        self.quarter_mL_LL_squared = 0.25 * self.mL * self.LL ** 2

        # State variables [theta, alpha, theta_dot, alpha_dot]
        self.state = np.zeros(4)

        # Track simulation time
        self.time = 0.0

    def reset(self):
        """Reset the simulation to initial state"""
        self.state = np.zeros(4)
        self.time = 0.0
        return self.state.copy()

    def step(self, voltage, dt=None):
        """Run one simulation step with the given voltage and time step"""
        # Use provided dt if given, otherwise use default
        if dt is None:
            dt = self.dt

        # Limit voltage
        voltage = max(-10.0, min(10.0, voltage))

        # Extract current state
        theta, alpha, theta_dot, alpha_dot = self.state

        # Calculate motor torque
        im = (voltage - self.Km * theta_dot) / self.Rm  # Motor current
        Tm = self.Km * im  # Motor torque

        # Equations of motion coefficients
        M11 = self.mL * self.LA ** 2 + self.quarter_mL_LL_squared - self.quarter_mL_LL_squared * np.cos(
            alpha) ** 2 + 0.5 * self.Jm
        M12 = -self.half_mL_LL_LA * np.cos(alpha)
        C1 = 0.5 * self.mL * self.LL ** 2 * np.sin(alpha) * np.cos(alpha) * theta_dot * alpha_dot
        C2 = self.half_mL_LL_LA * np.sin(alpha) * alpha_dot ** 2

        M21 = self.half_mL_LL_LA * np.cos(alpha)
        M22 = 0.25 * self.mL * self.LL ** 2
        C3 = -self.quarter_mL_LL_squared * np.cos(alpha) * np.sin(alpha) * theta_dot ** 2
        G = self.half_mL_LL_g * np.sin(alpha)

        # Calculate determinant for matrix inversion
        det_M = M11 * M22 - M12 * M21

        # Handle near-singular matrix
        if abs(det_M) < 1e-10:
            theta_ddot, alpha_ddot = 0, 0
        else:
            # Right-hand side of equations
            RHS1 = Tm - C1 - C2 - self.bm * theta_dot
            RHS2 = -G - 0.0005 * alpha_dot - C3  # Added small pendulum damping

            # Solve for accelerations
            theta_ddot = (M22 * RHS1 - M12 * RHS2) / det_M
            alpha_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M

        # Update state using simple Euler integration with the actual time step
        theta_new = theta + theta_dot * dt
        alpha_new = alpha + alpha_dot * dt
        theta_dot_new = theta_dot + theta_ddot * dt
        alpha_dot_new = alpha_dot + alpha_ddot * dt



        # Update state
        self.state = np.array([theta_new, alpha_new, theta_dot_new, alpha_dot_new])
        self.time += dt

        return self.state.copy()

    def run_step_test(self, voltage, duration, cooldown=1.0, time_steps=None):
        """
        Run a complete step voltage test simulation

        Args:
            voltage: Voltage to apply
            duration: Duration to apply voltage
            cooldown: Time after voltage is removed to continue recording
            time_steps: List of actual time steps from real data (if None, uses fixed dt)
        """
        # Reset simulation
        self.reset()

        # Prepare for data collection
        data = []

        if time_steps is None:
            # Use fixed time step if not provided
            total_steps = int((duration + cooldown) / self.dt)
            time_points = [i * self.dt for i in range(total_steps)]
            dt_values = [self.dt] * total_steps
        else:
            # Use provided time steps
            time_points = np.cumsum(time_steps)
            dt_values = time_steps

        # Run simulation loop
        current_time = 0.0
        for i, dt in enumerate(dt_values):
            # Apply voltage during test duration
            if current_time < duration:
                v = voltage
            else:
                v = 0.0

            # Step simulation with actual time step
            self.step(v, dt)

            # Record data
            data.append({
                'time': current_time,
                'motor_angle': self.state[0] * (180 / np.pi),  # Convert to degrees
                'pendulum_angle': self.state[1] * (180 / np.pi),  # Convert to degrees
                'motor_rpm': self.state[2] * (60 / (2 * np.pi)),  # Convert rad/s to RPM
                'voltage': v
            })

            current_time += dt

        # Return as DataFrame
        return pd.DataFrame(data)



# Actor Network - same architecture as in training script
class Actor(nn.Module):
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
        self.max_voltage = 10.0  # Use full voltage range as in training

        # Initialize the RL model (but don't load weights yet)
        self.initialize_rl_model()

        # Create GUI elements
        self.create_gui()

    def initialize_rl_model(self):
        """Initialize the RL model architecture"""
        state_dim = 6  # Our observation space (same as in training)
        action_dim = 1  # Motor voltage (normalized)
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
        # Existing GUI code (keep all your original code)

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

        # Move to position input and button
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

        Label(status_frame, text="Motor RPM:").grid(row=4, column=0, sticky=tk.W)
        self.rpm_label = Label(status_frame, text="0.0")
        self.rpm_label.grid(row=4, column=1, sticky=tk.W)

        Label(status_frame, text="Current Voltage:").grid(row=5, column=0, sticky=tk.W)
        self.voltage_label = Label(status_frame, text="0.0 V")
        self.voltage_label.grid(row=5, column=1, sticky=tk.W)

        # RGB Control
        rgb_frame = Frame(self.master, padx=10, pady=10)
        rgb_frame.pack()

        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

        # ADD THIS SECTION: Step Test Controls
        step_test_frame = Frame(self.master, padx=10, pady=10)
        step_test_frame.pack()

        Label(step_test_frame, text="Step Voltage Test", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=5,
                                                                                          pady=5)

        Label(step_test_frame, text="Voltage (V):").grid(row=1, column=0, padx=5)
        self.test_voltage_entry = Entry(step_test_frame, width=8)
        self.test_voltage_entry.grid(row=1, column=1, padx=5)
        self.test_voltage_entry.insert(0, "3.0")

        Label(step_test_frame, text="Duration (s):").grid(row=1, column=2, padx=5)
        self.test_duration_entry = Entry(step_test_frame, width=8)
        self.test_duration_entry.grid(row=1, column=3, padx=5)
        self.test_duration_entry.insert(0, "1.0")

        self.run_test_btn = Button(step_test_frame, text="Run Step Test",
                                   command=self.start_step_test, width=15)
        self.run_test_btn.grid(row=1, column=4, padx=5)

        # Create test status label
        self.test_status_label = Label(step_test_frame, text="", width=50)
        self.test_status_label.grid(row=2, column=0, columnspan=5, pady=5)

        # Add test state variables
        self.testing = False
        self.test_data = []
        self.test_voltage = 0.0
        self.test_duration = 0.0
        self.test_start_time = 0.0
        self.test_cooldown = 0.0  # Time after test to record settling

    def start_step_test(self):
        """Start a step voltage test"""
        if self.calibrating or self.moving_to_position or self.rl_mode or self.testing:
            self.test_status_label.config(text="Cannot start test - another operation in progress")
            return

        try:
            # Get test parameters
            self.test_voltage = float(self.test_voltage_entry.get())
            self.test_duration = float(self.test_duration_entry.get())

            # Check valid parameters
            if abs(self.test_voltage) > self.max_voltage:
                self.test_status_label.config(text=f"Voltage limited to ±{self.max_voltage}V")
                self.test_voltage = max(-self.max_voltage, min(self.max_voltage, self.test_voltage))
                self.test_voltage_entry.delete(0, tk.END)
                self.test_voltage_entry.insert(0, str(self.test_voltage))

            if self.test_duration <= 0 or self.test_duration > 5.0:
                self.test_status_label.config(text="Duration must be between 0 and 5 seconds")
                return

            # Reset data collection
            self.test_data = []
            self.testing = True
            self.test_start_time = time.time()
            self.motor_voltage = 0.0  # Start from zero
            self.voltage_slider.set(0)
            self.test_cooldown = min(2.0, self.test_duration * 2)  # Record cooldown period

            # Set orange LED for testing
            self.r_slider.set(999)
            self.g_slider.set(400)
            self.b_slider.set(0)

            self.test_status_label.config(text=f"Running {self.test_voltage}V step test for {self.test_duration}s...")
            self.status_label.config(text="Step test in progress")

        except ValueError:
            self.test_status_label.config(text="Invalid test parameters")

    def update_step_test(self):
        """Update step test data collection"""
        current_time = time.time()
        elapsed = current_time - self.test_start_time

        # Apply voltage during test
        if elapsed < self.test_duration:
            self.motor_voltage = self.test_voltage
        else:
            self.motor_voltage = 0.0

        # Collect data point (approximately every 10ms)
        data_point = {
            'time': elapsed,
            'motor_angle': self.qube.getMotorAngle() + 136.0,  # Adjusted angle
            'pendulum_angle': self.qube.getPendulumAngle(),
            'motor_rpm': self.qube.getMotorRPM(),
            'voltage': self.motor_voltage
        }
        self.test_data.append(data_point)

        # Update status during test
        if elapsed < self.test_duration:
            remaining = self.test_duration - elapsed
            self.test_status_label.config(text=f"Applying {self.test_voltage}V... ({remaining:.1f}s remaining)")
        else:
            cooldown_remaining = (self.test_duration + self.test_cooldown) - elapsed
            if cooldown_remaining > 0:
                self.test_status_label.config(text=f"Recording response... ({cooldown_remaining:.1f}s)")

        # End test after duration + cooldown
        if elapsed > (self.test_duration + self.test_cooldown):
            self.motor_voltage = 0.0
            self.testing = False

            # Set blue LED when done
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

            self.test_status_label.config(text="Test complete. Processing results...")

            # Process and save results
            self.process_test_results()

    def process_test_results(self):
        """Process and save test results, run simulation, and plot comparison"""
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(self.test_data)

            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save real data
            filename = f"step_test_{self.test_voltage}V_{self.test_duration}s_{timestamp}.csv"
            df.to_csv(filename, index=False)

            # Run simulation comparison
            self.run_simulation_comparison(df, timestamp)

            # Update status
            self.test_status_label.config(text=f"Test complete. Data saved to {filename}")
            self.status_label.config(text="Ready")

        except Exception as e:
            self.test_status_label.config(text=f"Error processing results: {str(e)}")

    def run_simulation_comparison(self, real_data, timestamp):
        """Run simulation with same parameters for comparison"""
        try:
            # Extract test parameters
            voltage = self.test_voltage
            duration = self.test_duration
            cooldown = self.test_cooldown

            # Calculate actual time steps from real data
            real_times = real_data['time'].values
            time_steps = []
            for i in range(1, len(real_times)):
                time_steps.append(real_times[i] - real_times[i - 1])

            # Calculate average time step for info
            avg_dt = np.mean(time_steps)

            # Create simulator with fixed parameters
            simulator = SimpleSimulator()

            # Run simulation with actual time steps from real data
            sim_data = simulator.run_step_test(voltage, duration, cooldown, time_steps=time_steps)

            # Save simulation data
            sim_filename = f"step_test_{voltage}V_{duration}s_{timestamp}_sim.csv"
            sim_data.to_csv(sim_filename, index=False)

            # Create comparison plot
            plt.figure(figsize=(12, 10))

            # Plot motor angle
            plt.subplot(4, 1, 1)
            plt.plot(real_data['time'], real_data['motor_angle'], 'b-', label='Real')
            plt.plot(sim_data['time'], sim_data['motor_angle'], 'r--', label='Simulation')
            plt.ylabel('Motor Angle (deg)')
            plt.title(f'Step Test Comparison: {voltage}V for {duration}s (avg dt={avg_dt * 1000:.2f}ms)')
            plt.legend()
            plt.grid(True)

            # Plot pendulum angle
            plt.subplot(4, 1, 2)
            plt.plot(real_data['time'], real_data['pendulum_angle'], 'b-', label='Real')
            plt.plot(sim_data['time'], sim_data['pendulum_angle'], 'r--', label='Simulation')
            plt.ylabel('Pendulum Angle (deg)')
            plt.legend()
            plt.grid(True)

            # Plot motor RPM
            plt.subplot(4, 1, 3)
            plt.plot(real_data['time'], real_data['motor_rpm'], 'b-', label='Real')
            plt.plot(sim_data['time'], sim_data['motor_rpm'], 'r--', label='Simulation')
            plt.ylabel('Motor RPM')
            plt.legend()
            plt.grid(True)

            # Plot voltage
            plt.subplot(4, 1, 4)
            plt.plot(real_data['time'], real_data['voltage'], 'g-')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')
            plt.grid(True)

            plt.tight_layout()
            plot_filename = f"step_test_{voltage}V_{duration}s_{timestamp}_comparison.png"
            plt.savefig(plot_filename)
            plt.close()

            self.test_status_label.config(text=f"Comparison saved to {plot_filename}")

        except Exception as e:
            self.test_status_label.config(text=f"Simulation error: {str(e)}")
            print(f"Simulation error: {str(e)}")
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

            self.status_label.config(text="Position reached")
            return

        # Simple proportional control
        kp = 0.02  # Low gain
        self.motor_voltage = kp * position_error

        # Limit voltage for safety
        self.motor_voltage = max(-self.max_voltage, min(self.max_voltage, self.motor_voltage))

    def update_rl_control(self):
        """Update RL control logic - let the model handle everything from the start"""
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
                pendulum_velocity = (current_pendulum_angle_rad - self.prev_pendulum_angle_rl) / dt
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

        # Get action from RL model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            action_mean, _ = self.actor(state_tensor)
            action = action_mean.cpu().numpy()[0][0]  # Get action as scalar

        # Convert normalized action [-1, 1] to voltage
        self.motor_voltage = float(action) * self.max_voltage

        # Update status
        upright_angle = abs(pendulum_angle_norm) * 180 / np.pi
        if upright_angle < 30:
            self.status_label.config(text=f"RL control active - Near balance ({upright_angle:.1f}° from upright)")
        else:
            self.status_label.config(text=f"RL control active - Swinging ({upright_angle:.1f}° from upright)")

    # This function is no longer needed as the RL model handles everything

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
        # Update automatic control modes if active
        if self.calibrating:
            self.update_calibration()
        elif self.moving_to_position:
            self.update_position_control()
        elif self.rl_mode:
            self.update_rl_control()
        elif self.testing:
            self.update_step_test()  # Add this condition for step test

        # Apply the current motor voltage - THIS IS CRITICAL TO DO ON EVERY LOOP!
        self.qube.setMotorVoltage(self.motor_voltage)

        # Apply RGB values
        self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())

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

        if self.rl_model:
            self.model_label.config(text=f"Using: {os.path.basename(self.rl_model)}")


def run_standalone_comparison(real_data_file):
    """Run a standalone comparison between real data and simulation"""
    try:
        # Load real data
        real_df = pd.read_csv(real_data_file)
        print(f"Loaded real data from {real_data_file}")

        # Extract test parameters
        voltage = real_df['voltage'].max()
        duration = real_df[real_df['voltage'] > 0]['time'].max()
        cooldown = real_df['time'].max() - duration

        # Calculate actual time steps from real data
        real_times = real_df['time'].values
        time_steps = []
        for i in range(1, len(real_times)):
            time_steps.append(real_times[i] - real_times[i - 1])

        # Calculate average time step
        avg_dt = np.mean(time_steps)
        print(f"Test parameters: {voltage}V for {duration:.2f}s with {cooldown:.2f}s cooldown")
        print(f"Average time step: {avg_dt * 1000:.2f}ms")

        # Create simulator with fixed parameters
        simulator = SimpleSimulator()

        # Run simulation with actual time steps
        print("Running simulation with same time steps as real data...")
        sim_df = simulator.run_step_test(voltage, duration, cooldown, time_steps=time_steps)
        sim_df = simulator.run_step_test(voltage, duration, cooldown)

        # Save simulation data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_filename = f"step_test_{voltage}V_{duration:.2f}s_sim_{timestamp}.csv"
        sim_df.to_csv(sim_filename, index=False)
        print(f"Saved simulation data to {sim_filename}")

        # Calculate RMSE for motor angle
        from scipy.interpolate import interp1d
        f_motor = interp1d(sim_df['time'], sim_df['motor_angle'], bounds_error=False, fill_value=np.nan)
        motor_angle_sim = f_motor(real_df['time'])
        valid_idx = ~np.isnan(motor_angle_sim)
        motor_rmse = np.sqrt(np.mean((real_df['motor_angle'][valid_idx] - motor_angle_sim[valid_idx]) ** 2))

        # Create comparison plot
        plt.figure(figsize=(12, 10))

        # Plot motor angle
        plt.subplot(4, 1, 1)
        plt.plot(real_df['time'], real_df['motor_angle'], 'b-', label='Real')
        plt.plot(sim_df['time'], sim_df['motor_angle'], 'r--', label='Simulation')
        plt.ylabel('Motor Angle (deg)')
        plt.title(f'Step Test Comparison: {voltage}V for {duration:.2f}s\nMotor Angle RMSE: {motor_rmse:.2f}°')
        plt.legend()
        plt.grid(True)

        # Plot pendulum angle
        plt.subplot(4, 1, 2)
        plt.plot(real_df['time'], real_df['pendulum_angle'], 'b-', label='Real')
        plt.plot(sim_df['time'], sim_df['pendulum_angle'], 'r--', label='Simulation')
        plt.ylabel('Pendulum Angle (deg)')
        plt.legend()
        plt.grid(True)

        # Plot motor RPM
        plt.subplot(4, 1, 3)
        plt.plot(real_df['time'], real_df['motor_rpm'], 'b-', label='Real')
        plt.plot(sim_df['time'], sim_df['motor_rpm'], 'r--', label='Simulation')
        plt.ylabel('Motor RPM')
        plt.legend()
        plt.grid(True)

        # Plot voltage
        plt.subplot(4, 1, 4)
        plt.plot(real_df['time'], real_df['voltage'], 'g-')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.grid(True)

        plt.tight_layout()
        comparison_filename = f"comparison_{voltage}V_{duration:.2f}s_{timestamp}.png"
        plt.savefig(comparison_filename)
        print(f"Saved comparison plot to {comparison_filename}")
        plt.show()

        return real_df, sim_df

    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

def main():
    print("Starting QUBE Controller with RL...")
    print("Will set corner position as zero")

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
    # Check if running in standalone comparison mode
    if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
        parser = argparse.ArgumentParser(description='QUBE step test comparison tool')
        parser.add_argument('data_file', help='CSV file with real test data')

        args = parser.parse_args()
        run_standalone_comparison(args.data_file)
    else:
        # Run the normal QUBE controller
        main()