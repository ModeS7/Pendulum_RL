import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog, messagebox
from QUBE import QUBE
import time
import numpy as np
import os
import csv
from datetime import datetime

# Update with your COM port
COM_PORT = "COM10"

# Control mode constants (from simulation)
EMERGENCY_MODE = 0
BANGBANG_MODE = 1
LQR_MODE = 2
ENERGY_MODE = 3


# Simplified parameters for real system
class SystemParameters:
    def __init__(self):
        # Control parameters
        self.max_voltage = 8.0  # Maximum voltage (V)
        self.theta_min = -2.0  # Minimum motor angle (rad)
        self.theta_max = 2.0  # Maximum motor angle (rad)
        self.balance_range = np.radians(20)  # Range where balance control activates (rad)

        # Energy control parameters (slightly adjusted for real system)
        self.m_p = 0.024  # Pendulum mass (kg)
        self.l = 0.129  # Pendulum length (m)
        self.l_com = self.l / 2  # Center of mass distance (m)
        self.J = (1 / 3) * self.m_p * self.l ** 2  # Moment of inertia (kg·m²)
        self.g = 9.81  # Gravity (m/s²)
        self.ke = 50.0  # Energy controller gain
        self.Er = 0.015  # Reference energy (J)

        # Derived energy parameters (from simulation)
        self.Mp_g_Lp = self.m_p * self.g * self.l  # used in energy calculations
        self.Jp = self.J  # Pendulum moment of inertia

        # LQR control parameters
        self.lqr_theta_gain = 3.0  # Gain for motor angle
        self.lqr_alpha_gain = 60.0  # Gain for pendulum angle
        self.lqr_theta_dot_gain = 2.5  # Gain for motor velocity
        self.lqr_alpha_dot_gain = 8.0  # Gain for pendulum velocity
        self.lqr_avoid_factor = 20.0  # Limit avoidance factor


class QUBEControllerWithBangBang:
    def __init__(self, master, qube):
        self.master = master
        self.qube = qube
        self.params = SystemParameters()
        master.title("QUBE Controller with Swing-Up and LQR Balance")

        # Control state
        self.motor_voltage = 0.0
        self.target_position = 0.0
        self.calibrating = False
        self.bang_bang_mode = False
        self.energy_mode = False
        self.lqr_mode = False
        self.moving_to_position = False
        self.current_controller_mode = EMERGENCY_MODE  # Track current controller

        # Data logging state
        self.data_logging = False
        self.csv_writer = None
        self.csv_file = None
        self.data_filename = ""
        self.log_counter = 0
        self.log_interval = 5  # Only log every 5 iterations (adjust for desired sampling rate)

        # Create GUI elements
        self.create_gui()

        # Tracking variables for calculations
        self.prev_time = time.time()
        self.prev_pendulum_angle = 0.0
        self.prev_motor_angle = 0.0

    def create_gui(self):
        # Main control frame
        control_frame = Frame(self.master, padx=10, pady=10)
        control_frame.pack()

        # Calibrate button
        self.calibrate_btn = Button(control_frame, text="Calibrate (Set Corner as Zero)",
                                    command=self.calibrate,
                                    width=25, height=2)
        self.calibrate_btn.grid(row=0, column=0, padx=5, pady=5)

        # Control mode frame
        control_mode_frame = Frame(control_frame)
        control_mode_frame.grid(row=1, column=0, pady=5)

        # Swing-up buttons
        self.bang_bang_btn = Button(control_mode_frame, text="Bang-Bang Swing-Up",
                                    command=self.toggle_bang_bang,
                                    width=15, height=2)
        self.bang_bang_btn.grid(row=0, column=0, padx=5, pady=5)

        self.energy_btn = Button(control_mode_frame, text="Energy Swing-Up",
                                 command=self.toggle_energy_control,
                                 width=15, height=2)
        self.energy_btn.grid(row=0, column=1, padx=5, pady=5)

        # Add LQR button
        self.lqr_btn = Button(control_mode_frame, text="LQR Balance",
                              command=self.toggle_lqr,
                              width=15, height=2)
        self.lqr_btn.grid(row=0, column=2, padx=5, pady=5)

        # Add Combined Control Button
        self.combined_btn = Button(control_mode_frame, text="Combined Control",
                                   command=self.toggle_combined_control,
                                   width=15, height=2)
        self.combined_btn.grid(row=0, column=3, padx=5, pady=5)

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

        # Control Parameters
        param_frame = Frame(control_frame)
        param_frame.grid(row=3, column=0, pady=10)

        # Bang-Bang parameters
        bb_frame = Frame(param_frame)
        bb_frame.grid(row=0, column=0, padx=10, pady=5)
        Label(bb_frame, text="Bang-Bang Parameters", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2)

        Label(bb_frame, text="Voltage:").grid(row=1, column=0, padx=5)
        self.bb_voltage_scale = Scale(bb_frame, from_=0, to=self.params.max_voltage,
                                      resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.bb_voltage_scale.set(self.params.max_voltage)
        self.bb_voltage_scale.grid(row=1, column=1, padx=5)

        # Energy control parameters
        energy_frame = Frame(param_frame)
        energy_frame.grid(row=0, column=1, padx=10, pady=5)
        Label(energy_frame, text="Energy Control Parameters", font=("Arial", 10, "bold")).grid(row=0, column=0,
                                                                                               columnspan=2)

        Label(energy_frame, text="Energy Gain:").grid(row=1, column=0, padx=5)
        self.ke_scale = Scale(energy_frame, from_=0, to=100.0,
                              resolution=1.0, orient=tk.HORIZONTAL, length=150)
        self.ke_scale.set(self.params.ke)
        self.ke_scale.grid(row=1, column=1, padx=5)

        Label(energy_frame, text="Reference Energy:").grid(row=2, column=0, padx=5)
        self.er_scale = Scale(energy_frame, from_=0, to=0.1,
                              resolution=0.001, orient=tk.HORIZONTAL, length=150)
        self.er_scale.set(self.params.Er)
        self.er_scale.grid(row=2, column=1, padx=5)

        # LQR parameters frame
        lqr_frame = Frame(param_frame)
        lqr_frame.grid(row=0, column=2, padx=10, pady=5)
        Label(lqr_frame, text="LQR Parameters", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2)

        Label(lqr_frame, text="Theta Gain:").grid(row=1, column=0, padx=5)
        self.lqr_theta_scale = Scale(lqr_frame, from_=0, to=20.0,
                                     resolution=0.5, orient=tk.HORIZONTAL, length=150)
        self.lqr_theta_scale.set(self.params.lqr_theta_gain)
        self.lqr_theta_scale.grid(row=1, column=1, padx=5)

        Label(lqr_frame, text="Alpha Gain:").grid(row=2, column=0, padx=5)
        self.lqr_alpha_scale = Scale(lqr_frame, from_=0, to=100.0,
                                     resolution=1.0, orient=tk.HORIZONTAL, length=150)
        self.lqr_alpha_scale.set(self.params.lqr_alpha_gain)
        self.lqr_alpha_scale.grid(row=2, column=1, padx=5)

        Label(lqr_frame, text="Theta Dot Gain:").grid(row=3, column=0, padx=5)
        self.lqr_theta_dot_scale = Scale(lqr_frame, from_=0, to=5.0,
                                         resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.lqr_theta_dot_scale.set(self.params.lqr_theta_dot_gain)
        self.lqr_theta_dot_scale.grid(row=3, column=1, padx=5)

        Label(lqr_frame, text="Alpha Dot Gain:").grid(row=4, column=0, padx=5)
        self.lqr_alpha_dot_scale = Scale(lqr_frame, from_=0, to=20.0,
                                         resolution=0.5, orient=tk.HORIZONTAL, length=150)
        self.lqr_alpha_dot_scale.set(self.params.lqr_alpha_dot_gain)
        self.lqr_alpha_dot_scale.grid(row=4, column=1, padx=5)

        # Common parameters
        common_frame = Frame(param_frame)
        common_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5)

        Label(common_frame, text="Balance Range (°):").grid(row=0, column=0, padx=5)
        self.balance_range_scale = Scale(common_frame, from_=0, to=45,
                                         resolution=1, orient=tk.HORIZONTAL, length=200)
        self.balance_range_scale.set(np.degrees(self.params.balance_range))
        self.balance_range_scale.grid(row=0, column=1, padx=5)

        # Update parameters button
        self.update_params_btn = Button(common_frame, text="Update Parameters",
                                        command=self.update_parameters, width=15)
        self.update_params_btn.grid(row=0, column=2, padx=10)

        # Stop button
        self.stop_btn = Button(control_frame, text="STOP MOTOR",
                               command=self.stop_motor,
                               width=20, height=2,
                               bg="red", fg="white")
        self.stop_btn.grid(row=4, column=0, pady=10)

        # Manual voltage control
        self.voltage_slider = Scale(
            control_frame,
            from_=-self.params.max_voltage,
            to=self.params.max_voltage,
            orient=tk.HORIZONTAL,
            label="Manual Voltage",
            length=400,
            resolution=0.1,
            command=self.set_manual_voltage
        )
        self.voltage_slider.set(0)
        self.voltage_slider.grid(row=5, column=0, padx=5, pady=10)

        # Status display
        status_frame = Frame(self.master, padx=10, pady=10)
        status_frame.pack()

        Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = Label(status_frame, text="Ready - Please calibrate", width=40)
        self.status_label.grid(row=0, column=1, sticky=tk.W)

        Label(status_frame, text="Control Mode:").grid(row=1, column=0, sticky=tk.W)
        self.mode_label = Label(status_frame, text="Manual", width=40)
        self.mode_label.grid(row=1, column=1, sticky=tk.W)

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

        # Add controller mode display
        Label(status_frame, text="Controller:").grid(row=6, column=0, sticky=tk.W)
        self.controller_label = Label(status_frame, text="None")
        self.controller_label.grid(row=6, column=1, sticky=tk.W)

        # Data Logging Panel
        logging_frame = Frame(self.master, padx=10, pady=10)
        logging_frame.pack()

        Label(logging_frame, text="Data Logging", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3)

        self.start_logging_btn = Button(logging_frame, text="Start Logging",
                                        command=self.start_data_logging,
                                        width=15, height=1)
        self.start_logging_btn.grid(row=1, column=0, padx=5, pady=5)

        self.stop_logging_btn = Button(logging_frame, text="Stop Logging",
                                       command=self.stop_data_logging,
                                       width=15, height=1, state="disabled")
        self.stop_logging_btn.grid(row=1, column=1, padx=5, pady=5)

        self.save_path_label = Label(logging_frame, text="No log file selected", width=40)
        self.save_path_label.grid(row=1, column=2, padx=5, pady=5)

        # Logging options
        log_options_frame = Frame(logging_frame)
        log_options_frame.grid(row=2, column=0, columnspan=3, pady=5)

        Label(log_options_frame, text="Log Interval (frames):").grid(row=0, column=0, padx=5)
        self.log_interval_scale = Scale(log_options_frame, from_=1, to=20,
                                        resolution=1, orient=tk.HORIZONTAL, length=150)
        self.log_interval_scale.set(self.log_interval)
        self.log_interval_scale.grid(row=0, column=1, padx=5)

        self.update_interval_btn = Button(log_options_frame, text="Update Interval",
                                          command=self.update_log_interval,
                                          width=15)
        self.update_interval_btn.grid(row=0, column=2, padx=10)

        # RGB Control
        rgb_frame = Frame(self.master, padx=10, pady=10)
        rgb_frame.pack()

        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

    def update_parameters(self):
        """Update control parameters from GUI"""
        self.params.max_voltage = self.bb_voltage_scale.get()
        self.params.balance_range = np.radians(self.balance_range_scale.get())
        self.params.ke = self.ke_scale.get()
        self.params.Er = self.er_scale.get()

        # Update LQR parameters
        self.params.lqr_theta_gain = self.lqr_theta_scale.get()
        self.params.lqr_alpha_gain = self.lqr_alpha_scale.get()
        self.params.lqr_theta_dot_gain = self.lqr_theta_dot_scale.get()
        self.params.lqr_alpha_dot_gain = self.lqr_alpha_dot_scale.get()

        self.status_label.config(text="Control parameters updated")

    # Helper function from simulation
    def clip_value(self, value, min_value, max_value):
        """Clip value to min/max range"""
        if value < min_value:
            return min_value
        elif value > max_value:
            return max_value
        else:
            return value

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def set_manual_voltage(self, value):
        """Set manual voltage from slider"""
        self.motor_voltage = float(value)
        # Reset any automatic control modes
        self.calibrating = False
        self.moving_to_position = False
        self.bang_bang_mode = False
        self.energy_mode = False
        self.lqr_mode = False
        self.combined_control_mode = False
        self.mode_label.config(text="Manual")
        self.current_controller_mode = EMERGENCY_MODE

    def calibrate(self):
        """Simple calibration - move to corner and set as zero"""
        self.calibrating = True
        self.moving_to_position = False
        self.bang_bang_mode = False
        self.energy_mode = False
        self.lqr_mode = False
        self.combined_control_mode = False
        self.voltage_slider.set(0)  # Reset slider
        self.current_controller_mode = EMERGENCY_MODE

        # Set calibration start time
        self.calibration_start_time = time.time()
        self.status_label.config(text="Calibrating - Finding corner...")
        self.mode_label.config(text="Calibrating")

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
            self.mode_label.config(text="Manual")

    def toggle_bang_bang(self):
        """Toggle bang-bang swing-up control mode"""
        if not self.bang_bang_mode:
            # Start bang-bang control
            self.bang_bang_mode = True
            self.energy_mode = False
            self.lqr_mode = False
            self.combined_control_mode = False
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.bang_bang_btn.config(text="Stop Bang-Bang")
            self.energy_btn.config(text="Energy Swing-Up")
            self.lqr_btn.config(text="LQR Balance")
            self.combined_btn.config(text="Combined Control")
            self.status_label.config(text="Bang-Bang swing-up active")
            self.mode_label.config(text="Bang-Bang Control")

            # Set orange LED during bang-bang control
            self.r_slider.set(999)
            self.g_slider.set(500)
            self.b_slider.set(0)
        else:
            # Stop bang-bang control
            self.bang_bang_mode = False
            self.motor_voltage = 0.0
            self.bang_bang_btn.config(text="Bang-Bang Swing-Up")
            self.status_label.config(text="Bang-Bang control stopped")
            self.mode_label.config(text="Manual")
            self.current_controller_mode = EMERGENCY_MODE

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def toggle_energy_control(self):
        """Toggle energy-based swing-up control mode"""
        if not self.energy_mode:
            # Start energy control
            self.energy_mode = True
            self.bang_bang_mode = False
            self.lqr_mode = False
            self.combined_control_mode = False
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.energy_btn.config(text="Stop Energy Control")
            self.bang_bang_btn.config(text="Bang-Bang Swing-Up")
            self.lqr_btn.config(text="LQR Balance")
            self.combined_btn.config(text="Combined Control")
            self.status_label.config(text="Energy-based swing-up active")
            self.mode_label.config(text="Energy Control")

            # Set purple LED during energy control
            self.r_slider.set(500)
            self.g_slider.set(0)
            self.b_slider.set(999)
        else:
            # Stop energy control
            self.energy_mode = False
            self.motor_voltage = 0.0
            self.energy_btn.config(text="Energy Swing-Up")
            self.status_label.config(text="Energy control stopped")
            self.mode_label.config(text="Manual")
            self.current_controller_mode = EMERGENCY_MODE

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def toggle_lqr(self):
        """Toggle LQR balance control mode"""
        if not self.lqr_mode:
            # Start LQR control
            self.lqr_mode = True
            self.energy_mode = False
            self.bang_bang_mode = False
            self.combined_control_mode = False
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.lqr_btn.config(text="Stop LQR Control")
            self.bang_bang_btn.config(text="Bang-Bang Swing-Up")
            self.energy_btn.config(text="Energy Swing-Up")
            self.combined_btn.config(text="Combined Control")
            self.status_label.config(text="LQR balance control active")
            self.mode_label.config(text="LQR Balance Control")

            # Set green LED during LQR control
            self.r_slider.set(0)
            self.g_slider.set(999)
            self.b_slider.set(400)
        else:
            # Stop LQR control
            self.lqr_mode = False
            self.motor_voltage = 0.0
            self.lqr_btn.config(text="LQR Balance")
            self.status_label.config(text="LQR control stopped")
            self.mode_label.config(text="Manual")
            self.current_controller_mode = EMERGENCY_MODE

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def toggle_combined_control(self):
        """Toggle combined control mode (using control_decision)"""
        if not hasattr(self, 'combined_control_mode') or not self.combined_control_mode:
            # Start combined control
            self.combined_control_mode = True
            self.lqr_mode = False
            self.energy_mode = False
            self.bang_bang_mode = False
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.combined_btn.config(text="Stop Combined Control")
            self.bang_bang_btn.config(text="Bang-Bang Swing-Up")
            self.energy_btn.config(text="Energy Swing-Up")
            self.lqr_btn.config(text="LQR Balance")
            self.status_label.config(text="Combined automatic control active")
            self.mode_label.config(text="Combined Control")

            # Set white LED during combined control
            self.r_slider.set(999)
            self.g_slider.set(999)
            self.b_slider.set(999)
        else:
            # Stop combined control
            self.combined_control_mode = False
            self.motor_voltage = 0.0
            self.combined_btn.config(text="Combined Control")
            self.status_label.config(text="Combined control stopped")
            self.mode_label.config(text="Manual")
            self.current_controller_mode = EMERGENCY_MODE

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def start_move_to_position(self):
        """Start moving to target position"""
        if not self.calibrating and not self.bang_bang_mode and not self.energy_mode and not self.lqr_mode and not self.combined_control_mode:
            try:
                # Get target position from entry field
                self.target_position = float(self.position_entry.get())

                self.moving_to_position = True
                self.bang_bang_mode = False
                self.energy_mode = False
                self.lqr_mode = False
                self.combined_control_mode = False
                self.voltage_slider.set(0)  # Reset slider
                self.status_label.config(text=f"Moving to {self.target_position:.1f}°...")
                self.mode_label.config(text="Moving to Position")
                self.current_controller_mode = EMERGENCY_MODE

                # Set green LED during movement
                self.r_slider.set(0)
                self.g_slider.set(999)
                self.b_slider.set(0)
            except ValueError:
                self.status_label.config(text="Invalid position value")

    def update_position_control(self):
        """Update position control using a simple P controller"""
        # Get current angle
        current_angle = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero

        # Calculate position error (difference from target)
        position_error = self.target_position - current_angle

        # Simple proportional control
        kp = 0.02  # Low gain
        self.motor_voltage = kp * position_error

        # Limit voltage for safety
        self.motor_voltage = max(-self.params.max_voltage,
                                 min(self.params.max_voltage, self.motor_voltage))

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
            self.mode_label.config(text="Manual")
            return

    # Implement bang-bang control from simulation (paste.txt)
    def simple_bang_bang(self, t, theta, alpha, theta_dot, alpha_dot):
        """Ultra-fast bang-bang controller for inverted pendulum swing-up"""
        # First, handle limit avoidance with highest priority
        limit_margin = 1.5
        if theta > self.params.theta_max - limit_margin and theta_dot > 0:
            return -self.params.max_voltage  # If close to upper limit, push back
        elif theta < self.params.theta_min + limit_margin and theta_dot < 0:
            return self.params.max_voltage  # If close to lower limit, push back

        pos_vel_same_sign = alpha * alpha_dot > 0
        if pos_vel_same_sign:
            # Apply torque against position to pump energy
            if alpha < 0:
                return -self.params.max_voltage
            else:
                return self.params.max_voltage
        else:
            # Apply torque with position
            if alpha < 0:
                return self.params.max_voltage
            else:
                return -self.params.max_voltage

    # Update bang-bang control using simulation algorithm
    def update_bang_bang_control(self):
        """Update bang-bang control for pendulum swing-up using simulation algorithm"""
        # Get current angles
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert to radians
        motor_angle = np.radians(motor_angle_deg)
        pendulum_angle = np.radians(pendulum_angle_deg)

        # Calculate velocities using finite difference
        current_time = time.time()
        dt = current_time - self.prev_time

        # Initialize previous values if needed
        if not hasattr(self, 'prev_pendulum_angle') or self.prev_pendulum_angle is None:
            self.prev_pendulum_angle = pendulum_angle
        if not hasattr(self, 'prev_motor_angle') or self.prev_motor_angle is None:
            self.prev_motor_angle = motor_angle

        # Calculate velocities
        pendulum_velocity = 0.0
        motor_velocity = 0.0
        if dt > 0:
            pendulum_velocity = (pendulum_angle - self.prev_pendulum_angle) / dt
            motor_velocity = (motor_angle - self.prev_motor_angle) / dt

        # Update previous values
        self.prev_pendulum_angle = pendulum_angle
        self.prev_motor_angle = motor_angle
        self.prev_time = current_time

        # Normalize pendulum angle for control
        pendulum_norm = self.normalize_angle(pendulum_angle)

        # Check if we've reached the upright position
        if abs(self.normalize_angle(pendulum_angle + np.pi)) < self.params.balance_range:
            # Switch to LQR for balancing
            if not self.lqr_mode:
                self.toggle_lqr()
                self.status_label.config(text="Switched to LQR for balance control")
                return

        # Use the simulation's bang-bang controller
        self.motor_voltage = self.simple_bang_bang(current_time, motor_angle, pendulum_norm,
                                                   motor_velocity, pendulum_velocity)

        # Update status based on angle from upright
        pendulum_upright = self.normalize_angle(pendulum_angle + np.pi)
        upright_deg = abs(pendulum_upright) * 180 / np.pi
        self.status_label.config(text=f"Bang-Bang control - {upright_deg:.1f}° from upright")

        # Set controller mode
        self.current_controller_mode = BANGBANG_MODE

    # Implement energy control from simulation (paste.txt)
    def energy_control(self, t, theta, alpha, theta_dot, alpha_dot):
        """Improved energy-based swing-up controller with enhanced limit avoidance"""
        # Calculate current energy and reference energy
        E_current = self.params.Mp_g_Lp * (1 - np.cos(alpha)) + 0.5 * self.params.Jp * alpha_dot ** 2
        E_ref = self.params.Er  # Energy target from GUI
        E_error = E_ref - E_current  # Positive when we need to add energy

        # Use standard energy pumping formula with sign(alpha_dot * sin(alpha))
        # This determines when to apply torque to efficiently add/remove energy
        pump_direction = np.sign(alpha_dot * np.sin(alpha))

        # Adaptive gain - use smaller gain when close to target energy for smoother control
        k_energy = self.params.ke * 0.01  # Scale down GUI gain value
        if abs(E_error) < 0.3 * self.params.Mp_g_Lp:
            k_energy = k_energy * 0.6  # More precise control when close to target

        # Energy pumping control
        u_energy = k_energy * E_error * pump_direction

        # Enhanced limit avoidance with velocity consideration
        # Use larger margin when moving quickly to account for momentum
        base_margin = 0.5
        velocity_factor = min(1.0, abs(theta_dot) / 2.0)
        dynamic_margin = base_margin * (1.0 + velocity_factor)

        # Calculate limit avoidance control
        u_limit = 0.0
        if theta > self.params.theta_max - dynamic_margin:
            # Stronger repulsion from upper limit when moving quickly
            distance_ratio = (theta - (self.params.theta_max - dynamic_margin)) / dynamic_margin
            u_limit = -12.0 * np.exp(distance_ratio) * (1.0 + velocity_factor)
        elif theta < self.params.theta_min + dynamic_margin:
            # Stronger repulsion from lower limit when moving quickly
            distance_ratio = ((self.params.theta_min + dynamic_margin) - theta) / dynamic_margin
            u_limit = 12.0 * np.exp(distance_ratio) * (1.0 + velocity_factor)

        # Add damping when pendulum has high velocity to prevent wild swings
        u_damping = -0.1 * alpha_dot if abs(alpha_dot) > 5.0 else 0.0

        # Combine all control components
        u = u_energy + u_limit + u_damping

        return self.clip_value(u, -self.params.max_voltage, self.params.max_voltage)

    # Update energy control using simulation algorithm
    def update_energy_control(self):
        """Update energy control using simulation algorithm"""
        # Get current angles
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert to radians
        motor_angle = np.radians(motor_angle_deg)
        pendulum_angle = np.radians(pendulum_angle_deg)

        # Calculate velocities using finite difference
        current_time = time.time()
        dt = current_time - self.prev_time

        # Initialize previous values if needed
        if not hasattr(self, 'prev_pendulum_angle') or self.prev_pendulum_angle is None:
            self.prev_pendulum_angle = pendulum_angle
        if not hasattr(self, 'prev_motor_angle') or self.prev_motor_angle is None:
            self.prev_motor_angle = motor_angle

        # Calculate velocities
        pendulum_velocity = 0.0
        motor_velocity = 0.0
        if dt > 0:
            pendulum_velocity = (pendulum_angle - self.prev_pendulum_angle) / dt
            motor_velocity = (motor_angle - self.prev_motor_angle) / dt

        # Update previous values
        self.prev_pendulum_angle = pendulum_angle
        self.prev_motor_angle = motor_angle
        self.prev_time = current_time

        # Normalize pendulum angle for physics calculations (down = 0)
        pendulum_norm = self.normalize_angle(pendulum_angle)

        # Check if pendulum is close to upright
        pendulum_upright = self.normalize_angle(pendulum_angle + np.pi)
        if abs(pendulum_upright) < self.params.balance_range:
            # Switch to LQR for balancing
            if not self.lqr_mode:
                self.toggle_lqr()
                self.status_label.config(text="Switched to LQR for balance control")
                return

        # Use the simulation's energy controller
        self.motor_voltage = self.energy_control(current_time, motor_angle, pendulum_norm,
                                                 motor_velocity, pendulum_velocity)

        # Calculate current energy for display
        E_current = self.params.Mp_g_Lp * (1 - np.cos(pendulum_norm)) + 0.5 * self.params.Jp * pendulum_velocity ** 2
        E_ref = self.params.Er

        # Update status based on energy and angle from upright
        upright_deg = abs(pendulum_upright) * 180 / np.pi
        self.status_label.config(
            text=f"Energy control - E={E_current:.3f}, E_ref={E_ref:.3f}, {upright_deg:.1f}° from upright")

        # Set controller mode
        self.current_controller_mode = ENERGY_MODE

    # Implement LQR control from simulation (paste.txt)
    def lqr_balance(self, theta, alpha, theta_dot, alpha_dot):
        """Ultra-fast LQR controller with theta limits consideration"""
        alpha_upright = self.normalize_angle(alpha - np.pi)

        # Regular LQR control with GUI-adjustable gains
        u = (self.params.lqr_theta_gain * theta +
             self.params.lqr_alpha_gain * alpha_upright +
             self.params.lqr_theta_dot_gain * theta_dot +
             self.params.lqr_alpha_dot_gain * alpha_dot) / 5.0

        # Add limit avoidance term
        limit_margin = 0.3
        if theta > self.params.theta_max - limit_margin:
            # Add strong negative control to avoid upper limit
            avoid_factor = self.params.lqr_avoid_factor * (
                        theta - (self.params.theta_max - limit_margin)) / limit_margin
            u -= avoid_factor
        elif theta < self.params.theta_min + limit_margin:
            # Add strong positive control to avoid lower limit
            avoid_factor = self.params.lqr_avoid_factor * (
                        (self.params.theta_min + limit_margin) - theta) / limit_margin
            u += avoid_factor

        return self.clip_value(u, -self.params.max_voltage, self.params.max_voltage)

    # Update LQR control using simulation algorithm
    def update_lqr_control(self):
        """LQR balance controller for pendulum using simulation algorithm"""
        # Get current angles
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert to radians
        motor_angle = np.radians(motor_angle_deg)
        pendulum_angle = np.radians(pendulum_angle_deg)

        # Calculate velocities using finite difference
        current_time = time.time()
        dt = current_time - self.prev_time

        # Initialize previous values if needed
        if not hasattr(self, 'prev_pendulum_angle') or self.prev_pendulum_angle is None:
            self.prev_pendulum_angle = pendulum_angle
        if not hasattr(self, 'prev_motor_angle') or self.prev_motor_angle is None:
            self.prev_motor_angle = motor_angle

        # Calculate velocities
        pendulum_velocity = 0.0
        motor_velocity = 0.0
        if dt > 0:
            pendulum_velocity = (pendulum_angle - self.prev_pendulum_angle) / dt
            motor_velocity = (motor_angle - self.prev_motor_angle) / dt

        # Update previous values
        self.prev_pendulum_angle = pendulum_angle
        self.prev_motor_angle = motor_angle
        self.prev_time = current_time

        # Normalize pendulum angle for LQR (we want pendulum_norm to be 0 at upright)
        pendulum_norm = self.normalize_angle(pendulum_angle + np.pi)

        # Check if we're still close enough to balance
        if abs(pendulum_norm) > 2 * self.params.balance_range:
            # Too far from upright, switch back to energy-based swing-up
            self.toggle_energy_control()
            self.status_label.config(text="Pendulum too far from upright, switching to swing-up")
            return

        # Use the simulation's LQR controller
        self.motor_voltage = self.lqr_balance(motor_angle, pendulum_angle, motor_velocity, pendulum_velocity)

        # Update status information
        upright_deg = abs(pendulum_norm) * 180 / np.pi
        self.status_label.config(text=f"LQR Balance - {upright_deg:.1f}° from upright")

        # Set controller mode
        self.current_controller_mode = LQR_MODE

    # Implement control decision function from simulation
    def control_decision(self):
        """Combined controller with controller mode tracking - adapted from simulation"""
        # Get current angles
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert to radians
        motor_angle = np.radians(motor_angle_deg)
        pendulum_angle = np.radians(pendulum_angle_deg)

        # Calculate velocities using finite difference
        current_time = time.time()
        dt = current_time - self.prev_time

        # Initialize previous values if needed
        if not hasattr(self, 'prev_pendulum_angle') or self.prev_pendulum_angle is None:
            self.prev_pendulum_angle = pendulum_angle
        if not hasattr(self, 'prev_motor_angle') or self.prev_motor_angle is None:
            self.prev_motor_angle = motor_angle

        # Calculate velocities
        pendulum_velocity = 0.0
        motor_velocity = 0.0
        if dt > 0:
            pendulum_velocity = (pendulum_angle - self.prev_pendulum_angle) / dt
            motor_velocity = (motor_angle - self.prev_motor_angle) / dt

        # Update previous values
        self.prev_pendulum_angle = pendulum_angle
        self.prev_motor_angle = motor_angle
        self.prev_time = current_time

        # Normalize pendulum angle (down = 0) for calculations
        pendulum_norm = self.normalize_angle(pendulum_angle)

        # Pendulum angle relative to upright (upright = 0)
        pendulum_upright = self.normalize_angle(pendulum_angle + np.pi)

        # Emergency limit handling - override all other controllers
        if (motor_angle >= self.params.theta_max and motor_velocity > 0) or (
                motor_angle <= self.params.theta_min and motor_velocity < 0):
            control_value = -self.params.max_voltage if motor_velocity > 0 else self.params.max_voltage
            self.motor_voltage = control_value
            self.current_controller_mode = EMERGENCY_MODE
            self.controller_label.config(text="Emergency Limit Control")
            return

        # Calculate current energy
        E_current = self.params.Mp_g_Lp * (1 - np.cos(pendulum_norm)) + 0.5 * self.params.Jp * pendulum_velocity ** 2

        # Check if energy is low (pendulum hanging down) - use bang-bang for initial swing
        if E_current > self.params.Mp_g_Lp * 1.1:
            control_value = self.energy_control(current_time, motor_angle, pendulum_norm,
                                                motor_velocity, pendulum_velocity)
            self.motor_voltage = control_value
            self.current_controller_mode = ENERGY_MODE
            self.controller_label.config(text="Energy Control")
            self.status_label.config(text=f"Energy - {abs(pendulum_upright) * 180 / np.pi:.1f}° from upright")
            return
        # If close to upright, use LQR
        elif abs(pendulum_upright) < 0.3:
            control_value = self.lqr_balance(motor_angle, pendulum_angle, motor_velocity, pendulum_velocity)
            self.motor_voltage = control_value
            self.current_controller_mode = LQR_MODE
            self.controller_label.config(text="LQR Balance Control")
            self.status_label.config(text=f"LQR - {abs(pendulum_upright) * 180 / np.pi:.1f}° from upright")
            return

        # Otherwise use energy-based control for swing-up
        else:
            control_value = self.simple_bang_bang(current_time, motor_angle, pendulum_norm,
                                                  motor_velocity, pendulum_velocity)
            self.motor_voltage = control_value
            self.current_controller_mode = BANGBANG_MODE
            self.controller_label.config(text="Bang-Bang Control")
            self.status_label.config(text=f"Bang-Bang - {abs(pendulum_upright) * 180 / np.pi:.1f}° from upright")
            return

    # Update combined control using control_decision function
    def update_combined_control(self):
        """Update using the automatic controller selection from simulation"""
        self.control_decision()

        # Set LED color based on active controller
        if self.current_controller_mode == EMERGENCY_MODE:
            self.r_slider.set(999)
            self.g_slider.set(0)
            self.b_slider.set(0)  # RED for emergency
        elif self.current_controller_mode == BANGBANG_MODE:
            self.r_slider.set(999)
            self.g_slider.set(500)
            self.b_slider.set(0)  # ORANGE for bang-bang
        elif self.current_controller_mode == LQR_MODE:
            self.r_slider.set(0)
            self.g_slider.set(999)
            self.b_slider.set(400)  # GREEN for LQR
        elif self.current_controller_mode == ENERGY_MODE:
            self.r_slider.set(500)
            self.g_slider.set(0)
            self.b_slider.set(999)  # PURPLE for energy

        # Update mode label
        mode_names = ["Emergency", "Bang-Bang", "LQR", "Energy"]
        self.mode_label.config(text=f"Combined Control: {mode_names[self.current_controller_mode]}")

    def start_data_logging(self):
        """Start logging data to CSV file"""
        try:
            # Create a timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"qube_data_{timestamp}.csv"

            # Ask user for save location
            self.data_filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=default_filename
            )

            if not self.data_filename:  # User canceled
                return

            # Open the file and create CSV writer
            self.csv_file = open(self.data_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # Write header row
            headers = [
                'Time(s)',
                'Motor_Angle(deg)',
                'Motor_Angle(rad)',
                'Pendulum_Angle(deg)',
                'Pendulum_Angle(rad)',
                'Pendulum_Angle_From_Upright(deg)',
                'Motor_Velocity(rad/s)',
                'Pendulum_Velocity(rad/s)',
                'Motor_Voltage(V)',
                'Controller_Mode',
                'Motor_RPM'
            ]
            self.csv_writer.writerow(headers)

            # Start logging
            self.data_logging = True
            self.log_counter = 0
            self.log_start_time = time.time()

            # Update UI
            self.start_logging_btn.config(state="disabled")
            self.stop_logging_btn.config(state="normal")
            self.save_path_label.config(text=f"Logging to: {os.path.basename(self.data_filename)}")
            self.status_label.config(text=f"Data logging started")

        except Exception as e:
            messagebox.showerror("Logging Error", f"Could not start logging: {str(e)}")
            self.stop_data_logging()

    def stop_data_logging(self):
        """Stop logging data"""
        if self.data_logging:
            try:
                # Close the file
                if self.csv_file:
                    self.csv_file.close()

                # Update state
                self.data_logging = False
                self.csv_writer = None
                self.csv_file = None

                # Update UI
                self.start_logging_btn.config(state="normal")
                self.stop_logging_btn.config(state="disabled")
                self.save_path_label.config(text=f"Logging stopped. File saved.")
                self.status_label.config(text=f"Data logging stopped")

            except Exception as e:
                messagebox.showerror("Logging Error", f"Error while stopping logging: {str(e)}")

    def update_log_interval(self):
        """Update the logging interval"""
        self.log_interval = self.log_interval_scale.get()
        self.status_label.config(text=f"Log interval updated to {self.log_interval} frames")

    def log_data_point(self):
        """Log a single data point to the CSV file"""
        if not self.data_logging or not self.csv_writer:
            return

        # Only log every 'log_interval' frames to control file size
        self.log_counter += 1
        if self.log_counter < self.log_interval:
            return

        self.log_counter = 0

        try:
            # Get current data
            current_time = time.time() - self.log_start_time
            motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
            pendulum_angle_deg = self.qube.getPendulumAngle()
            motor_angle_rad = np.radians(motor_angle_deg)
            pendulum_angle_rad = np.radians(pendulum_angle_deg)

            # Calculate derived values
            pendulum_upright = self.normalize_angle(pendulum_angle_rad + np.pi)
            upright_angle_deg = abs(pendulum_upright) * 180 / np.pi

            # Get velocities (if available)
            motor_velocity = 0
            pendulum_velocity = 0

            if hasattr(self, 'prev_pendulum_angle') and hasattr(self, 'prev_motor_angle') and hasattr(self,
                                                                                                      'prev_time'):
                dt = current_time - (time.time() - self.prev_time)
                if dt > 0:
                    motor_velocity = (motor_angle_rad - self.prev_motor_angle) / dt
                    pendulum_velocity = (pendulum_angle_rad - self.prev_pendulum_angle) / dt

            # Get controller mode name
            mode_names = ["Emergency", "Bang-Bang", "LQR", "Energy"]
            controller_mode = mode_names[self.current_controller_mode]

            # Prepare row data
            row_data = [
                f"{current_time:.3f}",
                f"{motor_angle_deg:.3f}",
                f"{motor_angle_rad:.3f}",
                f"{pendulum_angle_deg:.3f}",
                f"{pendulum_angle_rad:.3f}",
                f"{upright_angle_deg:.3f}",
                f"{motor_velocity:.3f}",
                f"{pendulum_velocity:.3f}",
                f"{self.motor_voltage:.3f}",
                controller_mode,
                f"{self.qube.getMotorRPM():.1f}"
            ]

            # Write to CSV
            self.csv_writer.writerow(row_data)

        except Exception as e:
            print(f"Error logging data: {str(e)}")

    def stop_motor(self):
        """Stop the motor"""
        self.calibrating = False
        self.moving_to_position = False
        self.bang_bang_mode = False
        self.energy_mode = False
        self.lqr_mode = False
        if hasattr(self, 'combined_control_mode'):
            self.combined_control_mode = False
        self.motor_voltage = 0.0
        self.voltage_slider.set(0)
        self.bang_bang_btn.config(text="Bang-Bang Swing-Up")
        self.energy_btn.config(text="Energy Swing-Up")
        self.lqr_btn.config(text="LQR Balance")
        if hasattr(self, 'combined_btn'):
            self.combined_btn.config(text="Combined Control")
        self.current_controller_mode = EMERGENCY_MODE

        # Set blue LED when stopped
        self.r_slider.set(0)
        self.g_slider.set(0)
        self.b_slider.set(999)

        self.status_label.config(text="Motor stopped")
        self.mode_label.config(text="Manual")
        self.controller_label.config(text="None")

    def update_gui(self):
        """Update the GUI and control the hardware - called continuously"""
        # Update automatic control modes if active
        if self.calibrating:
            self.update_calibration()
        elif self.moving_to_position:
            self.update_position_control()
        elif self.bang_bang_mode:
            self.update_bang_bang_control()
        elif self.energy_mode:
            self.update_energy_control()
        elif self.lqr_mode:
            self.update_lqr_control()
        elif hasattr(self, 'combined_control_mode') and self.combined_control_mode:
            self.update_combined_control()

        # Apply the current motor voltage - THIS IS CRITICAL TO DO ON EVERY LOOP!
        self.qube.setMotorVoltage(self.motor_voltage)

        # Apply RGB values
        self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())

        # Update display information
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Get pendulum angle relative to upright position
        pendulum_angle_rad = np.radians(pendulum_angle_deg)
        pendulum_norm = self.normalize_angle(pendulum_angle_rad + np.pi)
        upright_angle = abs(pendulum_norm) * 180 / np.pi

        rpm = self.qube.getMotorRPM()

        self.angle_label.config(text=f"{motor_angle_deg:.1f}°")
        self.pendulum_label.config(
            text=f"{pendulum_angle_deg:.1f}° ({upright_angle:.1f}° from upright)")
        self.rpm_label.config(text=f"{rpm:.1f}")
        self.voltage_label.config(text=f"{self.motor_voltage:.1f} V")

        # Log data if enabled
        if self.data_logging:
            self.log_data_point()


def main():
    print("Starting QUBE Controller with Advanced Control Algorithms...")
    print("Will set corner position as zero")

    app = None

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
        app = QUBEControllerWithBangBang(root, qube)

        # Main loop
        while True:
            qube.update()
            app.update_gui()
            root.update()
            time.sleep(0.01)  # 100Hz update rate

    except tk.TclError:
        # Window closed
        print("Window closed")
    except KeyboardInterrupt:
        # User pressed Ctrl+C
        print("User interrupted program")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up data logging
        if app and app.data_logging:
            try:
                app.stop_data_logging()
                print("Data logging stopped and file closed properly")
            except:
                print("Error while stopping data logging")

        # Final stop attempt
        try:
            qube.setMotorVoltage(0.0)
            qube.update()
            print("Motor stopped")
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")


if __name__ == "__main__":
    main()