import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog
from QUBE import QUBE
import time
import numpy as np
import os

# Update with your COM port
COM_PORT = "COM10"


# PID Controller class
class PIDController:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, dt=0.01, output_limits=None):
        """
        PID Controller implementation

        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            dt (float): Time step in seconds
            output_limits (tuple): (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits

        # Controller state
        self.last_error = 0.0
        self.integral = 0.0
        self.setpoint = 0.0

    def set_gains(self, kp, ki, kd):
        """Set PID gains"""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_setpoint(self, setpoint):
        """Set the target value"""
        self.setpoint = setpoint
        # Reset integral when setpoint changes to prevent integral windup
        self.integral = 0.0

    def reset(self):
        """Reset controller state"""
        self.last_error = 0.0
        self.integral = 0.0

    def compute(self, measurement, dt=None):
        """
        Compute control output based on measurement

        Args:
            measurement (float): Current process value
            dt (float): Time step override (optional)

        Returns:
            float: Control output
        """
        if dt is None:
            dt = self.dt

        # Calculate error
        error = self.setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term (on measurement, not error, to avoid derivative kicks)
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative

        # Save error for next iteration
        self.last_error = error

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits if specified
        if self.output_limits:
            output = max(self.output_limits[0], min(self.output_limits[1], output))

        return output


class QUBEControllerWithPID:
    def __init__(self, master, qube):
        self.master = master
        self.qube = qube
        master.title("QUBE Controller with PID Control")

        # Control state
        self.motor_voltage = 0.0
        self.target_position = 0.0
        self.calibrating = False
        self.pid_mode = False
        self.balance_mode = False
        self.moving_to_position = False
        self.max_voltage = 8.0  # Slightly less than max for safety

        # Physical parameters (from C++ code)
        self.m_p = 0.1  # Pendulum stick mass (kg)
        self.l = 0.095  # Length of pendulum (m)
        self.l_com = self.l / 2  # Distance to center of mass (m)
        self.J = (1 / 3) * self.m_p * self.l * self.l  # Inertia (kg/m^2)
        self.g = 9.81  # Gravity (m/s^2)
        self.balance_range = 35.0  # Range where mode switches to balancing (degrees)

        # PID Controllers setup
        # For motor position control
        self.position_pid = PIDController(
            kp=0.07,  # Based on C++ kp_pos
            ki=0.01,  # Adding some integral action
            kd=0.06,  # Based on C++ kd_pos
            dt=0.01,  # 100Hz update rate
            output_limits=(-self.max_voltage, self.max_voltage)
        )

        # For pendulum balance
        self.pendulum_pid = PIDController(
            kp=2.0,  # Based on C++ kp_theta
            ki=0.0,  # No integral for faster response
            kd=0.125,  # Based on C++ kd_theta
            dt=0.01,  # 100Hz update rate
            output_limits=(-self.max_voltage, self.max_voltage)
        )

        # For energy-based swing-up
        self.energy_controller = {
            'ke': 50.0,  # Energy gain (from C++ code)
            'Er': 0.015,  # Reference energy (from C++ code)
            'u_max': 3.0,  # Maximum acceleration (from C++ code)
            'prev_angle': 0.0,  # For velocity calculation
            'prev_time': 0.0  # For velocity calculation
        }

        # Create GUI elements
        self.create_gui()

        # Tracking variables for velocity calculation
        self.prev_pendulum_angle = 0.0
        self.prev_motor_angle = 0.0
        self.prev_time = time.time()

    def create_gui(self):
        # Main control frame
        control_frame = Frame(self.master, padx=10, pady=10)
        control_frame.pack()

        # Calibrate button
        self.calibrate_btn = Button(control_frame, text="Calibrate (Set Corner as Zero)",
                                    command=self.calibrate,
                                    width=25, height=2)
        self.calibrate_btn.grid(row=0, column=0, padx=5, pady=5)

        # PID Control buttons
        pid_frame = Frame(control_frame)
        pid_frame.grid(row=1, column=0, pady=10)

        self.pid_control_btn = Button(pid_frame, text="Start Position PID",
                                      command=self.toggle_pid_control,
                                      width=15)
        self.pid_control_btn.grid(row=0, column=0, padx=5)

        self.balance_btn = Button(pid_frame, text="Start Balance Control",
                                  command=self.toggle_balance_control,
                                  width=15)
        self.balance_btn.grid(row=0, column=1, padx=5)

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

        # PID Tuning frame
        tuning_frame = Frame(control_frame)
        tuning_frame.grid(row=3, column=0, pady=10)

        # Position PID Tuning
        pos_frame = Frame(tuning_frame)
        pos_frame.grid(row=0, column=0, padx=10, pady=5)
        Label(pos_frame, text="Position PID", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2)

        Label(pos_frame, text="P:").grid(row=1, column=0)
        self.pos_p_scale = Scale(pos_frame, from_=0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=150)
        self.pos_p_scale.set(self.position_pid.kp)
        self.pos_p_scale.grid(row=1, column=1)

        Label(pos_frame, text="I:").grid(row=2, column=0)
        self.pos_i_scale = Scale(pos_frame, from_=0, to=0.1, resolution=0.001, orient=tk.HORIZONTAL, length=150)
        self.pos_i_scale.set(self.position_pid.ki)
        self.pos_i_scale.grid(row=2, column=1)

        Label(pos_frame, text="D:").grid(row=3, column=0)
        self.pos_d_scale = Scale(pos_frame, from_=0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=150)
        self.pos_d_scale.set(self.position_pid.kd)
        self.pos_d_scale.grid(row=3, column=1)

        # Pendulum PID Tuning
        pend_frame = Frame(tuning_frame)
        pend_frame.grid(row=0, column=1, padx=10, pady=5)
        Label(pend_frame, text="Pendulum PID", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2)

        Label(pend_frame, text="P:").grid(row=1, column=0)
        self.pend_p_scale = Scale(pend_frame, from_=0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.pend_p_scale.set(self.pendulum_pid.kp)
        self.pend_p_scale.grid(row=1, column=1)

        Label(pend_frame, text="I:").grid(row=2, column=0)
        self.pend_i_scale = Scale(pend_frame, from_=0, to=0.1, resolution=0.001, orient=tk.HORIZONTAL, length=150)
        self.pend_i_scale.set(self.pendulum_pid.ki)
        self.pend_i_scale.grid(row=2, column=1)

        Label(pend_frame, text="D:").grid(row=3, column=0)
        self.pend_d_scale = Scale(pend_frame, from_=0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, length=150)
        self.pend_d_scale.set(self.pendulum_pid.kd)
        self.pend_d_scale.grid(row=3, column=1)

        # Energy Control Parameters
        energy_frame = Frame(tuning_frame)
        energy_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5)
        Label(energy_frame, text="Energy Control (Swing-up)", font=("Arial", 10, "bold")).grid(row=0, column=0,
                                                                                               columnspan=3)

        Label(energy_frame, text="Energy Gain (ke):").grid(row=1, column=0)
        self.ke_scale = Scale(energy_frame, from_=0, to=100.0, resolution=1.0, orient=tk.HORIZONTAL, length=150)
        self.ke_scale.set(self.energy_controller['ke'])
        self.ke_scale.grid(row=1, column=1)

        Label(energy_frame, text="Reference Energy (Er):").grid(row=2, column=0)
        self.er_scale = Scale(energy_frame, from_=0, to=0.1, resolution=0.001, orient=tk.HORIZONTAL, length=150)
        self.er_scale.set(self.energy_controller['Er'])
        self.er_scale.grid(row=2, column=1)

        # Update PID parameters button
        self.update_pid_btn = Button(tuning_frame, text="Update Control Parameters",
                                     command=self.update_control_params, width=20)
        self.update_pid_btn.grid(row=2, column=0, columnspan=2, pady=5)

        # Stop button
        self.stop_btn = Button(control_frame, text="STOP MOTOR",
                               command=self.stop_motor,
                               width=20, height=2,
                               bg="red", fg="white")
        self.stop_btn.grid(row=4, column=0, pady=10)

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

        # RGB Control
        rgb_frame = Frame(self.master, padx=10, pady=10)
        rgb_frame.pack()

        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

    def update_control_params(self):
        """Update control parameters from GUI"""
        # Update Position PID
        self.position_pid.set_gains(
            self.pos_p_scale.get(),
            self.pos_i_scale.get(),
            self.pos_d_scale.get()
        )

        # Update Pendulum PID
        self.pendulum_pid.set_gains(
            self.pend_p_scale.get(),
            self.pend_i_scale.get(),
            self.pend_d_scale.get()
        )

        # Update Energy Controller
        self.energy_controller['ke'] = self.ke_scale.get()
        self.energy_controller['Er'] = self.er_scale.get()

        self.status_label.config(text="Control parameters updated")

    def set_manual_voltage(self, value):
        """Set manual voltage from slider"""
        self.motor_voltage = float(value)
        # Reset any automatic control modes
        self.calibrating = False
        self.moving_to_position = False
        self.pid_mode = False
        self.balance_mode = False
        self.mode_label.config(text="Manual")

    def calibrate(self):
        """Simple calibration - move to corner and set as zero"""
        self.calibrating = True
        self.moving_to_position = False
        self.pid_mode = False
        self.balance_mode = False
        self.voltage_slider.set(0)  # Reset slider

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

    def toggle_pid_control(self):
        """Toggle position PID control mode on/off"""
        if not self.pid_mode:
            # Start PID control
            self.pid_mode = True
            self.balance_mode = False
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.pid_control_btn.config(text="Stop Position PID")
            self.balance_btn.config(text="Start Balance Control")
            self.status_label.config(text="Position PID control active")
            self.mode_label.config(text="Position PID")

            # Reset PID controller
            self.position_pid.reset()
            self.position_pid.set_setpoint(float(self.position_entry.get()))

            # Set green LED during PID control
            self.r_slider.set(0)
            self.g_slider.set(999)
            self.b_slider.set(0)
        else:
            # Stop PID control
            self.pid_mode = False
            self.motor_voltage = 0.0
            self.pid_control_btn.config(text="Start Position PID")
            self.status_label.config(text="PID control stopped")
            self.mode_label.config(text="Manual")

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def toggle_balance_control(self):
        """Toggle pendulum balance control mode on/off"""
        if not self.balance_mode:
            # Start balance control
            self.balance_mode = True
            self.pid_mode = False
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.balance_btn.config(text="Stop Balance Control")
            self.pid_control_btn.config(text="Start Position PID")
            self.status_label.config(text="Balance control active - Push pendulum up")
            self.mode_label.config(text="Balance Control")

            # Reset PID controllers
            self.pendulum_pid.reset()
            self.pendulum_pid.set_setpoint(0.0)  # Target upright position
            self.position_pid.reset()
            self.position_pid.set_setpoint(float(self.position_entry.get()))  # Target motor position

            # Set purple LED during balance control
            self.r_slider.set(500)
            self.g_slider.set(0)
            self.b_slider.set(999)
        else:
            # Stop balance control
            self.balance_mode = False
            self.motor_voltage = 0.0
            self.balance_btn.config(text="Start Balance Control")
            self.status_label.config(text="Balance control stopped")
            self.mode_label.config(text="Manual")

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def start_move_to_position(self):
        """Start moving to target position"""
        if not self.calibrating and not self.balance_mode:
            try:
                # Get target position from entry field
                self.target_position = float(self.position_entry.get())

                self.moving_to_position = True
                self.pid_mode = False
                self.balance_mode = False
                self.voltage_slider.set(0)  # Reset slider
                self.status_label.config(text=f"Moving to {self.target_position:.1f}°...")
                self.mode_label.config(text="Moving to Position")

                # Reset PID controller
                self.position_pid.reset()
                self.position_pid.set_setpoint(self.target_position)

                # Set green LED during movement
                self.r_slider.set(0)
                self.g_slider.set(999)
                self.b_slider.set(0)
            except ValueError:
                self.status_label.config(text="Invalid position value")

    def update_position_control(self):
        """Update position control using PID"""
        # Get current angle
        current_angle = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero

        # Calculate motor velocity
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0:
            motor_velocity = (current_angle - self.prev_motor_angle) / dt
        else:
            motor_velocity = 0.0
        self.prev_motor_angle = current_angle
        self.prev_time = current_time

        # Update PID controller
        self.motor_voltage = self.position_pid.compute(current_angle, dt)

        # Check if we're close enough
        position_error = abs(self.target_position - current_angle)
        velocity_low = abs(motor_velocity) < 5.0  # Low velocity threshold

        if position_error < 0.5 and velocity_low:
            # We've reached the target
            self.moving_to_position = False
            self.pid_mode = False  # Also disable continuous PID mode
            self.motor_voltage = 0.0

            # Set blue LED when done
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

            self.status_label.config(text="Position reached")
            self.mode_label.config(text="Manual")
            return

    def update_pid_control(self):
        """Update position PID control"""
        # Similar to position control but runs continuously
        current_angle = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero

        # Calculate motor velocity
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt > 0:
            motor_velocity = (current_angle - self.prev_motor_angle) / dt
        else:
            motor_velocity = 0.0
        self.prev_motor_angle = current_angle
        self.prev_time = current_time

        # Update current setpoint from entry field
        try:
            target = float(self.position_entry.get())
            if target != self.position_pid.setpoint:
                self.position_pid.set_setpoint(target)
        except ValueError:
            pass  # Ignore invalid entries

        # Compute PID output
        self.motor_voltage = self.position_pid.compute(current_angle, dt)

        # Update status
        self.status_label.config(
            text=f"Position PID control: Target={self.position_pid.setpoint:.1f}°, Current={current_angle:.1f}°"
        )

    def update_balance_control(self):
        """Update balance control - includes swing-up and balance"""
        # Get current angles
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # For pendulum angle, flip convention:
        # QUBE: 0 degrees = down position
        # We want: 0 degrees = upright position, 180/-180 = down position
        # Following the C++ code from the second file
        if pendulum_angle_deg > 0:
            pendulum_angle_deg -= 180
        else:
            pendulum_angle_deg += 180

        # Convert to radians for calculations
        motor_angle_rad = np.radians(motor_angle_deg)
        pendulum_angle_rad = np.radians(pendulum_angle_deg)

        # Calculate velocities
        current_time = time.time()
        dt = current_time - self.prev_time

        if dt > 0:
            pendulum_velocity = (pendulum_angle_rad - self.prev_pendulum_angle) / dt
            motor_velocity = (motor_angle_rad - self.prev_motor_angle) / dt
        else:
            pendulum_velocity = 0.0
            motor_velocity = 0.0

        self.prev_pendulum_angle = pendulum_angle_rad
        self.prev_motor_angle = motor_angle_rad
        self.prev_time = current_time

        # Check if pendulum is within balance range
        within_balance_range = abs(pendulum_angle_deg) < self.balance_range

        if within_balance_range:
            # Balance control mode
            # Reset PID controllers if we just entered balance mode
            if not hasattr(self, 'was_in_balance_range') or not self.was_in_balance_range:
                self.pendulum_pid.reset()
                self.position_pid.reset()
                self.position_pid.set_setpoint(float(self.position_entry.get()))

            # Compute pendulum PID output
            self.pendulum_pid.set_setpoint(0.0)  # Target is upright (0 degrees)
            pendulum_control = self.pendulum_pid.compute(pendulum_angle_deg, dt)

            # Compute position PID output
            position_control = self.position_pid.compute(motor_angle_deg, dt)

            # Combine outputs
            self.motor_voltage = pendulum_control + position_control

            # Limit voltage for safety
            self.motor_voltage = max(-self.max_voltage, min(self.max_voltage, self.motor_voltage))

            self.status_label.config(text=f"Balancing - {abs(pendulum_angle_deg):.1f}° from upright")

        else:
            # Swing-up control mode using energy based approach from C++ file
            # Calculate energy
            E = 0.5 * self.J * pendulum_velocity ** 2 + self.m_p * self.g * self.l_com * (
                        1 - np.cos(pendulum_angle_rad))

            # Energy control law
            u = self.energy_controller['ke'] * (E - self.energy_controller['Er']) * (
                        -pendulum_velocity * np.cos(pendulum_angle_rad))

            # Limit control for safety
            u_sat = max(-self.energy_controller['u_max'], min(self.energy_controller['u_max'], u))

            # Convert to voltage - formula from C++ code
            self.motor_voltage = u_sat * (8.4 * 0.095 * 0.085) / 0.042

            # Limit voltage for safety
            self.motor_voltage = max(-self.max_voltage, min(self.max_voltage, self.motor_voltage))

            self.status_label.config(text=f"Swinging up - {abs(pendulum_angle_deg):.1f}° from upright")

        # Store balance range state for next iteration
        self.was_in_balance_range = within_balance_range

    def stop_motor(self):
        """Stop the motor"""
        self.calibrating = False
        self.moving_to_position = False
        self.pid_mode = False
        self.balance_mode = False
        self.motor_voltage = 0.0
        self.voltage_slider.set(0)

        # Set blue LED when stopped
        self.r_slider.set(0)
        self.g_slider.set(0)
        self.b_slider.set(999)

        self.status_label.config(text="Motor stopped")
        self.mode_label.config(text="Manual")

    def update_gui(self):
        """Update the GUI and control the hardware - called continuously"""
        # Update automatic control modes if active
        if self.calibrating:
            self.update_calibration()
        elif self.moving_to_position:
            self.update_position_control()
        elif self.pid_mode:
            self.update_pid_control()
        elif self.balance_mode:
            self.update_balance_control()

        # Apply the current motor voltage - THIS IS CRITICAL TO DO ON EVERY LOOP!
        self.qube.setMotorVoltage(self.motor_voltage)

        # Apply RGB values
        self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())

        # Update display information
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Get normalized pendulum angle for display (relative to upright position)
        if pendulum_angle_deg > 0:
            norm_pendulum_angle = pendulum_angle_deg - 180
        else:
            norm_pendulum_angle = pendulum_angle_deg + 180

        rpm = self.qube.getMotorRPM()

        self.angle_label.config(text=f"{motor_angle_deg:.1f}°")
        self.pendulum_label.config(
            text=f"{pendulum_angle_deg:.1f}° ({abs(norm_pendulum_angle):.1f}° from upright)")
        self.rpm_label.config(text=f"{rpm:.1f}")
        self.voltage_label.config(text=f"{self.motor_voltage:.1f} V")


def main():
    print("Starting QUBE Controller with PID...")
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
        app = QUBEControllerWithPID(root, qube)

        # Main loop
        while True:
            qube.update()
            app.update_gui()
            root.update()
            time.sleep(0.01)  # 100Hz update rate

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