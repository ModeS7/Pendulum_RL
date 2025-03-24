import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry
import time
import numpy as np
import os
import threading
import serial
import struct

# Update with your COM port
COM_PORT = "COM10"


class QUBE:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.connected = False

        # Data storage
        self.motor_angle = 0.0
        self.pendulum_angle = 0.0
        self.motor_rpm = 0.0
        self.motor_current = 0.0

        # Communication thread
        self.comm_thread = None
        self.running = False

        # Locks for thread safety
        self.data_lock = threading.Lock()
        self.command_lock = threading.Lock()

        # Command values
        self.reset_motor = False
        self.reset_pendulum = False
        self.r = 0
        self.g = 0
        self.b = 999  # Default blue
        self.motor_voltage = 0.0

        self.connect()

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.connected = True
            self.running = True
            self.comm_thread = threading.Thread(target=self._communication_loop)
            self.comm_thread.daemon = True
            self.comm_thread.start()
            print(f"Connected to QUBE on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect: {str(e)}")
            self.connected = False
            return False

    def _communication_loop(self):
        """Background thread for serial communication"""
        last_send_time = 0
        send_interval = 0.01  # 10ms (100Hz) command rate

        while self.running:
            current_time = time.time()

            # Send commands at controlled rate
            if current_time - last_send_time >= send_interval:
                self._send_commands()
                last_send_time = current_time

            # Always check for incoming data (non-blocking)
            self._receive_data()

            # Small sleep to prevent CPU hogging
            time.sleep(0.001)  # 1ms

    def _send_commands(self):
        if not self.connected:
            return

        try:
            with self.command_lock:
                # Pack command data
                reset_motor_byte = 1 if self.reset_motor else 0
                reset_pendulum_byte = 1 if self.reset_pendulum else 0

                # Convert voltage to int format (-999 to 999)
                voltage_int = int(self.motor_voltage * 100) + 999

                # Ensure in valid range
                voltage_int = max(0, min(1998, voltage_int))

                # Format command packet (10 bytes)
                command = bytearray([
                    reset_motor_byte,
                    reset_pendulum_byte,
                    (self.r >> 8) & 0xFF,
                    self.r & 0xFF,
                    (self.g >> 8) & 0xFF,
                    self.g & 0xFF,
                    (self.b >> 8) & 0xFF,
                    self.b & 0xFF,
                    (voltage_int >> 8) & 0xFF,
                    voltage_int & 0xFF
                ])

                self.ser.write(command)

                # Reset flags after sending
                self.reset_motor = False
                self.reset_pendulum = False

        except Exception as e:
            print(f"Send error: {str(e)}")
            self.connected = False

    def _receive_data(self):
        if not self.connected:
            return

        try:
            # Check if we have a complete data packet (12 bytes)
            if self.ser.in_waiting >= 12:
                data = self.ser.read(12)

                # Parse motor encoder data
                motor_rev_raw = (data[0] << 8) | data[1]
                motor_angle_raw = (data[2] << 8) | data[3]

                # Parse pendulum encoder data
                pendulum_rev_raw = (data[4] << 8) | data[5]
                pendulum_angle_raw = (data[6] << 8) | data[7]

                # Parse RPM data
                rpm_raw = (data[8] << 8) | data[9]

                # Parse current data
                current_raw = (data[10] << 8) | data[11]

                with self.data_lock:
                    # Process motor angle
                    motor_rev = motor_rev_raw & 0x7FFF  # Mask out sign bit
                    motor_sign = -1 if (motor_rev_raw & 0x8000) else 1  # Extract sign
                    motor_int = (motor_angle_raw >> 7) & 0x1FF  # Integer part (9 bits)
                    motor_dec = (motor_angle_raw & 0x7F) / 100.0  # Decimal part (7 bits)

                    # For negative angles, only the revolutions have sign, angle always positive
                    if motor_sign < 0 and motor_rev == 0:
                        # Special case for small negative angles (between 0 and -360)
                        self.motor_angle = -1 * (motor_int + motor_dec)
                    else:
                        # Normal case
                        self.motor_angle = motor_sign * (motor_rev * 360.0 + motor_int + motor_dec)

                    # Process pendulum angle (same logic as motor angle)
                    pendulum_rev = pendulum_rev_raw & 0x7FFF
                    pendulum_sign = -1 if (pendulum_rev_raw & 0x8000) else 1
                    pendulum_int = (pendulum_angle_raw >> 7) & 0x1FF
                    pendulum_dec = (pendulum_angle_raw & 0x7F) / 100.0

                    if pendulum_sign < 0 and pendulum_rev == 0:
                        self.pendulum_angle = -1 * (pendulum_int + pendulum_dec)
                    else:
                        self.pendulum_angle = pendulum_sign * (pendulum_rev * 360.0 + pendulum_int + pendulum_dec)

                    # Process RPM
                    rpm_value = rpm_raw & 0x7FFF
                    rpm_sign = -1 if (rpm_raw & 0x8000) else 1
                    self.motor_rpm = rpm_sign * rpm_value

                    # Process current
                    self.motor_current = current_raw

        except Exception as e:
            print(f"Receive error: {str(e)}")
            self.connected = False

    def getMotorAngle(self):
        with self.data_lock:
            return self.motor_angle

    def getPendulumAngle(self):
        with self.data_lock:
            return self.pendulum_angle

    def getMotorRPM(self):
        with self.data_lock:
            return self.motor_rpm

    def getMotorCurrent(self):
        with self.data_lock:
            return self.motor_current

    def setMotorVoltage(self, voltage):
        with self.command_lock:
            self.motor_voltage = voltage

    def setRGB(self, r, g, b):
        with self.command_lock:
            self.r = r
            self.g = g
            self.b = b

    def resetMotorEncoder(self):
        with self.command_lock:
            self.reset_motor = True

    def resetPendulumEncoder(self):
        with self.command_lock:
            self.reset_pendulum = True

    def update(self):
        # No longer needed - communication is handled in background thread
        pass

    def close(self):
        self.running = False
        if self.comm_thread and self.comm_thread.is_alive():
            self.comm_thread.join(timeout=1.0)
        if self.ser and self.connected:
            try:
                self.ser.close()
            except:
                pass


# Simplified parameters for real system
class SystemParameters:
    def __init__(self):
        # Control parameters
        self.max_voltage = 8.0  # Maximum voltage (V)
        self.theta_min = -2.0  # Minimum motor angle (rad)
        self.theta_max = 2.0  # Maximum motor angle (rad)
        self.balance_range = np.radians(20)  # Range where balance control activates (rad)

        # Energy control parameters
        self.m_p = 0.024  # Pendulum mass (kg)
        self.l = 0.129  # Pendulum length (m)
        self.l_com = self.l / 2  # Center of mass distance (m)
        self.J = (1 / 3) * self.m_p * self.l ** 2  # Moment of inertia (kg·m²)
        self.g = 9.81  # Gravity (m/s²)
        self.ke = 50.0  # Energy controller gain
        self.Er = 0.015  # Reference energy (J)

        # LQR control parameters
        self.lqr_theta_gain = 5.0  # Gain for motor angle
        self.lqr_alpha_gain = 50.0  # Gain for pendulum angle
        self.lqr_theta_dot_gain = 1.5  # Gain for motor velocity
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

        # Track sign of LQR parameters
        self.lqr_theta_sign = 1
        self.lqr_alpha_sign = 1
        self.lqr_theta_dot_sign = 1
        self.lqr_alpha_dot_sign = 1

        # Create GUI elements
        self.create_gui()

        # Tracking variables for calculations
        self.prev_time = time.time()
        self.prev_pendulum_angle = 0.0
        self.prev_motor_angle = 0.0

        # Variables for high-frequency control loop
        self.control_thread = None
        self.running = True
        self.control_interval = 0.001  # 1ms for controller (1000Hz)
        self.gui_interval = 0.02  # 20ms for GUI update (50Hz)
        self.last_gui_update = time.time()

        # Start control thread
        self.start_control_thread()

    def create_gui(self):
        # Main control frame
        control_frame = Frame(self.master, padx=10, pady=10)
        control_frame.pack()

        # ... [GUI code remains the same as before] ...
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
        Label(lqr_frame, text="LQR Parameters", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3)

        # Row 1: Theta Gain
        Label(lqr_frame, text="Theta Gain:").grid(row=1, column=0, padx=5)
        self.lqr_theta_scale = Scale(lqr_frame, from_=0, to=20.0,
                                     resolution=0.5, orient=tk.HORIZONTAL, length=150)
        self.lqr_theta_scale.set(self.params.lqr_theta_gain)
        self.lqr_theta_scale.grid(row=1, column=1, padx=5)

        # Add sign toggle button for Theta Gain
        self.lqr_theta_sign_btn = Button(lqr_frame, text="+", width=3,
                                         command=self.toggle_theta_sign)
        self.lqr_theta_sign_btn.grid(row=1, column=2, padx=5)

        # Row 2: Alpha Gain
        Label(lqr_frame, text="Alpha Gain:").grid(row=2, column=0, padx=5)
        self.lqr_alpha_scale = Scale(lqr_frame, from_=0, to=100.0,
                                     resolution=1.0, orient=tk.HORIZONTAL, length=150)
        self.lqr_alpha_scale.set(self.params.lqr_alpha_gain)
        self.lqr_alpha_scale.grid(row=2, column=1, padx=5)

        # Add sign toggle button for Alpha Gain
        self.lqr_alpha_sign_btn = Button(lqr_frame, text="+", width=3,
                                         command=self.toggle_alpha_sign)
        self.lqr_alpha_sign_btn.grid(row=2, column=2, padx=5)

        # Row 3: Theta Dot Gain
        Label(lqr_frame, text="Theta Dot Gain:").grid(row=3, column=0, padx=5)
        self.lqr_theta_dot_scale = Scale(lqr_frame, from_=0, to=5.0,
                                         resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.lqr_theta_dot_scale.set(self.params.lqr_theta_dot_gain)
        self.lqr_theta_dot_scale.grid(row=3, column=1, padx=5)

        # Add sign toggle button for Theta Dot Gain
        self.lqr_theta_dot_sign_btn = Button(lqr_frame, text="+", width=3,
                                             command=self.toggle_theta_dot_sign)
        self.lqr_theta_dot_sign_btn.grid(row=3, column=2, padx=5)

        # Row 4: Alpha Dot Gain
        Label(lqr_frame, text="Alpha Dot Gain:").grid(row=4, column=0, padx=5)
        self.lqr_alpha_dot_scale = Scale(lqr_frame, from_=0, to=20.0,
                                         resolution=0.5, orient=tk.HORIZONTAL, length=150)
        self.lqr_alpha_dot_scale.set(self.params.lqr_alpha_dot_gain)
        self.lqr_alpha_dot_scale.grid(row=4, column=1, padx=5)

        # Add sign toggle button for Alpha Dot Gain
        self.lqr_alpha_dot_sign_btn = Button(lqr_frame, text="+", width=3,
                                             command=self.toggle_alpha_dot_sign)
        self.lqr_alpha_dot_sign_btn.grid(row=4, column=2, padx=5)

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

        # Performance display
        Label(status_frame, text="Control Loop Time:").grid(row=6, column=0, sticky=tk.W)
        self.dt_label = Label(status_frame, text="0.0 ms")
        self.dt_label.grid(row=6, column=1, sticky=tk.W)

        # RGB Control
        rgb_frame = Frame(self.master, padx=10, pady=10)
        rgb_frame.pack()

        self.r_slider = Scale(rgb_frame, from_=999, to=0, label="Red")
        self.r_slider.grid(row=0, column=0, padx=5)

        self.g_slider = Scale(rgb_frame, from_=999, to=0, label="Green")
        self.g_slider.grid(row=0, column=1, padx=5)

        self.b_slider = Scale(rgb_frame, from_=999, to=0, label="Blue")
        self.b_slider.grid(row=0, column=2, padx=5)

    def start_control_thread(self):
        """Start the high-frequency control thread"""
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def control_loop(self):
        """High-frequency control loop running in separate thread"""
        last_control_time = time.time()
        control_times = []  # For calculating average dt

        while self.running:
            try:
                # Calculate dt
                current_time = time.time()
                dt = current_time - last_control_time
                last_control_time = current_time

                # Store dt for averaging (but ignore outliers)
                if dt < 0.1:  # Ignore huge delays (e.g., from system sleep)
                    control_times.append(dt)
                    if len(control_times) > 100:
                        control_times.pop(0)

                # Call the control update function
                self.update_control()

                # Update dt display and GUI occasionally
                if current_time - self.last_gui_update >= self.gui_interval:
                    avg_dt = sum(control_times) / len(control_times) if control_times else 0
                    freq = 1.0 / avg_dt if avg_dt > 0 else 0
                    self.master.after(0, lambda: self.dt_label.config(text=f"{avg_dt * 1000:.2f} ms ({freq:.1f} Hz)"))
                    self.update_gui_display()
                    self.last_gui_update = current_time

                # Calculate sleep time to maintain target frequency
                sleep_time = self.control_interval - (time.time() - current_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                print(f"Error in control loop: {e}")
                # Short sleep to prevent CPU spin if there's a continuous error
                time.sleep(0.01)

    def update_control(self):
        """Update control algorithms - runs at high frequency"""
        # Execute appropriate control mode
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

        # Apply motor voltage (automatically sends to Arduino via thread)
        self.qube.setMotorVoltage(self.motor_voltage)

        # RGB control
        self.qube.setRGB(self.r_slider.get(), self.g_slider.get(), self.b_slider.get())

    def update_gui_display(self):
        """Update GUI display elements - runs at lower frequency"""
        # Get current sensor values
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjust for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Get pendulum angle relative to upright position
        pendulum_angle_rad = np.radians(pendulum_angle_deg)
        pendulum_norm = self.normalize_angle(pendulum_angle_rad + np.pi)
        upright_angle = abs(pendulum_norm) * 180 / np.pi

        rpm = self.qube.getMotorRPM()

        # Update display labels
        self.angle_label.config(text=f"{motor_angle_deg:.1f}°")
        self.pendulum_label.config(
            text=f"{pendulum_angle_deg:.1f}° ({upright_angle:.1f}° from upright)")
        self.rpm_label.config(text=f"{rpm:.1f}")
        self.voltage_label.config(text=f"{self.motor_voltage:.1f} V")

    # Toggle sign methods for LQR parameters
    def toggle_theta_sign(self):
        self.lqr_theta_sign *= -1
        self.lqr_theta_sign_btn.config(text="+" if self.lqr_theta_sign > 0 else "-")
        self.status_label.config(text=f"Theta Gain sign changed to {'+' if self.lqr_theta_sign > 0 else '-'}")

    def toggle_alpha_sign(self):
        self.lqr_alpha_sign *= -1
        self.lqr_alpha_sign_btn.config(text="+" if self.lqr_alpha_sign > 0 else "-")
        self.status_label.config(text=f"Alpha Gain sign changed to {'+' if self.lqr_alpha_sign > 0 else '-'}")

    def toggle_theta_dot_sign(self):
        self.lqr_theta_dot_sign *= -1
        self.lqr_theta_dot_sign_btn.config(text="+" if self.lqr_theta_dot_sign > 0 else "-")
        self.status_label.config(text=f"Theta Dot Gain sign changed to {'+' if self.lqr_theta_dot_sign > 0 else '-'}")

    def toggle_alpha_dot_sign(self):
        self.lqr_alpha_dot_sign *= -1
        self.lqr_alpha_dot_sign_btn.config(text="+" if self.lqr_alpha_dot_sign > 0 else "-")
        self.status_label.config(text=f"Alpha Dot Gain sign changed to {'+' if self.lqr_alpha_dot_sign > 0 else '-'}")

    def update_parameters(self):
        """Update control parameters from GUI"""
        self.params.max_voltage = self.bb_voltage_scale.get()
        self.params.balance_range = np.radians(self.balance_range_scale.get())
        self.params.ke = self.ke_scale.get()
        self.params.Er = self.er_scale.get()

        # Update LQR parameters (with signs)
        self.params.lqr_theta_gain = self.lqr_theta_sign * self.lqr_theta_scale.get()
        self.params.lqr_alpha_gain = self.lqr_alpha_sign * self.lqr_alpha_scale.get()
        self.params.lqr_theta_dot_gain = self.lqr_theta_dot_sign * self.lqr_theta_dot_scale.get()
        self.params.lqr_alpha_dot_gain = self.lqr_alpha_dot_sign * self.lqr_alpha_dot_scale.get()

        self.status_label.config(text="Control parameters updated")

    def set_manual_voltage(self, value):
        """Set manual voltage from slider"""
        self.motor_voltage = float(value)
        # Reset any automatic control modes
        self.calibrating = False
        self.moving_to_position = False
        self.bang_bang_mode = False
        self.energy_mode = False
        self.lqr_mode = False
        self.mode_label.config(text="Manual")

    def calibrate(self):
        """Simple calibration - move to corner and set as zero"""
        self.calibrating = True
        self.moving_to_position = False
        self.bang_bang_mode = False
        self.energy_mode = False
        self.lqr_mode = False
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

            # Update status - use after() to safely update GUI from thread
            self.master.after(0, lambda: self.status_label.config(text="Calibration complete - Corner is now zero"))
            self.master.after(0, lambda: self.mode_label.config(text="Manual"))

    # Other control methods (toggle_bang_bang, toggle_energy_control, etc.) remain mostly the same,
    # but use self.master.after(0, lambda: ...) for GUI updates from the control thread

    def toggle_bang_bang(self):
        """Toggle bang-bang swing-up control mode"""
        if not self.bang_bang_mode:
            # Start bang-bang control
            self.bang_bang_mode = True
            self.energy_mode = False
            self.lqr_mode = False
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.bang_bang_btn.config(text="Stop Bang-Bang")
            self.energy_btn.config(text="Energy Swing-Up")
            self.lqr_btn.config(text="LQR Balance")
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
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.energy_btn.config(text="Stop Energy Control")
            self.bang_bang_btn.config(text="Bang-Bang Swing-Up")
            self.lqr_btn.config(text="LQR Balance")
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
            self.moving_to_position = False
            self.calibrating = False
            self.voltage_slider.set(0)  # Reset slider
            self.lqr_btn.config(text="Stop LQR Control")
            self.bang_bang_btn.config(text="Bang-Bang Swing-Up")
            self.energy_btn.config(text="Energy Swing-Up")
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

            # Set blue LED when stopped
            self.r_slider.set(0)
            self.g_slider.set(0)
            self.b_slider.set(999)

    def start_move_to_position(self):
        """Start moving to target position"""
        if not self.calibrating and not self.bang_bang_mode and not self.energy_mode and not self.lqr_mode:
            try:
                # Get target position from entry field
                self.target_position = float(self.position_entry.get())

                self.moving_to_position = True
                self.bang_bang_mode = False
                self.energy_mode = False
                self.lqr_mode = False
                self.voltage_slider.set(0)  # Reset slider
                self.status_label.config(text=f"Moving to {self.target_position:.1f}°...")
                self.mode_label.config(text="Moving to Position")

                # Set green LED during movement
                self.r_slider.set(0)
                self.g_slider.set(999)
                self.b_slider.set(0)
            except ValueError:
                self.status_label.config(text="Invalid position value")

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

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

            # Update GUI (thread-safe)
            self.master.after(0, lambda: self.status_label.config(text="Position reached"))
            self.master.after(0, lambda: self.mode_label.config(text="Manual"))

    def update_bang_bang_control(self):
        """Update bang-bang control for pendulum swing-up - simplified for real system"""
        # Get current angles
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert to radians
        motor_angle = np.radians(motor_angle_deg)
        pendulum_angle = np.radians(pendulum_angle_deg)

        # Initialize prev_pendulum_angle if this is the first update
        if not hasattr(self, 'prev_pendulum_angle') or self.prev_pendulum_angle is None:
            self.prev_pendulum_angle = pendulum_angle

        # Calculate pendulum angular velocity by simple finite difference
        current_time = time.time()
        dt = current_time - self.prev_time
        pendulum_velocity = 0.0

        if dt > 0:
            pendulum_velocity = (pendulum_angle - self.prev_pendulum_angle) / dt

        self.prev_pendulum_angle = pendulum_angle
        self.prev_time = current_time

        # Normalize pendulum angle to make upright = 0
        pendulum_norm = self.normalize_angle(pendulum_angle + np.pi)

        # Check if we've reached the upright position
        if abs(pendulum_norm) < self.params.balance_range:
            # Switch to LQR for balancing if LQR mode is enabled
            if not self.lqr_mode:
                # Thread-safe toggle via after()
                self.master.after(0, self.toggle_lqr)
                self.master.after(0, lambda: self.status_label.config(text="Switched to LQR for balance control"))
                return

        # Simple Bang-Bang control logic
        if abs(pendulum_norm) < self.params.balance_range:
            # We're close to upright position
            if abs(pendulum_velocity) < 0.1:
                # Very little motion, don't apply voltage
                self.motor_voltage = 0.0
            else:
                # Apply opposite force to slow down
                self.motor_voltage = -np.sign(pendulum_velocity) * self.params.max_voltage

            # Thread-safe status update
            self.master.after(0, lambda: self.status_label.config(
                text=f"Near balance - {abs(pendulum_norm) * 180 / np.pi:.1f}° from upright"))
        else:
            # Motor angle limit protection
            if motor_angle > self.params.theta_max - 0.6 and pendulum_velocity > 0:
                self.motor_voltage = -self.params.max_voltage  # If close to upper limit, push back
            elif motor_angle < self.params.theta_min + 0.6 and pendulum_velocity < 0:
                self.motor_voltage = self.params.max_voltage  # If close to lower limit, push back
            else:
                # Simple bang-bang control: apply full voltage based on pendulum velocity direction
                if pendulum_velocity == 0:
                    # Minimal motion detected, apply small voltage to get it moving
                    self.motor_voltage = 0.5 * self.params.max_voltage
                else:
                    # Full bang-bang: apply max voltage in direction of motion
                    self.motor_voltage = np.sign(pendulum_velocity) * self.params.max_voltage

            # Thread-safe status update
            self.master.after(0, lambda: self.status_label.config(
                text=f"Swinging - {abs(pendulum_norm) * 180 / np.pi:.1f}° from upright"))

    # Other control methods remain similar but with thread-safe GUI updates

    def update_energy_control(self):
        """Update energy-based swing-up control for the pendulum"""
        # Get current angles
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert to radians
        motor_angle = np.radians(motor_angle_deg)
        pendulum_angle = np.radians(pendulum_angle_deg)

        # Initialize prev_pendulum_angle if this is the first update
        if not hasattr(self, 'prev_pendulum_angle') or self.prev_pendulum_angle is None:
            self.prev_pendulum_angle = pendulum_angle

        # Calculate pendulum velocity via finite difference
        current_time = time.time()
        dt = current_time - self.prev_time

        pendulum_velocity = 0.0
        if dt > 0:
            pendulum_velocity = (pendulum_angle - self.prev_pendulum_angle) / dt

        self.prev_pendulum_angle = pendulum_angle
        self.prev_time = current_time

        # Adjust pendulum angle to make upright = 0
        # Note: for energy calculations, we want down position = 0, upright = π
        pendulum_norm = self.normalize_angle(pendulum_angle + np.pi)

        # Check if we've reached the upright position
        if abs(pendulum_norm) < self.params.balance_range:
            # Switch to LQR for balancing if LQR mode is enabled
            if not self.lqr_mode:
                # Thread-safe toggle via after()
                self.master.after(0, self.toggle_lqr)
                self.master.after(0, lambda: self.status_label.config(text="Switched to LQR for balance control"))
                return

        # Calculate current energy of the pendulum
        # E = Kinetic Energy + Potential Energy
        # E = 0.5 * J * ω² + m*g*l*(1-cos(θ))
        # Where θ is measured from downward (θ=0 at bottom, θ=π at top)
        E = 0.5 * self.params.J * pendulum_velocity ** 2 + \
            self.params.m_p * self.params.g * self.params.l_com * (1 - np.cos(pendulum_angle))

        # Reference energy is the energy at upright position with zero velocity
        E_ref = self.params.Er  # Small non-zero value helps with convergence

        # Energy error
        E_error = E - E_ref

        # Check if we're within balance range
        if abs(pendulum_norm) < self.params.balance_range:
            # Near upright position - apply damping control
            if abs(pendulum_velocity) < 0.1:
                # Very little motion, don't apply voltage
                self.motor_voltage = 0.0
            else:
                # Apply opposite force to slow down
                self.motor_voltage = -np.sign(pendulum_velocity) * 2.0

            self.master.after(0, lambda: self.status_label.config(
                text=f"Near balance - {abs(pendulum_norm) * 180 / np.pi:.1f}° from upright"))
        else:
            # Check motor angle limits first
            if motor_angle > self.params.theta_max - 0.3:
                # Near upper limit, push back
                self.motor_voltage = -self.params.max_voltage
                self.master.after(0, lambda: self.status_label.config(text=f"Upper limit reached - pushing back"))
            elif motor_angle < self.params.theta_min + 0.3:
                # Near lower limit, push back
                self.motor_voltage = self.params.max_voltage
                self.master.after(0, lambda: self.status_label.config(text=f"Lower limit reached - pushing back"))
            else:
                # Apply energy control law:
                # u = k_e * (E - E_ref) * sign(cos(θ) * dθ/dt)
                # Where sign(cos(θ) * dθ/dt) determines if pendulum is moving toward or away from upright
                energy_direction = np.cos(pendulum_angle) * pendulum_velocity
                direction = -1.0 if energy_direction > 0 else 1.0

                # Apply the control law
                u = self.params.ke * E_error * direction

                # Limit voltage
                self.motor_voltage = max(-self.params.max_voltage, min(self.params.max_voltage, u))

                self.master.after(0, lambda: self.status_label.config(
                    text=f"Energy control - E={E:.3f}, E_ref={E_ref:.3f}, E_error={E_error:.3f}"))

        # Clamp motor voltage to system limits
        self.motor_voltage = max(-self.params.max_voltage, min(self.params.max_voltage, self.motor_voltage))

    def update_lqr_control(self):
        """LQR balance controller for pendulum"""
        # Get current angles
        motor_angle_deg = self.qube.getMotorAngle() + 136.0  # Adjusted for corner as zero
        pendulum_angle_deg = self.qube.getPendulumAngle()

        # Convert to radians
        motor_angle = np.radians(motor_angle_deg)
        pendulum_angle = np.radians(pendulum_angle_deg)

        # Calculate velocities using finite difference
        current_time = time.time()
        dt = current_time - self.prev_time

        pendulum_velocity = 0.0
        motor_velocity = 0.0

        if dt > 0:
            # Make sure we have previous values stored
            if not hasattr(self, 'prev_pendulum_angle') or self.prev_pendulum_angle is None:
                self.prev_pendulum_angle = pendulum_angle

            if not hasattr(self, 'prev_motor_angle') or self.prev_motor_angle is None:
                self.prev_motor_angle = motor_angle

            # Calculate velocities
            pendulum_velocity = (pendulum_angle - self.prev_pendulum_angle) / dt
            motor_velocity = (motor_angle - self.prev_motor_angle) / dt

        # Update previous values
        self.prev_pendulum_angle = pendulum_angle
        self.prev_motor_angle = motor_angle
        self.prev_time = current_time

        # Normalize pendulum angle to make upright = 0
        # For LQR, we want pendulum_norm to be 0 at the upright position
        pendulum_norm = self.normalize_angle(pendulum_angle + np.pi)

        # Check if we're still close enough to balance
        if abs(pendulum_norm) > 2 * self.params.balance_range:
            # Too far from upright, switch back to swing-up
            self.master.after(0, self.toggle_energy_control)  # Switch to energy-based swing-up
            self.master.after(0, lambda: self.status_label.config(
                text="Pendulum too far from upright, switching to swing-up"))
            return

        # LQR control
        # u = -K*x where x = [theta, alpha, theta_dot, alpha_dot]
        # Using the signs from the toggle buttons
        u = -(
                self.params.lqr_theta_gain * motor_angle +
                self.params.lqr_alpha_gain * pendulum_norm +
                self.params.lqr_theta_dot_gain * motor_velocity +
                self.params.lqr_alpha_dot_gain * pendulum_velocity
        )

        # Add limit avoidance term
        limit_margin = 0.3  # radians
        if motor_angle > self.params.theta_max - limit_margin:
            # Add strong negative control to avoid upper limit
            avoid_factor = self.params.lqr_avoid_factor * (
                    motor_angle - (self.params.theta_max - limit_margin)) / limit_margin
            u -= avoid_factor
        elif motor_angle < self.params.theta_min + limit_margin:
            # Add strong positive control to avoid lower limit
            avoid_factor = self.params.lqr_avoid_factor * (
                    (self.params.theta_min + limit_margin) - motor_angle) / limit_margin
            u += avoid_factor

        # Limit voltage to system constraints
        self.motor_voltage = max(-self.params.max_voltage, min(self.params.max_voltage, u))

        # Update status information (thread-safe)
        self.master.after(0, lambda: self.status_label.config(
            text=f"LQR Balance - {abs(pendulum_norm) * 180 / np.pi:.1f}° from upright"))

    def stop_motor(self):
        """Stop the motor"""
        self.calibrating = False
        self.moving_to_position = False
        self.bang_bang_mode = False
        self.energy_mode = False
        self.lqr_mode = False
        self.motor_voltage = 0.0
        self.voltage_slider.set(0)
        self.bang_bang_btn.config(text="Bang-Bang Swing-Up")
        self.energy_btn.config(text="Energy Swing-Up")
        self.lqr_btn.config(text="LQR Balance")

        # Set blue LED when stopped
        self.r_slider.set(0)
        self.g_slider.set(0)
        self.b_slider.set(999)

        self.status_label.config(text="Motor stopped")
        self.mode_label.config(text="Manual")

    def shutdown(self):
        """Clean shutdown of threads"""
        self.running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)

        # Stop the motor
        self.motor_voltage = 0.0
        self.qube.setMotorVoltage(0.0)
        self.qube.close()


def main():
    print("Starting QUBE Controller with Swing-Up and LQR Balance Control...")
    print("Will set corner position as zero")

    try:
        # Initialize QUBE with new communication model
        qube = QUBE(COM_PORT, 115200)

        # Create GUI
        root = tk.Tk()
        app = QUBEControllerWithBangBang(root, qube)

        # Clean shutdown on window close
        root.protocol("WM_DELETE_WINDOW", lambda: [app.shutdown(), root.destroy()])

        # Main loop - just for GUI updates
        root.mainloop()

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Final stop attempt
        try:
            if 'qube' in locals():
                qube.setMotorVoltage(0.0)
                qube.close()
            print("Motor stopped")
        except:
            pass


if __name__ == "__main__":
    main()