import tkinter as tk
from tkinter import Button, Label, Frame, Scale, Entry, filedialog
from QUBE import QUBE
import time
import numpy as np
import torch
import torch.nn as nn
import os

# Update with your COM port
COM_PORT = "COM10"


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
    main()