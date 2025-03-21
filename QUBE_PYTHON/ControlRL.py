import numpy as np
import torch
import time
from QUBE import QUBE
import matplotlib.pyplot as plt
from collections import deque

# Define the COM port (from your control.py)
COM_PORT = "COM10"  # Update this to match your system

# Constants similar to training environment
THETA_MIN = -2.2  # Minimum arm angle (radians)
THETA_MAX = 2.2  # Maximum arm angle (radians)


class SACController:
    def __init__(self, model_path, state_dim=6, action_dim=1, hidden_dim=256):
        """
        Initialize the SAC controller for the real QUBE system

        Args:
            model_path (str): Path to the pretrained actor model
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden dimension of the network
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the QUBE hardware interface
        self.qube = QUBE(COM_PORT, 115200)

        # Create observation buffer to match delay in training
        self.delay_steps = 5  # Should match what was used in training
        self.observation_buffer = deque(maxlen=self.delay_steps + 1)

        # Load the pretrained model
        self.actor = self._load_actor(model_path, state_dim, action_dim, hidden_dim)

        # Record historical data for plotting
        self.history = {
            'time': [],
            'motor_angle': [],
            'pendulum_angle': [],
            'motor_voltage': [],
            'motor_rpm': []
        }
        self.start_time = time.time()

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def _load_actor(self, model_path, state_dim, action_dim, hidden_dim):
        """
        Load the pretrained actor model

        Args:
            model_path (str): Path to the saved model
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden dimension size

        Returns:
            nn.Module: Loaded actor model
        """
        print(f"Loading model from {model_path}")

        # Define Actor network (same structure as in training)
        class Actor(torch.nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim=256):
                super(Actor, self).__init__()

                self.network = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU()
                )

                # Mean and log std for continuous action
                self.mean = torch.nn.Linear(hidden_dim, action_dim)
                self.log_std = torch.nn.Linear(hidden_dim, action_dim)

            def forward(self, state):
                features = self.network(state)

                # Get mean and constrain it to [-1, 1]
                action_mean = torch.tanh(self.mean(features))

                # Get log standard deviation and clamp it
                action_log_std = self.log_std(features)
                action_log_std = torch.clamp(action_log_std, -20, 2)

                return action_mean, action_log_std

        # Create new model instance
        actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)

        # Load the state dict
        actor.load_state_dict(torch.load(model_path, map_location=self.device))

        # Set to evaluation mode
        actor.eval()

        print("Model loaded successfully!")
        return actor

    def calibrate_system(self):
        """
        Calibrate the system by applying constant voltage to find edge limit
        and then position the arm at 2.3 rad from center
        """
        print("Starting system calibration...")

        # Reset encoders
        self.qube.resetMotorEncoder()
        self.qube.resetPendulumEncoder()
        self.qube.update()
        time.sleep(0.1)

        # Set RGB to indicate calibration mode
        self.qube.setRGB(0, 999, 0)  # Green

        # Phase 1: Apply 2.5V constant for 2 seconds to find edge limit
        print("Applying 2.5V for 2 seconds to find edge limit...")
        start_time = time.time()
        max_theta = -float('inf')

        while time.time() - start_time < 3.5:
            self.qube.setMotorVoltage(1.5)
            self.qube.update()
            current_theta = np.radians(self.qube.getMotorAngle())
            max_theta = max(max_theta, current_theta)
            time.sleep(0.01)

        # Stop the motor
        self.qube.setMotorVoltage(0.0)
        self.qube.update()

        print(f"Edge limit found at approximately {max_theta:.4f} radians")

        # Set THETA_MAX based on measurement with safety margin
        effective_theta_max = max_theta * 0.95  # 5% safety margin
        print(f"Setting effective THETA_MAX to {effective_theta_max:.4f} radians")

        # Phase 2: Position the arm at 2.3 rad from center
        target_position = 2.3  # radians
        print(f"Positioning arm at {target_position} radians from center...")

        # Simple proportional control to position the arm
        max_iterations = 100
        iteration = 0
        position_error = 1.0  # Initial error

        while abs(position_error) > 0.05 and iteration < max_iterations:
            # Get current position
            current_position = np.radians(self.qube.getMotorAngle())

            # Calculate error
            position_error = target_position - current_position

            # Simple P control
            voltage = np.clip(position_error * 5.0, -10.0, 10.0)

            # Apply voltage
            self.qube.setMotorVoltage(voltage)
            self.qube.update()

            iteration += 1
            time.sleep(0.02)

        # Stop the motor at the desired position
        self.qube.setMotorVoltage(0.0)
        self.qube.update()

        final_position = np.radians(self.qube.getMotorAngle())
        print(f"Arm positioned at {final_position:.4f} radians (target: {target_position})")

        # Set RGB to indicate ready state
        self.qube.setRGB(0, 0, 999)  # Blue
        time.sleep(0.5)

        return final_position

    def get_observation(self):
        """
        Get the current observation from the QUBE system and format it
        like the training environment

        Returns:
            numpy.ndarray: Observation vector for the RL model
        """
        # Update the hardware
        self.qube.update()

        # Get motor and pendulum angles in radians
        theta = np.radians(self.qube.getMotorAngle())
        alpha = np.radians(self.qube.getPendulumAngle())

        # Get velocities (filtered to reduce noise)
        try:
            theta_dot = np.radians(self.qube.getMotorRPM()) * (2 * np.pi / 60)  # Convert RPM to rad/s
        except:
            theta_dot = 0.0  # Fallback if reading fails

        try:
            # Estimate pendulum velocity using simple differentiation (will be noisy)
            # In a real implementation, you'd want a better filter here
            if len(self.history['pendulum_angle']) > 1:
                dt = self.history['time'][-1] - self.history['time'][-2]
                if dt > 0:
                    alpha_prev = self.history['pendulum_angle'][-1]
                    alpha_dot = (alpha - alpha_prev) / dt
                    print(dt)
                else:
                    alpha_dot = 0.0
            else:
                alpha_dot = 0.0
        except:
            alpha_dot = 0.0

        # Normalize the pendulum angle for upright reference
        alpha_norm = self.normalize_angle(alpha + np.pi)

        # Format observation to match training environment
        obs = np.array([
            np.sin(theta), np.cos(theta),
            np.sin(alpha_norm), np.cos(alpha_norm),
            theta_dot / 10.0,  # Same scaling as in training
            alpha_dot / 10.0  # Same scaling as in training
        ])

        # Store current observation in the buffer
        self.observation_buffer.append(obs)

        # Add to history for plotting
        current_time = time.time() - self.start_time
        self.history['time'].append(current_time)
        self.history['motor_angle'].append(theta)
        self.history['pendulum_angle'].append(alpha_norm)

        # For the first call, we haven't applied voltage yet
        if 'last_voltage' in self.__dict__:
            self.history['motor_voltage'].append(self.last_voltage)
        else:
            self.history['motor_voltage'].append(0.0)

        try:
            self.history['motor_rpm'].append(self.qube.getMotorRPM())
        except:
            self.history['motor_rpm'].append(0.0)

        # Return delayed observation to match training environment
        if len(self.observation_buffer) <= self.delay_steps:
            return self.observation_buffer[0]
        return self.observation_buffer[-(self.delay_steps + 1)]

    def select_action(self, state):
        """
        Get action from the pretrained model

        Args:
            state (numpy.ndarray): Current state observation

        Returns:
            float: Action value (-1 to 1) for motor voltage
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action_mean, _ = self.actor(state_tensor)
            return action_mean.cpu().numpy()[0][0]  # Return scalar action

    def run_controller(self, max_duration=30.0, target_balance_duration=10.0):
        """
        Run the controller on the real system

        Args:
            max_duration (float): Maximum run time in seconds
            target_balance_duration (float): Target duration to balance the pendulum

        Returns:
            bool: True if successfully balanced for target duration
        """
        print("Starting controller. Press Ctrl+C to stop...")

        # Reset encoders and prepare system
        self.qube.resetPendulumEncoder()  # Reset only pendulum, keep motor position
        self.qube.update()

        # Clear the observation buffer
        self.observation_buffer.clear()

        # Clear history
        for key in self.history:
            self.history[key] = []

        # Reset start time
        self.start_time = time.time()

        # Main control loop
        balanced_start_time = None
        balanced_duration = 0.0
        max_voltage = 10.0  # Maximum voltage allowed

        try:
            while time.time() - self.start_time < max_duration:
                # Get observation
                state = self.get_observation()

                # Select action using the model
                action = self.select_action(state)

                # Convert normalized action (-1, 1) to voltage
                voltage = float(action) * max_voltage

                # Apply voltage to the motor
                self.qube.setMotorVoltage(voltage)
                self.last_voltage = voltage  # Save for history

                # Track balancing - pendulum is near upright when sin(alpha_norm) is near 0 and cos(alpha_norm) is near 1
                pendulum_upright = abs(state[2]) < 0.3 and state[3] > 0.95  # sin close to 0, cos close to 1

                # Update balanced duration tracking
                if pendulum_upright:
                    if balanced_start_time is None:
                        balanced_start_time = time.time()
                        self.qube.setRGB(0, 999, 0)  # Green when balancing
                    balanced_duration = time.time() - balanced_start_time
                    print(f"\rBalancing: {balanced_duration:.1f}s / {target_balance_duration:.1f}s", end="")

                    # Check if balanced for target duration
                    if balanced_duration >= target_balance_duration:
                        print(f"\nSuccess! Balanced for {balanced_duration:.1f} seconds")
                        break
                else:
                    balanced_start_time = None
                    balanced_duration = 0.0
                    self.qube.setRGB(999, 0, 0)  # Red when not balanced

                # Brief sleep for timing
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nController interrupted by user")
        finally:
            # Stop the motor
            self.qube.setMotorVoltage(0.0)
            self.qube.setRGB(0, 0, 0)
            self.qube.update()

        # Return success status
        return balanced_duration >= target_balance_duration

    def plot_results(self):
        """Plot the results of the controller run"""
        if len(self.history['time']) == 0:
            print("No data to plot")
            return

        plt.figure(figsize=(12, 10))

        # Plot motor angle
        plt.subplot(3, 1, 1)
        plt.plot(self.history['time'], self.history['motor_angle'], 'b-')
        plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.7, label='Theta Limits')
        plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.7)
        plt.ylabel('Motor angle (rad)')
        plt.title('Real System Performance')
        plt.grid(True)

        # Plot pendulum angle
        plt.subplot(3, 1, 2)
        plt.plot(self.history['time'], self.history['pendulum_angle'], 'g-')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.ylabel('Pendulum angle (rad)')
        plt.grid(True)

        # Plot motor voltage
        plt.subplot(3, 1, 3)
        plt.plot(self.history['time'], self.history['motor_voltage'], 'r-')
        plt.ylabel('Motor voltage (V)')
        plt.xlabel('Time (s)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("real_system_performance.png")
        plt.show()


def main():
    # Path to the pretrained model
    model_path = "sacpen7.pth"  # Update with your model path

    # Create the controller
    controller = SACController(model_path)

    # Calibrate the system
    initial_position = controller.calibrate_system()

    # Run the controller
    success = controller.run_controller(max_duration=60, target_balance_duration=10)

    # Plot results
    controller.plot_results()

    if success:
        print("Controller successfully balanced the pendulum!")
    else:
        print("Controller did not achieve target balance duration.")


if __name__ == "__main__":
    main()