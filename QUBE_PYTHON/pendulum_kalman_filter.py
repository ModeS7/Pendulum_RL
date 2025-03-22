import numpy as np
import torch
from scipy.linalg import block_diag

class PendulumKalmanFilter:
    def __init__(self, dt=0.01):
        """
        Initialize Kalman Filter for the pendulum system.
        
        Args:
            dt: Time step in seconds
        """
        self.dt = dt
        
        # State vector: [theta, alpha, theta_dot, alpha_dot]
        # where theta is motor angle, alpha is pendulum angle
        
        # Initialize state estimate
        self.x = np.zeros(4)
        
        # Initialize state covariance matrix
        self.P = np.eye(4)
        
        # Process noise covariance (Q) - tuned for the specific system
        # Higher values indicate less trust in the model
        self.Q = np.diag([0.001, 0.001, 0.01, 0.01])
        
        # Measurement noise covariance (R) - tuned for sensor noise characteristics
        # Higher values indicate less trust in the measurements
        self.R = np.diag([0.005, 0.005, 0.05, 0.05])
        
        # State transition matrix (will be updated each step based on simulation)
        self.F = np.eye(4)
        
        # Measurement matrix (Identity because we directly measure all states)
        self.H = np.eye(4)

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle
        
    def predict(self, simulation_model, action):
        """
        Prediction step using the simulation model
        
        Args:
            simulation_model: Function that predicts next state based on current state and action
            action: Control input to the system
        """
        # Use simulation model to predict next state
        predicted_state = simulation_model(self.x, action)
        
        # Update state prediction
        self.x = predicted_state
        
        # Update state covariance: P = F*P*F' + Q
        # For simplicity, we use a simple linearized model for covariance update
        self.F = self._approximate_jacobian(simulation_model, self.x, action)
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x

    def _approximate_jacobian(self, model, state, action, epsilon=1e-5):
        """
        Approximate the Jacobian matrix (state transition matrix) for the system dynamics
        by using finite differences.
        """
        n = len(state)
        jacobian = np.zeros((n, n))
        
        for i in range(n):
            # Create perturbed state
            state_plus = state.copy()
            state_plus[i] += epsilon
            
            # Get predictions for original and perturbed states
            prediction = model(state, action)
            prediction_plus = model(state_plus, action)
            
            # Calculate partial derivatives
            jacobian[:, i] = (prediction_plus - prediction) / epsilon
            
        return jacobian
        
    def update(self, measurement):
        """
        Update step using real system measurements
        
        Args:
            measurement: Array of measurements [theta, alpha, theta_dot, alpha_dot]
        """
        # Normalize the angle measurements
        measurement = measurement.copy()
        measurement[0] = self.normalize_angle(measurement[0])
        measurement[1] = self.normalize_angle(measurement[1])
        
        # Calculate innovation (measurement residual): y = z - H*x
        y = measurement - self.H @ self.x
        
        # Normalize angle differences to ensure proper error calculation
        y[0] = self.normalize_angle(y[0])
        y[1] = self.normalize_angle(y[1])
        
        # Calculate innovation covariance: S = H*P*H' + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Calculate Kalman gain: K = P*H'/S
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate: x = x + K*y
        self.x = self.x + K @ y
        
        # Update error covariance: P = (I - K*H)*P
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        
        return self.x

    def get_filtered_state(self):
        """Return the current filtered state estimate"""
        return self.x


# Function to use the dynamics model from your simulation
def simulation_model(state, action):
    """Simplified wrapper for your dynamics step function"""
    # This would call your dynamics_step function with state and apply the action (voltage)
    # You'd need to integrate it over your dt to get the next state
    
    # For example, using RK4 as in your original code:
    # (This is a placeholder - replace with your actual dynamics simulation)
    dt = 0.01  # Time step
    
    theta, alpha, theta_dot, alpha_dot = state
    vm = action
    
    # Call your dynamics_step (this is just an example)
    # Include all the actual physics from your simulation
    # k1 = dynamics_step([theta, alpha, theta_dot, alpha_dot], 0, vm)
    # k2 = ...
    # ...
    
    # Instead, for this example, let's use a simple approximation
    # Replace this with your actual simulation model
    next_theta = theta + dt * theta_dot
    next_alpha = alpha + dt * alpha_dot
    next_theta_dot = theta_dot + dt * (vm - 0.1*theta_dot)  # Simplified motor dynamics
    next_alpha_dot = alpha_dot + dt * (9.81 * np.sin(alpha) - 0.1*alpha_dot)  # Simplified pendulum dynamics
    
    return np.array([next_theta, next_alpha, next_theta_dot, next_alpha_dot])


# Integration with your reinforcement learning controller
def integrated_control_loop():
    """Example function showing integration of Kalman filter with RL controller"""
    # Initialize Kalman filter
    kf = PendulumKalmanFilter(dt=0.01)
    
    # Initialize RL model (from your existing code)
    # actor = Actor(state_dim=6, action_dim=1)
    # actor.load_state_dict(torch.load("model.pth"))
    # actor.eval()
    
    # Main control loop
    while True:
        # 1. Get raw measurements from QUBE hardware
        # (example - replace with your actual code to read sensors)
        raw_theta = qube.getMotorAngle()  # Adjusted for zero
        raw_alpha = qube.getPendulumAngle()
        raw_theta_dot = qube.getMotorRPM() * (2 * np.pi / 60)  # Convert RPM to rad/s
        
        # Calculate alpha_dot (you might need to implement this)
        # For example, using finite differences
        raw_alpha_dot = calculate_alpha_dot()  # Replace with your implementation
        
        # Combine into measurement vector
        measurement = np.array([raw_theta, raw_alpha, raw_theta_dot, raw_alpha_dot])
        
        # 2. Process measurements through Kalman filter
        
        # First predict step using simulation model and previous action
        # (if this is the first iteration, use zero or a reasonable default action)
        if 'previous_action' not in locals():
            previous_action = 0.0
        
        kf.predict(simulation_model, previous_action)
        
        # Then update step using real measurements
        filtered_state = kf.update(measurement)
        
        # 3. Prepare filtered state for neural network
        # Convert to observation format expected by your RL policy
        theta, alpha, theta_dot, alpha_dot = filtered_state
        
        # Normalize the pendulum angle as in your RL training
        pendulum_angle_norm = kf.normalize_angle(alpha + np.pi)
        
        # Create observation vector for RL policy (same format as in training)
        obs = np.array([
            np.sin(theta), np.cos(theta),
            np.sin(pendulum_angle_norm), np.cos(pendulum_angle_norm),
            theta_dot / 10.0,  # Scale velocities as in training
            alpha_dot / 10.0
        ])
        
        # 4. Get action from RL policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_mean, _ = actor(state_tensor)
            action = action_mean.numpy()[0][0]  # Get action as scalar
        
        # Scale action to voltage
        voltage = action * max_voltage
        
        # 5. Apply action to real system
        qube.setMotorVoltage(voltage)
        
        # Save action for next iteration
        previous_action = voltage
        
        # Wait for next control interval
        time.sleep(0.01)
