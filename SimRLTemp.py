import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba as nb
from time import time

# System Parameters - These are now base values that will vary between episodes
base_Rm = 8.94  # Motor resistance (Ohm)
base_Km = 0.0431  # Motor back-emf constant
base_Jm = 6e-5  # Total moment of inertia acting on motor shaft (kg·m^2)
base_bm = 3e-4  # Viscous damping coefficient (Nm/rad/s)
base_DA = 3e-4  # Damping coefficient of pendulum arm (Nm/rad/s)
base_DL = 5e-4  # Damping coefficient of pendulum link (Nm/rad/s)
base_mA = 0.053  # Weight of pendulum arm (kg)
base_mL = 0.024  # Weight of pendulum link (kg)
base_LA = 0.086  # Length of pendulum arm (m)
base_LL = 0.128  # Length of pendulum link (m)

# Constants that don't vary
g = 9.81  # Gravity constant (m/s^2)
max_voltage = 10.0  # Maximum motor voltage
THETA_MIN = -2.2  # Minimum arm angle (radians)
THETA_MAX = 2.2  # Maximum arm angle (radians)

# Current parameter values (will be updated per episode)
Rm = base_Rm
Km = base_Km
Jm = base_Jm
bm = base_bm
DA = base_DA
DL = base_DL
mA = base_mA
mL = base_mL
LA = base_LA
LL = base_LL
JA = mA * LA ** 2 * 7 / 48  # Inertia moment of pendulum arm (kg·m^2)
JL = mL * LL ** 2 / 3  # Inertia moment of pendulum link (kg·m^2)

# Pre-compute constants for optimization
half_mL_LL_g = 0.5 * mL * LL * g
half_mL_LL_LA = 0.5 * mL * LL * LA
quarter_mL_LL_squared = 0.25 * mL * LL ** 2
Mp_g_Lp = mL * g * LL
Jp = (1 / 3) * mL * LL ** 2  # Pendulum moment of inertia (kg·m²)

batch_size = 256 * 32  # Batch size for training


# Parameter Manager class to handle parameter variations
class ParameterManager:
    """Manages system parameters with random variations between episodes."""

    def __init__(self, variation_pct=0.1, fixed_params=False):
        """
        Initialize parameter manager.

        Args:
            variation_pct: Percentage variation allowed (0.1 = 10%)
            fixed_params: If True, parameters won't vary between episodes
        """
        self.variation_pct = variation_pct
        self.fixed_params = fixed_params

        # Store base parameter values
        self.base_params = {
            'Rm': base_Rm,
            'Km': base_Km,
            'Jm': base_Jm,
            'bm': base_bm,
            'DA': base_DA,
            'DL': base_DL,
            'mA': base_mA,
            'mL': base_mL,
            'LA': base_LA,
            'LL': base_LL
        }

        # Current parameter values (will be updated on reset)
        self.current_params = {}

        # History of parameter values used
        self.param_history = []

        # Initialize parameters
        self.reset()

    def reset(self):
        """Reset parameters with random variations if not fixed."""
        if self.fixed_params:
            # Use base parameters without variation
            self.current_params = self.base_params.copy()
        else:
            # Apply random variations to each parameter
            self.current_params = {}
            for name, base_value in self.base_params.items():
                # Generate random variation within ±variation_pct
                variation_factor = 1.0 + np.random.uniform(-self.variation_pct, self.variation_pct)
                self.current_params[name] = base_value * variation_factor

        # Store this set of parameters in history
        self.param_history.append(self.current_params.copy())

        # Update global parameters (needed for Numba compatibility)
        self._update_globals()

        return self.current_params

    def _update_globals(self):
        """Update global parameter values for use in Numba-accelerated functions."""
        global Rm, Km, Jm, bm, DA, DL, mA, mL, LA, LL, JA, JL
        global half_mL_LL_g, half_mL_LL_LA, quarter_mL_LL_squared, Mp_g_Lp, Jp

        # Update basic parameters
        Rm = self.current_params['Rm']
        Km = self.current_params['Km']
        Jm = self.current_params['Jm']
        bm = self.current_params['bm']
        DA = self.current_params['DA']
        DL = self.current_params['DL']
        mA = self.current_params['mA']
        mL = self.current_params['mL']
        LA = self.current_params['LA']
        LL = self.current_params['LL']

        # Compute derived parameters
        JA = mA * LA ** 2 * 7 / 48
        JL = mL * LL ** 2 / 3

        # Pre-compute constants for optimization
        half_mL_LL_g = 0.5 * mL * LL * g
        half_mL_LL_LA = 0.5 * mL * LL * LA
        quarter_mL_LL_squared = 0.25 * mL * LL ** 2
        Mp_g_Lp = mL * g * LL
        Jp = (1 / 3) * mL * LL ** 2

    def get_current_params(self):
        """Get current parameter values including computed ones."""
        params = self.current_params.copy()
        params.update({
            'JA': JA,
            'JL': JL,
            'half_mL_LL_g': half_mL_LL_g,
            'half_mL_LL_LA': half_mL_LL_LA,
            'quarter_mL_LL_squared': quarter_mL_LL_squared,
            'Mp_g_Lp': Mp_g_Lp,
            'Jp': Jp
        })
        return params

    def plot_parameter_history(self, save_path=None):
        """Plot the history of parameter variations."""
        if len(self.param_history) <= 1:
            print("Not enough episodes to plot parameter history")
            return

        param_names = list(self.base_params.keys())
        fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 2 * len(param_names)))

        for i, param_name in enumerate(param_names):
            base_value = self.base_params[param_name]
            values = [params[param_name] for params in self.param_history]

            # Convert to percentage variation from base
            pct_variations = [(val / base_value - 1.0) * 100 for val in values]

            axes[i].plot(pct_variations, 'b-')
            axes[i].axhline(y=0, color='r', linestyle='--')
            axes[i].axhline(y=self.variation_pct * 100, color='g', linestyle=':')
            axes[i].axhline(y=-self.variation_pct * 100, color='g', linestyle=':')
            axes[i].set_ylabel(f'{param_name} var %')
            axes[i].grid(True)

        axes[-1].set_xlabel('Episode')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


# Helper functions with Numba acceleration
@nb.njit(fastmath=True, cache=True)
def clip_value(value, min_value, max_value):
    """Fast custom clip function"""
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value


@nb.njit(fastmath=True, cache=True)
def apply_voltage_deadzone(vm):
    """Apply motor voltage dead zone"""
    if -0.2 <= vm <= 0.2:
        vm = 0.0
    return vm


@nb.njit(fastmath=True, cache=True)
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


@nb.njit(fastmath=True, cache=True)
def enforce_theta_limits(state):
    """Enforce hard limits on theta angle and velocity"""
    theta, alpha, theta_dot, alpha_dot = state

    # Apply hard limit on theta
    if theta > THETA_MAX:
        theta = THETA_MAX
        # If hitting upper limit with positive velocity, stop the motion
        if theta_dot > 0:
            theta_dot = 0.0
    elif theta < THETA_MIN:
        theta = THETA_MIN
        # If hitting lower limit with negative velocity, stop the motion
        if theta_dot < 0:
            theta_dot = 0.0

    return np.array([theta, alpha, theta_dot, alpha_dot])


@nb.njit(fastmath=True, cache=True)
def dynamics_step(state, t, vm):
    """Ultra-optimized dynamics calculation with theta limits, using current global parameters"""
    theta_m, theta_L, theta_m_dot, theta_L_dot = state[0], state[1], state[2], state[3]

    # Check theta limits - implement hard stops
    if (theta_m >= THETA_MAX and theta_m_dot > 0) or (theta_m <= THETA_MIN and theta_m_dot < 0):
        theta_m_dot = 0.0  # Stop the arm motion at the limits

    # Apply dead zone and calculate motor torque
    vm = apply_voltage_deadzone(vm)

    # Motor torque calculation
    im = (vm - Km * theta_m_dot) / Rm
    Tm = Km * im

    # Equations of motion coefficients from Eq. (9) in paper
    # For theta_m equation:
    M11 = mL * LA ** 2 + quarter_mL_LL_squared - quarter_mL_LL_squared * np.cos(theta_L) ** 2 + JA
    M12 = -half_mL_LL_LA * np.cos(theta_L)
    C1 = 0.5 * mL * LL ** 2 * np.sin(theta_L) * np.cos(theta_L) * theta_m_dot * theta_L_dot
    C2 = half_mL_LL_LA * np.sin(theta_L) * theta_L_dot ** 2

    # For theta_L equation:
    M21 = -half_mL_LL_LA * np.cos(theta_L)
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


# Time step generator based on real measurements
class VariableTimeGenerator:
    """Generate variable time steps based on real system measurements."""

    def __init__(
            self,
            mean=0.013926,
            std_dev=0.001871,
            min_dt=0.011,
            max_dt=0.041
    ):
        self.mean = mean
        self.std_dev = std_dev
        self.min_dt = min_dt
        self.max_dt = max_dt

    def get_next_dt(self):
        """Generate next time step following a normal distribution."""
        dt = np.random.normal(self.mean, self.std_dev)
        return clip_value(dt, self.min_dt, self.max_dt)


# Simulation environment for RL - MODIFIED to support parameter variation
class PendulumEnv:
    def __init__(self, dt=0.014, max_steps=1300, variable_dt=False, param_variation=0.1, fixed_params=False):
        """
        Initialize pendulum environment.

        Args:
            dt: Fixed time step (if variable_dt=False)
            max_steps: Maximum steps per episode
            variable_dt: If True, use variable time steps
            param_variation: Parameter variation percentage (0.1 = 10%)
            fixed_params: If True, parameters won't vary between episodes
        """
        self.fixed_dt = dt
        self.variable_dt = variable_dt

        # Initialize time generator if using variable time steps
        if variable_dt:
            self.time_generator = VariableTimeGenerator()
            self.dt = self.time_generator.get_next_dt()
        else:
            self.dt = self.fixed_dt

        # Initialize parameter manager
        self.param_manager = ParameterManager(
            variation_pct=param_variation,
            fixed_params=fixed_params
        )

        self.max_steps = max_steps
        self.step_count = 0
        self.state = None
        self.time_history = []  # Track actual time steps used

    def reset(self, random_init=True):
        # Reset dt if using variable time steps
        if self.variable_dt:
            self.dt = self.time_generator.get_next_dt()
        else:
            self.dt = self.fixed_dt

        # Reset parameters - this updates the global variables
        self.param_manager.reset()

        # Initialize with small random variations if requested
        if random_init:
            self.state = np.array([
                np.random.uniform(-0.1, 0.1),  # theta
                np.random.uniform(-0.1, 0.1),  # alpha (small angle from bottom)
                np.random.uniform(-0.05, 0.05),  # theta_dot
                np.random.uniform(-0.05, 0.05)  # alpha_dot
            ])
        else:
            self.state = np.array([0.0, 0.1, 0.0, 0.0])  # Default initial state

        self.step_count = 0
        self.time_history = []  # Reset time history
        return self._get_observation()

    def _get_observation(self):
        # Normalize the state for RL
        theta, alpha, theta_dot, alpha_dot = self.state

        # Normalize angles to [-π, π] for the observation
        alpha_norm = normalize_angle(alpha + np.pi)  # Normalized relative to upright

        # Create observation vector with sin/cos values for angles to avoid discontinuities
        obs = np.array([
            np.sin(theta), np.cos(theta),
            np.sin(alpha_norm), np.cos(alpha_norm),
            theta_dot / 10.0,  # Scale velocities to roughly [-1, 1]
            alpha_dot / 10.0
        ])

        return obs

    def step(self, action):
        # Convert normalized action [-1, 1] to voltage - safe conversion to float
        voltage = float(action.item() if hasattr(action, 'item') else action) * max_voltage

        # Record current dt value
        self.time_history.append(self.dt)

        # RK4 integration step with current dt
        self.state = self._rk4_step(self.state, voltage)

        # Update dt for next step if using variable time
        if self.variable_dt:
            self.dt = self.time_generator.get_next_dt()

        # Enforce limits
        self.state = enforce_theta_limits(self.state)

        # Get reward
        reward = self._compute_reward()

        # Check if done
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_observation(), reward, done, {}

    def _rk4_step(self, state, vm):
        """4th-order Runge-Kutta integrator step using current parameters"""
        # Apply limits to initial state
        state = enforce_theta_limits(state)

        k1 = dynamics_step(state, 0, vm)
        state_k2 = enforce_theta_limits(state + 0.5 * self.dt * k1)
        k2 = dynamics_step(state_k2, 0, vm)

        state_k3 = enforce_theta_limits(state + 0.5 * self.dt * k2)
        k3 = dynamics_step(state_k3, 0, vm)

        state_k4 = enforce_theta_limits(state + self.dt * k3)
        k4 = dynamics_step(state_k4, 0, vm)

        new_state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return enforce_theta_limits(new_state)

    def _compute_reward(self):
        theta, alpha, theta_dot, alpha_dot = self.state
        alpha_norm = normalize_angle(alpha + np.pi)  # Normalized for upright position

        # COMPONENT 1: Base reward for pendulum being upright (range: -1 to 1)
        # Uses cosine which is a naturally smooth function
        upright_reward = 2.0 * np.cos(alpha_norm)

        # COMPONENT 2: Smooth penalty for high velocities - quadratic falloff
        # Use tanh to create a smoother penalty that doesn't grow excessively large
        velocity_norm = (theta_dot ** 2 + alpha_dot ** 2) / 10.0  # Normalize velocities
        velocity_penalty = -0.3 * np.tanh(velocity_norm)  # Bounded penalty

        # COMPONENT 3: Smooth penalty for arm position away from center
        # Again using tanh for smooth bounded penalties
        pos_penalty = -0.1 * np.tanh(theta ** 2 / 2.0)

        # COMPONENT 4: Smoother bonus for being close to upright position
        upright_closeness = np.exp(-10.0 * alpha_norm ** 2)  # Close to 1 when near upright, falls off quickly
        stability_factor = np.exp(-1.0 * alpha_dot ** 2)  # Close to 1 when velocity is low
        bonus = 3.0 * upright_closeness * stability_factor  # Smoothly scales based on both factors

        # COMPONENT 4.5: Smoother cost for being close to downright position
        # For new convention, downright is at π
        downright_alpha = normalize_angle(alpha - np.pi)
        downright_closeness = np.exp(-10.0 * downright_alpha ** 2)
        stability_factor = np.exp(-1.0 * alpha_dot ** 2)
        bonus += -3.0 * downright_closeness * stability_factor  # Smoothly scales based on both factors

        # COMPONENT 5: Smoother penalty for approaching limits
        # Create a continuous penalty that increases as the arm approaches limits
        # Map the distance to limits to a 0-1 range, with 1 being at the limit
        theta_max_dist = np.clip(1.0 - abs(theta - THETA_MAX) / 0.5, 0, 1)
        theta_min_dist = np.clip(1.0 - abs(theta - THETA_MIN) / 0.5, 0, 1)
        limit_distance = max(theta_max_dist, theta_min_dist)

        # Apply a nonlinear function to create gradually increasing penalty
        # The penalty grows more rapidly as the arm gets very close to limits
        limit_penalty = -10.0 * limit_distance ** 3

        # COMPONENT 6: Energy management reward
        # This component is already quite smooth, just adjust scaling
        energy_reward = 2 - 0.15 * abs(Mp_g_Lp * (np.cos(alpha_norm))
                                       + 0.5 * Jp * alpha_dot ** 2
                                       - Mp_g_Lp)

        # Combine all components
        reward = (
                upright_reward
                # + velocity_penalty
                + pos_penalty
                + bonus
                + limit_penalty
                + energy_reward
        )

        return reward

    def get_current_parameters(self):
        """Get current parameter values for reporting and analysis."""
        return self.param_manager.get_current_params()


# LSTM-based Actor (Policy) Network for improved sim-to-real transfer
class LSTMActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lstm_layers=1):
        super(LSTMActor, self).__init__()

        # LSTM layer - using LSTMCell for more control and efficiency
        self.lstm_cell = nn.LSTMCell(state_dim, hidden_dim)

        # Post-LSTM fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Action mean and log standard deviation
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Hidden state for inference (single sample)
        self.inference_hidden = None
        self.inference_cell = None

        # State dimension for creating temporary states
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Initialize parameters with appropriate values for better initial performance
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def reset_hidden(self):
        """Reset the hidden state - call at the beginning of each episode"""
        self.inference_hidden = None
        self.inference_cell = None

    def forward(self, state, use_stored_state=True):
        """
        Forward pass to get action distribution parameters
        Args:
            state: Current state tensor
            use_stored_state: If True, use and update the stored hidden state
                             If False, use a fresh hidden state (for batch processing)
        Returns:
            action_mean: Mean of action distribution
            action_log_std: Log standard deviation of action distribution
        """
        batch_size = state.size(0) if len(state.size()) > 1 else 1
        device = state.device

        if use_stored_state and batch_size == 1:
            # Single sample inference case - use the stored state
            if self.inference_hidden is None or self.inference_cell is None:
                self.inference_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
                self.inference_cell = torch.zeros(batch_size, self.hidden_dim, device=device)

            # Update the stored state
            self.inference_hidden, self.inference_cell = self.lstm_cell(
                state, (self.inference_hidden, self.inference_cell))
            features = self.fc(self.inference_hidden)
        else:
            # Batch processing case - use a temporary state
            # This is used during training updates with multiple samples
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, device=device)
            h, c = self.lstm_cell(state, (h, c))
            features = self.fc(h)

        # Get action distribution parameters
        action_mean = torch.tanh(self.mean(features))
        action_log_std = torch.clamp(self.log_std(features), -20, 2)

        return action_mean, action_log_std

    def sample(self, state, use_stored_state=True):
        """
        Sample action from the policy
        Args:
            state: Current state tensor
            use_stored_state: If True, use and update the stored hidden state
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
        """
        mean, log_std = self.forward(state, use_stored_state)
        std = log_std.exp()

        # Sample from normal distribution
        normal = Normal(mean, std)
        x = normal.rsample()  # Reparameterization trick

        # Constrain to [-1, 1]
        action = torch.tanh(x)

        # Calculate log probability
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


# Critic (Value) Network - unchanged from original
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 architecture (for Twin Delayed DDPG)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2


# LSTM-based Soft Actor-Critic (SAC) Agent
class LSTMSACAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=128,
            critic_dim=128,  # Larger critic network
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            automatic_entropy_tuning=True
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Initialize networks
        self.actor = LSTMActor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, critic_dim)
        self.critic_target = Critic(state_dim, action_dim, critic_dim)

        # Copy parameters to target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        if automatic_entropy_tuning:
            self.log_alpha = self.log_alpha.to(self.device)

    def select_action(self, state, evaluate=False):
        """Select an action from the policy"""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate:
            # Use mean action (no exploration)
            with torch.no_grad():
                action, _ = self.actor(state)
                return action.cpu().numpy()[0]
        else:
            # Sample action with exploration
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy()[0]

    def reset(self):
        """Reset LSTM states at the beginning of each episode"""
        self.actor.reset_hidden()

    def update_parameters(self, memory, batch_size=batch_size):
        """Update policy and value parameters using batch of experience tuples"""
        # Sample batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            # Get next actions and log probs from current policy
            # Use batch processing with fresh hidden states (not stored ones)
            next_action, next_log_prob = self.actor.sample(next_state_batch, use_stored_state=False)

            # Get target Q values
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)

            # Compute target with entropy
            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            target_q = target_q - alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        # Current Q estimates
        current_q1, current_q2 = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        # Use batch processing with fresh hidden states (not stored ones)
        actions, log_probs = self.actor.sample(state_batch, use_stored_state=False)

        # Get Q-values for these actions
        q1, q2 = self.critic(state_batch, actions)
        min_q = torch.min(q1, q2)

        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # Actor loss (maximize Q - alpha * log_prob)
        actor_loss = (alpha * log_probs - min_q).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update automatic entropy tuning parameter
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }


# Replay Buffer - unchanged from original
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = map(np.array, zip(*[self.buffer[i] for i in batch]))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# Function to load pre-trained models
def load_agent(actor_path=None, critic_path=None, state_dim=6, action_dim=1, hidden_dim=128, critic_dim=128):
    """
    Load pre-trained actor and critic models and create an agent.

    Args:
        actor_path (str): Path to the saved actor model
        critic_path (str): Path to the saved critic model
        state_dim (int): State dimension
        action_dim (int): Action dimension
        hidden_dim (int): Hidden layer dimension for actor
        critic_dim (int): Hidden layer dimension for critic

    Returns:
        LSTMSACAgent: Agent with loaded models
    """
    # Initialize a new agent
    agent = LSTMSACAgent(state_dim, action_dim, hidden_dim, critic_dim)

    # Load actor if path is provided
    if actor_path:
        agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
        print(f"LSTM Actor model loaded from {actor_path}")

    # Load critic if path is provided
    if critic_path:
        agent.critic.load_state_dict(torch.load(critic_path, map_location=agent.device))
        # Update target critic as well
        agent.critic_target.load_state_dict(torch.load(critic_path, map_location=agent.device))
        print(f"Critic model loaded from {critic_path}")

    return agent


# Modified training function to use LSTM-SAC
def train(actor_path=None, critic_path=None, variable_dt=False, param_variation=0.1, fixed_params=False):
    print("Starting LSTM-SAC training for inverted pendulum control...")
    print(f"Using variable time steps: {variable_dt}")
    print(f"Using parameter variation: {param_variation if not fixed_params else 'OFF'}")

    # Environment setup with variable time step and parameter variation options
    env = PendulumEnv(
        variable_dt=variable_dt,
        param_variation=param_variation,
        fixed_params=fixed_params
    )

    # Hyperparameters
    state_dim = 6  # Our observation space
    action_dim = 1  # Motor voltage (normalized)
    max_episodes = 1000
    max_steps = 1300
    batch_size = 256
    replay_buffer_size = 100000
    updates_per_step = 1

    # Initialize agent (with pre-trained models if provided)
    if actor_path or critic_path:
        agent = load_agent(actor_path, critic_path, state_dim, action_dim)
    else:
        agent = LSTMSACAgent(state_dim, action_dim)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # Metrics
    episode_rewards = []
    avg_rewards = []
    parameter_variations = []

    # For visualization - store episode data
    states_history = []
    actions_history = []

    # Training loop
    start_time = time()

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        # Reset LSTM hidden state for new episode
        agent.reset()

        # Store current parameter set for analysis
        current_params = env.get_current_parameters()
        parameter_variations.append(current_params)

        # Track losses and alpha for this episode
        critic_losses = []
        actor_losses = []
        alpha_values = []

        # If we're going to plot this episode, prepare to collect state data
        plot_this_episode = (episode + 1) % 10 == 0
        if plot_this_episode:
            episode_states = []
            episode_actions = []

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # If we're plotting, store the raw state and action
            if plot_this_episode:
                episode_states.append(env.state.copy())
                episode_actions.append(action)

            state = next_state
            episode_reward += reward

            # Update if enough samples
            if len(replay_buffer) > batch_size:
                for _ in range(updates_per_step):
                    update_info = agent.update_parameters(replay_buffer, batch_size)
                    critic_losses.append(update_info['critic_loss'])
                    actor_losses.append(update_info['actor_loss'])
                    # Store alpha if using automatic entropy tuning
                    if agent.automatic_entropy_tuning:
                        alpha_values.append(agent.alpha.item())

            if done:
                break

        # Log progress
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)

        # Calculate average losses and alpha for this episode
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_alpha = np.mean(alpha_values) if alpha_values else 0.0

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{max_episodes} | Reward: {episode_reward:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | C_Loss: {avg_critic_loss:.4f} | "
                f"A_Loss: {avg_actor_loss:.4f} | Alpha: {avg_alpha:.4f}")

            # Print some current parameter values
            if not fixed_params:
                print(
                    f"  Current Rm: {current_params['Rm']:.4f} | Km: {current_params['Km']:.6f} | mL: {current_params['mL']:.6f}")

            # Plot simulation for visual progress tracking
            if plot_this_episode:
                plot_training_episode(episode, episode_states, episode_actions,
                                      env.time_history if variable_dt else [env.dt] * len(episode_states),
                                      episode_reward)

            # Plot parameter variation history
            if not fixed_params and episode > 10:
                env.param_manager.plot_parameter_history(f"param_variations_ep{episode + 1}.png")

        # Early stopping if well trained
        if avg_reward > 4900 and episode > 150:
            print(f"Environment solved in {episode + 1} episodes!")
            break

    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")

    # Save trained model
    timestamp = int(time())
    torch.save(agent.actor.state_dict(), f"lstm_actor_{timestamp}.pth")
    torch.save(agent.critic.state_dict(), f"lstm_critic_{timestamp}.pth")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='100-Episode Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('LSTM-SAC Training for Inverted Pendulum')
    plt.legend()
    plt.grid(True)
    plt.savefig("lstm_sac_training_progress.png")
    plt.close()

    # Plot time step distribution if variable time was used
    if variable_dt and env.time_history:
        plt.figure(figsize=(10, 5))
        plt.hist(env.time_history, bins=30, alpha=0.7)
        plt.axvline(x=np.mean(env.time_history), color='r', linestyle='--',
                    label=f'Mean: {np.mean(env.time_history):.6f}')
        plt.axvline(x=np.median(env.time_history), color='g', linestyle='--',
                    label=f'Median: {np.median(env.time_history):.6f}')
        plt.xlabel('Time Step (s)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Variable Time Steps')
        plt.legend()
        plt.grid(True)
        plt.savefig("time_step_distribution.png")
        plt.close()

    # Final parameter variation plot
    if not fixed_params:
        env.param_manager.plot_parameter_history("param_variations_final.png")

    return agent


def plot_training_episode(episode, states_history, actions_history, dt_history, episode_reward):
    """Plot the pendulum state evolution for a training episode"""
    # Convert history to numpy arrays
    states_history = np.array(states_history)
    actions_history = np.array(actions_history)

    # Extract components
    thetas = states_history[:, 0]
    alphas = states_history[:, 1]
    theta_dots = states_history[:, 2]
    alpha_dots = states_history[:, 3]

    # Normalize alpha for visualization
    alpha_normalized = np.zeros(len(alphas))
    for i in range(len(alphas)):
        alpha_normalized[i] = normalize_angle(alphas[i] + np.pi)

    # Generate time array - calculate cumulative time if variable dt
    if isinstance(dt_history, list) and len(dt_history) > 0:
        # Create cumulative time array
        t = np.zeros(len(dt_history))
        t[0] = dt_history[0]
        for i in range(1, len(dt_history)):
            t[i] = t[i - 1] + dt_history[i]
    else:
        # Use fixed dt
        t = np.arange(len(states_history)) * dt_history[0]

    # Count performance metrics
    balanced_time = 0.0
    num_upright_points = 0

    for i in range(len(alpha_normalized)):
        # Check if pendulum is close to upright
        if abs(alpha_normalized[i]) < 0.17:  # about 10 degrees
            if i < len(dt_history):
                balanced_time += dt_history[i]
            else:
                balanced_time += dt_history[0]  # Use first dt if index out of range
            num_upright_points += 1

    # Plot results
    plt.figure(figsize=(12, 9))

    # Plot arm angle
    plt.subplot(3, 1, 1)
    plt.plot(t, thetas, 'b-')
    plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.7, label='Theta Limits')
    plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.7)
    plt.ylabel('Arm angle (rad)')
    plt.title(f'Training Episode {episode} | Reward: {episode_reward:.2f} | Balanced: {balanced_time:.2f}s')
    plt.legend()
    plt.grid(True)

    # Plot pendulum angle
    plt.subplot(3, 1, 2)
    plt.plot(t, alpha_normalized, 'g-')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Line at upright position
    plt.axhline(y=0.17, color='r', linestyle=':', alpha=0.5)  # Upper balanced threshold
    plt.axhline(y=-0.17, color='r', linestyle=':', alpha=0.5)  # Lower balanced threshold
    plt.ylabel('Pendulum angle (rad)')
    plt.grid(True)

    # Plot control actions
    plt.subplot(3, 1, 3)
    plt.plot(t, actions_history * max_voltage, 'r-')  # Scale back to actual voltage
    plt.xlabel('Time (s)')
    plt.ylabel('Control voltage (V)')
    plt.ylim([-max_voltage * 1.1, max_voltage * 1.1])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"lstm_training_episode_{episode}.png")
    plt.close()  # Close the figure to avoid memory issues


# Modified evaluate function to include parameter variation analysis
def evaluate(agent, num_episodes=5, render=True, variable_dt=False, param_variation=0.1, fixed_params=False):
    """Evaluate the trained agent's performance"""
    env = PendulumEnv(
        variable_dt=variable_dt,
        param_variation=param_variation,
        fixed_params=fixed_params
    )

    # For storing performance metrics
    all_rewards = []
    all_balance_times = []
    all_params = []  # Store parameters for each episode

    for episode in range(num_episodes):
        state = env.reset(random_init=False)  # Start from standard position
        total_reward = 0

        # Reset LSTM hidden state for new episode
        agent.reset()

        # Store the parameters for this episode
        current_params = env.get_current_parameters()
        all_params.append(current_params)

        # Store episode data for plotting
        states_history = []
        actions_history = []

        for step in range(env.max_steps):
            # Select action
            action = agent.select_action(state, evaluate=True)

            # Step environment
            next_state, reward, done, _ = env.step(action)

            # Store data
            states_history.append(env.state.copy())  # Store raw state
            actions_history.append(action)

            total_reward += reward
            state = next_state

            if done:
                break

        all_rewards.append(total_reward)

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

        # Print parameter values if varying
        if not fixed_params:
            print(
                f"  Parameters - Rm: {current_params['Rm']:.4f}, Km: {current_params['Km']:.6f}, mL: {current_params['mL']:.6f}")

        if render:
            # Convert history to numpy arrays
            states_history = np.array(states_history)
            actions_history = np.array(actions_history)

            # Extract components
            thetas = states_history[:, 0]
            alphas = states_history[:, 1]
            theta_dots = states_history[:, 2]
            alpha_dots = states_history[:, 3]

            # Normalize alpha for visualization
            alpha_normalized = np.zeros(len(alphas))
            for i in range(len(alphas)):
                alpha_normalized[i] = normalize_angle(alphas[i] + np.pi)

            # Generate time array - calculate cumulative time for variable dt
            if variable_dt:
                # Create cumulative time array
                t = np.zeros(len(env.time_history))
                t[0] = env.time_history[0]
                for i in range(1, len(env.time_history)):
                    t[i] = t[i - 1] + env.time_history[i]
            else:
                # Use fixed dt
                t = np.arange(len(states_history)) * env.dt

            # Count performance metrics
            balanced_time = 0.0
            num_upright_points = 0

            for i in range(len(alpha_normalized)):
                # Check if pendulum is close to upright
                if abs(alpha_normalized[i]) < 0.17:  # about 10 degrees
                    if variable_dt and i < len(env.time_history):
                        balanced_time += env.time_history[i]
                    else:
                        balanced_time += env.dt
                    num_upright_points += 1

            all_balance_times.append(balanced_time)

            # Plot results
            plt.figure(figsize=(14, 12))

            # Plot arm angle
            plt.subplot(4, 1, 1)
            plt.plot(t, thetas, 'b-')
            plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.7, label='Theta Limits')
            plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.7)
            plt.ylabel('Arm angle (rad)')

            title = f'LSTM-SAC Inverted Pendulum Control - Episode {episode + 1}'
            if variable_dt:
                title += " (Variable dt)"
            if not fixed_params:
                title += f" (Param var: {param_variation * 100:.0f}%)"
            plt.title(title)

            plt.legend()
            plt.grid(True)

            # Plot pendulum angle
            plt.subplot(4, 1, 2)
            plt.plot(t, alpha_normalized, 'g-')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Line at upright position
            plt.ylabel('Pendulum angle (rad)')
            plt.grid(True)

            # Plot motor voltage (actions)
            plt.subplot(4, 1, 3)
            plt.plot(t, actions_history * max_voltage, 'r-')  # Scale back to actual voltage
            plt.ylabel('Control voltage (V)')
            plt.ylim([-max_voltage * 1.1, max_voltage * 1.1])
            plt.grid(True)

            # Plot angular velocities
            plt.subplot(4, 1, 4)
            plt.plot(t, theta_dots, 'b-', label='Arm velocity')
            plt.plot(t, alpha_dots, 'g-', label='Pendulum velocity')
            plt.xlabel('Time (s)')
            plt.ylabel('Angular velocity (rad/s)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"lstm_evaluation_episode_{episode + 1}.png")
            plt.close()

            print(f"Time spent balanced: {balanced_time:.2f} seconds")
            print(f"Data points with pendulum upright: {num_upright_points}")
            print(f"Max arm angle: {np.max(np.abs(thetas)):.2f} rad")
            print(f"Max pendulum angular velocity: {np.max(np.abs(alpha_dots)):.2f} rad/s")
            final_angle_deg = abs(alpha_normalized[-1]) * 180 / np.pi
            print(
                f"Final pendulum angle from vertical: {abs(alpha_normalized[-1]):.2f} rad ({final_angle_deg:.1f} degrees)")
            print("-" * 50)

            # If variable dt was used, plot the time step distribution
            if variable_dt and env.time_history:
                plt.figure(figsize=(10, 5))
                plt.hist(env.time_history, bins=30, alpha=0.7)
                plt.axvline(x=np.mean(env.time_history), color='r', linestyle='--',
                            label=f'Mean: {np.mean(env.time_history):.6f}')
                plt.axvline(x=np.median(env.time_history), color='g', linestyle='--',
                            label=f'Median: {np.median(env.time_history):.6f}')
                plt.xlabel('Time Step (s)')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of Variable Time Steps - Episode {episode + 1}')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"time_step_distribution_ep_{episode + 1}.png")
                plt.close()

    # Print summary statistics
    print("\n===== Evaluation Summary =====")
    print(f"Average reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    if all_balance_times:
        print(f"Average balanced time: {np.mean(all_balance_times):.2f} ± {np.std(all_balance_times):.2f} seconds")

    # If parameters varied, analyze robustness
    if not fixed_params and len(all_params) > 1:
        print("\n===== Parameter Variation Analysis =====")
        param_keys = ['Rm', 'Km', 'mL', 'LA', 'LL']  # Key parameters to analyze

        # Create correlation analysis between parameters and performance
        if len(all_rewards) > 3:  # Only meaningful with enough data points
            plt.figure(figsize=(15, 10))
            for i, param in enumerate(param_keys):
                values = [p[param] for p in all_params]

                # Calculate correlation
                if len(set(values)) > 1:  # Only if parameter actually varied
                    corr = np.corrcoef(values, all_rewards)[0, 1]
                    plt.subplot(len(param_keys), 1, i + 1)
                    plt.scatter(values, all_rewards)
                    plt.xlabel(f'{param} value')
                    plt.ylabel('Reward')
                    plt.title(f'Correlation between {param} and reward: {corr:.3f}')
                    plt.grid(True)

            plt.tight_layout()
            plt.savefig("lstm_parameter_reward_correlation.png")
            plt.close()


if __name__ == "__main__":
    print("TorchRL LSTM-SAC Inverted Pendulum Training and Evaluation")
    print("=" * 50)

    # Available training options:
    # 1. Train a new LSTM-SAC agent with variable time steps and parameter variation
    # agent = train(variable_dt=True, param_variation=0.4)

    # 2. Continue training from pre-trained model with high parameter variation
    # agent = train(actor_path="lstm_actor_1740000000.pth", critic_path="lstm_critic_1740000000.pth",
    #               variable_dt=True, param_variation=0.5)

    agent = train(variable_dt=False, param_variation=False)

    print("\nEvaluating trained LSTM-SAC agent with parameter variation...")
    evaluate(agent, num_episodes=3, variable_dt=True, param_variation=0.5)

    # 4. Test with larger parameter variation to check robustness
    print("\nEvaluating trained LSTM-SAC agent with LARGER parameter variation...")
    evaluate(agent, num_episodes=3, variable_dt=True, param_variation=0.7)

    print("=" * 50)
    print("LSTM-SAC TRAINING COMPLETE")
    print("=" * 50)