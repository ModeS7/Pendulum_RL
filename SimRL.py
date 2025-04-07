import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba as nb
from time import time
from numpy.ma.core import arctan2

# ====== System Constants ======
g = 9.81  # Gravity constant (m/s^2)
base_max_voltage = 4.0  # Base maximum motor voltage
THETA_MIN = -2.2  # Minimum arm angle (radians)
THETA_MAX = 2.2  # Maximum arm angle (radians)

# ====== Base Parameter Values ======
# Motor parameters
base_Rm = 8.94  # Motor resistance (Ohm)
base_Km = 0.0431  # Motor back-emf constant

# Damping coefficients
base_DA = 0.0004  # Viscous friction coefficient for arm (N·m·s/rad)
base_DL = 0.000003  # Viscous friction coefficient for pendulum (N·m·s/rad)

# Mass and length
base_mA = 0.053  # Weight of pendulum arm (kg)
base_mL = 0.024  # Weight of pendulum link (kg)
base_LA = 0.086  # Length of pendulum arm (m)
base_LL = 0.128  # Length of pendulum link (m)

# Inertia and spring
base_JA = 0.0000572 + 0.00006  # Arm inertia about pivot (kg·m²)
base_JL = 0.0000235  # Pendulum inertia about pivot (kg·m²)
base_k = 0.002  # Torsional spring constant (N·m/rad)


# ====== Helper Functions ======
@nb.njit(fastmath=True, cache=True)
def clip_value(value, min_value, max_value):
    """Fast custom clip function"""
    return min_value if value < min_value else (max_value if value > max_value else value)


@nb.njit(fastmath=True, cache=True)
def apply_voltage_deadzone(vm):
    """Apply motor voltage dead zone"""
    return 0.0 if -0.2 <= vm <= 0.2 else vm


@nb.njit(fastmath=True, cache=True)
def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    angle = angle % (2 * np.pi)
    return angle - 2 * np.pi if angle > np.pi else angle


@nb.njit(fastmath=True, cache=True)
def enforce_theta_limits(state):
    """Enforce hard limits on theta angle and velocity"""
    theta_0, theta_1, theta_0_dot, theta_1_dot = state

    # Apply hard limit on theta_0 (arm angle)
    if theta_0 > THETA_MAX:
        theta_0 = THETA_MAX
        # If hitting upper limit with positive velocity, stop the motion
        if theta_0_dot > 0:
            theta_0_dot = 0.0
    elif theta_0 < THETA_MIN:
        theta_0 = THETA_MIN
        # If hitting lower limit with negative velocity, stop the motion
        if theta_0_dot < 0:
            theta_0_dot = 0.0

    return np.array([theta_0, theta_1, theta_0_dot, theta_1_dot])


# ====== Parameter Manager ======
class ParameterManager:
    """Manages system parameters with random variations between episodes."""

    def __init__(
            self,
            variation_pct=0.1,
            fixed_params=False,
            voltage_range=None
    ):
        """
        Initialize parameter manager.

        Args:
            variation_pct: Percentage variation allowed (0.1 = 10%)
            fixed_params: If True, parameters won't vary between episodes
            voltage_range: Tuple (min_voltage, max_voltage) for varying max_voltage
                          If None, uses base_max_voltage with normal variation
        """
        self.variation_pct = variation_pct
        self.fixed_params = fixed_params
        self.voltage_range = voltage_range

        # Store base parameter values
        self.base_params = {
            'Rm': base_Rm,
            'Km': base_Km,
            'DA': base_DA,
            'DL': base_DL,
            'mA': base_mA,
            'mL': base_mL,
            'LA': base_LA,
            'LL': base_LL,
            'JA': base_JA,
            'JL': base_JL,
            'k': base_k,
            'max_voltage': base_max_voltage  # Added max_voltage as a parameter
        }

        # Current parameter values and history
        self.current_params = {}
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
                # Special handling for max_voltage if voltage_range is provided
                if name == 'max_voltage' and self.voltage_range is not None:
                    min_v, max_v = self.voltage_range
                    self.current_params[name] = np.random.uniform(min_v, max_v)
                else:
                    # Generate random variation within ±variation_pct
                    variation_factor = 1.0 + np.random.uniform(-self.variation_pct, self.variation_pct)
                    self.current_params[name] = base_value * variation_factor

        # Store this set of parameters in history
        self.param_history.append(self.current_params.copy())

        # Compute derived parameters
        self.current_params['l_1'] = self.current_params['LL'] / 2  # Pendulum length to CoM
        self.current_params['half_mL_l1_g'] = (
                self.current_params['mL'] * self.current_params['l_1'] * g  # Weight moment
        )

        return self.current_params

    def get_current_params(self):
        """Get current parameter values."""
        return self.current_params.copy()

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


# ====== Time Step Generator ======
class VariableTimeGenerator:
    """Generate variable time steps based on real system measurements."""

    def __init__(
            self,
            mean=0.005,
            std_dev=0.002,
            min_dt=0.0025,
            max_dt=0.01
    ):
        self.mean = mean
        self.std_dev = std_dev
        self.min_dt = min_dt
        self.max_dt = max_dt

    def get_next_dt(self):
        """Generate next time step following a normal distribution."""
        dt = np.random.normal(self.mean, self.std_dev)
        return clip_value(dt, self.min_dt, self.max_dt)


# ====== Dynamics Function ======
def dynamics_step(state, t, vm, params):
    """
    Physics-based dynamics calculation for the inverted pendulum.

    Args:
        state: System state [theta_0, theta_1, theta_0_dot, theta_1_dot]
        t: Current time (not used but included for compatibility)
        vm: Motor voltage
        params: Dictionary of current system parameters

    Returns:
        Array of state derivatives
    """
    theta_0, theta_1, theta_0_dot, theta_1_dot = state

    # Extract parameters
    Rm = params['Rm']
    Km = params['Km']
    DA = params['DA']
    DL = params['DL']
    mL = params['mL']
    LA = params['LA']
    l_1 = params['l_1']  # Half of LL (pendulum length to CoM)
    JA = params['JA']
    JL = params['JL']
    k = params['k']

    # Calculate sines and cosines for dynamics
    s0, c0 = np.sin(theta_0), np.cos(theta_0)
    s1, c1 = np.sin(theta_1), np.cos(theta_1)

    # Apply hard limits on arm angle
    if (theta_0 >= THETA_MAX and theta_0_dot > 0) or (theta_0 <= THETA_MIN and theta_0_dot < 0):
        theta_0_dot = 0.0

    # Apply motor voltage deadzone and calculate torque
    vm = 0.0 if -0.2 <= vm <= 0.2 else vm
    torque = Km * (vm - Km * theta_0_dot) / Rm

    # Set up the mass matrix and force vector according to the model
    alpha = JA + mL * LA ** 2 + mL * l_1 ** 2 * s1 ** 2
    beta = -mL * l_1 ** 2 * (2 * s1 * c1)
    gamma = -mL * LA * l_1 * c1
    sigma = mL * LA * l_1 * s1

    # Mass matrix
    M = np.array([
        [-alpha, -gamma],
        [-gamma, -(JL + mL * l_1 ** 2)]
    ])

    # Force vector
    f = np.array([
        -torque + DA * theta_0_dot + k * arctan2(s0, c0) + sigma * theta_1_dot ** 2 - beta * theta_0_dot * theta_1_dot,
        DL * theta_1_dot + mL * g * l_1 * s1 + 0.5 * beta * theta_0_dot ** 2
    ])

    # Solve for accelerations
    det_M = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]

    # Handle near-singular matrix
    if abs(det_M) < 1e-10:
        theta_0_ddot = 0
        theta_1_ddot = 0
    else:
        # Solve using Cramer's rule
        theta_0_ddot = (M[1, 1] * f[0] - M[0, 1] * f[1]) / det_M
        theta_1_ddot = (M[0, 0] * f[1] - M[1, 0] * f[0]) / det_M

    return np.array([theta_0_dot, theta_1_dot, theta_0_ddot, theta_1_ddot])


# ====== Environment ======
class PendulumEnv:
    """Simulation environment for the inverted pendulum control."""

    def __init__(
            self,
            dt=0.005,
            max_steps=2000,
            variable_dt=False,
            param_variation=0.1,
            fixed_params=False,
            voltage_range=None
    ):
        """
        Initialize pendulum environment.

        Args:
            dt: Fixed time step (if variable_dt=False)
            max_steps: Maximum steps per episode
            variable_dt: If True, use variable time steps
            param_variation: Parameter variation percentage (0.1 = 10%)
            fixed_params: If True, parameters won't vary between episodes
            voltage_range: Tuple (min_voltage, max_voltage) for varying max_voltage
                          If None, uses base_max_voltage with normal variation
        """
        self.fixed_dt = dt
        self.variable_dt = variable_dt
        self.voltage_range = voltage_range

        # Initialize time generator if using variable time steps
        if variable_dt:
            self.time_generator = VariableTimeGenerator()
            self.dt = self.time_generator.get_next_dt()
        else:
            self.dt = self.fixed_dt

        # Initialize parameter manager
        self.param_manager = ParameterManager(
            variation_pct=param_variation,
            fixed_params=fixed_params,
            voltage_range=voltage_range
        )

        self.max_steps = max_steps
        self.step_count = 0
        self.state = None
        self.time_history = []

        # Initialize parameters (including max_voltage)
        self.params = None

    def reset(self, random_init=True):
        """Reset the environment to initial state."""
        # Reset dt if using variable time steps
        if self.variable_dt:
            self.dt = self.time_generator.get_next_dt()
        else:
            self.dt = self.fixed_dt

        # Reset parameters
        self.params = self.param_manager.reset()

        # Initialize state
        if random_init:
            self.state = np.array([
                np.random.uniform(-0.1, 0.1),  # theta_0 (arm angle)
                np.random.uniform(-0.1, 0.1),  # theta_1 (pendulum angle from upright)
                np.random.uniform(-0.05, 0.05),  # theta_0_dot
                np.random.uniform(-0.05, 0.05)  # theta_1_dot
            ])
        else:
            # Default initial state - pendulum slightly off upright
            self.state = np.array([0.0, 0.1, 0.0, 0.0])

        self.step_count = 0
        self.time_history = []  # Reset time history
        return self._get_observation()

    def _get_observation(self):
        """Convert raw state to agent observation."""
        # Get state values
        theta_0, theta_1, theta_0_dot, theta_1_dot = self.state

        # Normalize theta_1 for observation
        theta_1_norm = normalize_angle(theta_1 + np.pi)

        # Create observation with sin/cos representation of angles
        obs = np.array([
            np.sin(theta_0), np.cos(theta_0),
            np.sin(theta_1_norm), np.cos(theta_1_norm),
            theta_0_dot / 10.0,  # Scale velocities to roughly [-1, 1]
            theta_1_dot / 10.0
        ])

        return obs

    def step(self, action):
        """Take a step in the environment with the given action."""
        # Convert normalized action [-1, 1] to voltage using current max_voltage
        max_voltage = self.params['max_voltage']
        voltage = float(action[0]) * max_voltage

        # Store the voltage for reward calculation
        self.last_voltage = voltage

        # Record current dt value
        self.time_history.append(self.dt)

        # RK4 integration step
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
        """4th-order Runge-Kutta integrator step."""
        # Apply limits to initial state
        state = enforce_theta_limits(state)

        k1 = dynamics_step(state, 0, vm, self.params)
        state_k2 = enforce_theta_limits(state + 0.5 * self.dt * k1)
        k2 = dynamics_step(state_k2, 0, vm, self.params)

        state_k3 = enforce_theta_limits(state + 0.5 * self.dt * k2)
        k3 = dynamics_step(state_k3, 0, vm, self.params)

        state_k4 = enforce_theta_limits(state + self.dt * k3)
        k4 = dynamics_step(state_k4, 0, vm, self.params)

        new_state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return enforce_theta_limits(new_state)

    def _compute_reward(self):
        """Calculate reward based on current state."""
        theta_0, theta_1, theta_0_dot, theta_1_dot = self.state

        # Save the last action (voltage) for reward calculation
        # This assumes the action has been stored during the step() method
        last_voltage = getattr(self, 'last_voltage', 0.0)

        # Normalize theta_1 for reward calculation
        theta_1_norm = normalize_angle(theta_1 + np.pi)

        # ===== Reward Components =====
        # 1. Base reward for pendulum being upright (range: -1 to 1)
        upright_reward = 1.0 * np.cos(theta_1_norm)

        # 2. Penalty for high velocities
        velocity_norm = (theta_0_dot ** 2 + theta_1_dot ** 2) / 10.0
        velocity_penalty = -0.05 * np.tanh(velocity_norm)

        # 3. Penalty for arm position away from center
        pos_penalty = np.cos(theta_0)

        # 4. Bonus for being close to upright position
        arm_center = np.exp(-1.0 * theta_0 ** 2)
        upright_closeness = np.exp(-10.0 * theta_1_norm ** 2)
        stability_factor = np.exp(-0.6 * theta_1_dot ** 2)
        bonus = 3.0 * upright_closeness * stability_factor * arm_center

        # 5. Penalty for being close to downward position
        downright_theta_1 = normalize_angle(theta_1)
        downright_closeness = np.exp(-10.0 * downright_theta_1 ** 2)
        bonus += -3.0 * downright_closeness * stability_factor

        # 6. Penalty for approaching arm angle limits
        theta_max_dist = np.clip(1.0 - abs(theta_0 - THETA_MAX) / 0.5, 0, 1)
        theta_min_dist = np.clip(1.0 - abs(theta_0 - THETA_MIN) / 0.5, 0, 1)
        limit_distance = max(theta_max_dist, theta_min_dist)
        limit_penalty = -10.0 * limit_distance ** 3

        # 7. Energy management reward
        JL = self.params['JL']
        mL = self.params['mL']
        l_1 = self.params['l_1']
        energy_reward = 2 - 0.15 * abs(
            mL * g * l_1 * (np.cos(theta_1_norm)) +
            0.5 * JL * theta_1_dot ** 2 -
            mL * g * l_1
        )

        # 8. Cost for high voltage usage
        max_voltage = self.params['max_voltage']
        # Using squared voltage to penalize higher voltages more
        upright_closeness = np.exp(-10.0 * theta_1_norm ** 2)
        stability_factor = np.exp(-0.1 * theta_1_dot ** 2)
        voltage_cost = 1.0 - 1.0 * (last_voltage / max_voltage) ** 2 * upright_closeness * stability_factor

        # Combine all components
        reward = (
                upright_reward +
                #velocity_penalty +
                #pos_penalty +
                bonus +
                limit_penalty +
                energy_reward
                #voltage_cost
        )

        return reward

    def get_current_parameters(self):
        """Get current parameter values."""
        return self.params


# ====== Actor Network ======
class Actor(nn.Module):
    """Policy network that outputs action distribution with modern improvements."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        # Improved network with SiLU activation and LayerNorm
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean and log std for continuous action
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

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


# ====== Critic Network ======
class Critic(nn.Module):
    """Value network that estimates Q-values with modern improvements."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        # Improved Q1 network with SiLU and LayerNorm
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network to reducing overestimation bias,
        # later in the code the min of these two networks
        # are chosen. Initially popularized in TD3.
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """Forward pass through both Q networks."""
        x = torch.cat([state, action], 1)

        q1 = self.q1(x)
        q2 = self.q2(x)

        return q1, q2


# ====== Replay Buffer ======
class ReplayBuffer:
    """Store and sample transitions for off-policy learning."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample a batch of transitions."""
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = map(np.array, zip(*[self.buffer[i] for i in batch]))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ====== SAC Agent ======
class SACAgent:
    """Soft Actor-Critic agent with modern training techniques."""

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            automatic_entropy_tuning=True,
            max_episodes=1000
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)

        # Copy parameters to target network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Improved optimizers with weight decay (AdamW)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)

        # Learning rate schedulers
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=max_episodes)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=max_episodes)

        # Automatic entropy tuning
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr, weight_decay=1e-4)
            self.alpha_scheduler = CosineAnnealingLR(self.alpha_optimizer, T_max=max_episodes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move networks to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        if automatic_entropy_tuning:
            self.log_alpha = self.log_alpha.to(self.device)

    def select_action(self, state, evaluate=False):
        """Select an action given a state."""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                # Use mean action (no exploration) for evaluation
                action, _ = self.actor(state)
            else:
                # Sample action for training (with exploration)
                action, _ = self.actor.sample(state)

            return action.cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        """Update actor and critic parameters using a batch of experiences."""
        # Sample batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)

        # === Update Critic ===
        with torch.no_grad():
            # Get next actions and log probs from current policy
            next_action, next_log_prob = self.actor.sample(next_state_batch)

            # Get target Q values
            target_q1, target_q2 = self.critic_target(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)

            # Apply entropy term
            if self.automatic_entropy_tuning:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            # Compute TD target
            target_q = target_q - alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q

        # Current Q estimates
        current_q1, current_q2 = self.critic(state_batch, action_batch)

        # Compute critic loss (MSE)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Apply gradient clipping to critic
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        self.critic_optimizer.step()

        # === Update Actor ===
        # Sample actions and log probs from current policy
        actions, log_probs = self.actor.sample(state_batch)

        # Get Q values for new actions
        q1, q2 = self.critic(state_batch, actions)
        min_q = torch.min(q1, q2)

        # Get current alpha
        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha

        # Actor loss (maximize Q - alpha * log_prob)
        actor_loss = (alpha * log_probs - min_q).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Apply gradient clipping to actor
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.actor_optimizer.step()

        # === Update Alpha (if automatic entropy tuning) ===
        if self.automatic_entropy_tuning:
            # Alpha loss
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            # Optimize alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # === Soft Update Target Networks ===
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha.item() if self.automatic_entropy_tuning else self.alpha
        }

    def scheduler_step(self):
        """Step all learning rate schedulers (call at end of episode)"""
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        if self.automatic_entropy_tuning:
            self.alpha_scheduler.step()

# ====== Training Function ======
def train(
        actor_path=None,
        critic_path=None,
        variable_dt=False,
        param_variation=0.1,
        fixed_params=False,
        voltage_range=None,
        max_episodes=500,
        eval_interval=10
):
    """Train the SAC agent on the pendulum environment."""
    print("Starting SAC training for inverted pendulum control...")
    print(f"Using variable time steps: {variable_dt}")
    print(f"Using parameter variation: {param_variation if not fixed_params else 'OFF'}")
    torch.manual_seed(0)
    np.random.seed(0)

    if voltage_range:
        print(f"Using voltage range: {voltage_range[0]} to {voltage_range[1]} V")
    else:
        print(f"Using base max voltage: {base_max_voltage} V with {param_variation * 100}% variation")

    # Environment setup
    env = PendulumEnv(
        variable_dt=variable_dt,
        param_variation=param_variation,
        fixed_params=fixed_params,
        voltage_range=voltage_range
    )

    # Setup parameters
    state_dim = 6  # Observation space dimension
    action_dim = 1  # Motor voltage (normalized)
    max_steps = 2000  # Max steps per episode
    batch_size = 256  # Increased batch size
    replay_buffer_size = 100000  # Buffer capacity
    updates_per_step = 1  # Updates per environment step

    # Initialize agent (load pre-trained models if provided)
    if actor_path or critic_path:
        agent = load_agent(actor_path, critic_path, state_dim, action_dim, max_episodes=max_episodes)
        print("Loaded pre-trained models")
    else:
        agent = SACAgent(state_dim, action_dim, max_episodes=max_episodes)
        print("Initialized new agent with improved architecture")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # Metrics tracking
    episode_rewards = []
    avg_rewards = []

    # Training loop
    start_time = time()

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        # Track losses for reporting
        critic_losses = []
        actor_losses = []
        alpha_values = []

        # Prepare for episode visualization if needed
        plot_this_episode = ((episode + 1) % eval_interval == 0) or (episode <= 1)
        episode_states = [] if plot_this_episode else None
        episode_actions = [] if plot_this_episode else None
        step_rewards = [] if plot_this_episode else None  # Track rewards per step

        # Episode loop
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Store state, action, and reward if plotting
            if plot_this_episode:
                episode_states.append(env.state.copy())
                episode_actions.append(action)
                step_rewards.append(reward)  # Store individual step rewards

            # Move to next state
            state = next_state
            episode_reward += reward

            # Update networks if enough samples
            if len(replay_buffer) > batch_size:
                for _ in range(updates_per_step):
                    update_info = agent.update_parameters(replay_buffer, batch_size)
                    critic_losses.append(update_info['critic_loss'])
                    actor_losses.append(update_info['actor_loss'])
                    alpha_values.append(update_info['alpha'])

            if done:
                break

        # Step the learning rate schedulers at the end of each episode
        agent.scheduler_step()

        # Log progress
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)

        # Calculate average losses for reporting
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
        avg_alpha = np.mean(alpha_values) if alpha_values else 0.0

        # Periodic reporting and visualization
        if (episode + 1) % eval_interval == 0 or episode <= 1:
            training_time = time() - start_time
            print(
                f"Episode {episode + 1}/{max_episodes} | Reward: {episode_reward:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | C_Loss: {avg_critic_loss:.3f} | "
                f"A_Loss: {avg_actor_loss:.3f} | Alpha: {avg_alpha:.3f} | Time: {training_time:.1f}s"
            )

            # Print sample parameter values
            current_params = env.get_current_parameters()
            print(
                f"  Parameters: Rm={current_params['Rm']:.4f}, Km={current_params['Km']:.6f}, "
                f"mL={current_params['mL']:.6f}, k={current_params['k']:.6f}, "
                f"max_voltage={current_params['max_voltage']:.2f}V"
            )

            # Plot simulation for visual progress tracking
            if plot_this_episode and episode_states and len(episode_states) > 0:
                # Make sure we have rewards to plot
                if step_rewards and len(step_rewards) > 0:
                    plot_training_episode(
                        episode,
                        episode_states,
                        episode_actions,
                        env.time_history if variable_dt else [env.dt] * len(episode_states),
                        episode_reward,
                        env.params['max_voltage'],
                        step_rewards  # Pass the recorded step rewards
                    )
                else:
                    # If no rewards recorded, call without rewards_history
                    plot_training_episode(
                        episode,
                        episode_states,
                        episode_actions,
                        env.time_history if variable_dt else [env.dt] * len(episode_states),
                        episode_reward,
                        env.params['max_voltage']
                    )

            # Plot parameter variation history
            if not fixed_params and episode > 10:
                env.param_manager.plot_parameter_history(f"param_variations_ep{episode + 1}.png")

            # Save model checkpoint
            if (episode + 1) % (eval_interval * 5) == 0:
                timestamp = int(time())
                torch.save(agent.actor.state_dict(), f"actor_ep{episode + 1}_{timestamp}.pth")

        # Early stopping if well trained
        if avg_reward > 10000 and episode > 500:
            print(f"Environment solved in {episode + 1} episodes!")
            break

    # Report training time
    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")

    # Save final model
    timestamp = int(time())
    torch.save(agent.actor.state_dict(), f"actor_final_{timestamp}.pth")
    torch.save(agent.critic.state_dict(), f"critic_final_{timestamp}.pth")

    # Plot final training curve
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('SAC Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_progress.png")
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


# ====== Load Agent Helper ======
def load_agent(actor_path=None, critic_path=None, state_dim=6, action_dim=1, hidden_dim=256, max_episodes=500):
    """Load pre-trained actor and critic models."""
    # Initialize a new agent with improved architecture
    agent = SACAgent(state_dim, action_dim, hidden_dim, max_episodes=max_episodes)

    # Load actor if path is provided
    if actor_path:
        agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
        print(f"Actor model loaded from {actor_path}")

    # Load critic if path is provided
    if critic_path:
        agent.critic.load_state_dict(torch.load(critic_path, map_location=agent.device))
        # Update target critic as well
        agent.critic_target.load_state_dict(torch.load(critic_path, map_location=agent.device))
        print(f"Critic model loaded from {critic_path}")

    return agent


# ====== Visualization Helper ======
def plot_training_episode(episode, states_history, actions_history, dt_history, episode_reward,
                          max_voltage=base_max_voltage, rewards_history=None, is_eval=False,
                          save_path=None):
    """Plot the pendulum state evolution for a training episode."""
    # Convert history to numpy arrays
    states_history = np.array(states_history)
    actions_history = np.array(actions_history)

    # Always use 5 subplots
    num_plots = 5

    # Extract components
    thetas = states_history[:, 0]  # theta_0 (arm angle)
    alphas = states_history[:, 1]  # theta_1 (pendulum angle)
    theta_dots = states_history[:, 2]  # theta_0_dot
    alpha_dots = states_history[:, 3]  # theta_1_dot

    # Normalize alpha for visualization
    alpha_normalized = np.array([normalize_angle(a + np.pi) for a in alphas])

    # Generate time array
    if isinstance(dt_history, list) and len(dt_history) > 0:
        # Create cumulative time array for variable dt
        t = np.zeros(len(dt_history))
        t[0] = dt_history[0]
        for i in range(1, len(dt_history)):
            t[i] = t[i - 1] + dt_history[i]
    else:
        # Use fixed dt
        t = np.arange(len(states_history)) * dt_history[0]

    # Calculate performance metrics
    balanced_time = 0.0
    num_upright_points = 0
    upright_threshold = 0.17  # about 10 degrees

    for i in range(len(alpha_normalized)):
        if abs(alpha_normalized[i]) < upright_threshold:
            dt_value = dt_history[i] if i < len(dt_history) else dt_history[0]
            balanced_time += dt_value
            num_upright_points += 1

    # Calculate voltage values
    voltage_values = actions_history * max_voltage

    # Plot results
    plt.figure(figsize=(12, 15))

    # 1. Plot arm angle
    plt.subplot(num_plots, 1, 1)
    plt.plot(t, thetas, 'b-')
    plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.7, label='Limits')
    plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.7)
    plt.ylabel('Arm angle (rad)')
    plt.title(
        f'Episode {episode + 1} | Reward: {episode_reward:.2f} | Balanced: {balanced_time:.2f}s | Max Voltage: {max_voltage:.2f}V')
    plt.legend()
    plt.grid(True)

    # 2. Plot pendulum angle
    plt.subplot(num_plots, 1, 2)
    plt.scatter(t, alpha_normalized, color='g', s=10)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Upright position
    plt.axhline(y=upright_threshold, color='r', linestyle=':', alpha=0.5)  # Threshold
    plt.axhline(y=-upright_threshold, color='r', linestyle=':', alpha=0.5)
    plt.ylabel('Pendulum angle (rad)')
    plt.grid(True)

    # 3. Plot control actions/voltage
    plt.subplot(num_plots, 1, 3)
    plt.plot(t, voltage_values, 'r-')  # Use current max_voltage value
    plt.ylabel('Control voltage (V)')
    plt.ylim([-max_voltage * 1.1, max_voltage * 1.1])
    plt.grid(True)

    # 4. Plot angular velocities
    plt.subplot(num_plots, 1, 4)
    plt.plot(t, theta_dots, 'b-', label='Arm velocity')
    plt.plot(t, alpha_dots, 'g-', label='Pendulum velocity')
    plt.ylabel('Angular velocity (rad/s)')
    plt.legend()
    plt.grid(True)

    # 5. Plot rewards with dual y-axis
    ax1 = plt.subplot(num_plots, 1, 5)
    if rewards_history is not None and len(rewards_history) > 0:
        # Make sure we don't try to plot more rewards than we have time points
        plot_length = min(len(t), len(rewards_history))

        # Step rewards on primary y-axis
        ax1.plot(t[:plot_length], rewards_history[:plot_length], 'purple', label='Step reward')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Step Reward', color='purple')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax1.grid(True)

        # Cumulative rewards on secondary y-axis
        ax2 = ax1.twinx()
        cumulative_rewards = np.cumsum(rewards_history[:plot_length])
        if len(cumulative_rewards) > 0:
            ax2.plot(t[:plot_length], cumulative_rewards, 'g--', linewidth=1.5, alpha=0.8,
                     label='Cumulative reward')
            ax2.set_ylabel('Cumulative Reward', color='green')
            ax2.tick_params(axis='y', labelcolor='green')

            # Add cumulative episode reward to the last point
            if len(cumulative_rewards) > 0:
                ax2.annotate(f'{cumulative_rewards[-1]:.1f}',
                             xy=(t[plot_length - 1], cumulative_rewards[-1]),
                             xytext=(0, 5), textcoords='offset points',
                             ha='center', va='bottom', color='green')

        # Create a combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig(f"{'eval' if is_eval else 'training'}_episode_{episode + 1}.png")

    plt.close()


# ====== Evaluation Function ======
def evaluate(agent, num_episodes=5, render=True, variable_dt=False, param_variation=0.1, fixed_params=False,
             voltage_range=None):
    """Evaluate the trained agent's performance."""
    # Create environment with specified parameters
    env = PendulumEnv(
        variable_dt=variable_dt,
        param_variation=param_variation,
        fixed_params=fixed_params,
        voltage_range=voltage_range
    )

    # Performance tracking
    all_rewards = []
    all_balance_times = []
    all_params = []

    for episode in range(num_episodes):
        state = env.reset(random_init=False)  # Start from standard position
        total_reward = 0

        # Store parameters
        current_params = env.get_current_parameters()
        all_params.append(current_params)

        # Data collection for visualization
        states_history = []
        actions_history = []
        step_rewards = []  # Track per-step rewards for visualization

        for step in range(env.max_steps):
            # Select action without exploration
            action = agent.select_action(state, evaluate=True)

            # Perform action
            next_state, reward, done, _ = env.step(action)

            # Record data
            states_history.append(env.state.copy())
            actions_history.append(action)
            step_rewards.append(reward)  # Store individual step rewards

            total_reward += reward
            state = next_state

            if done:
                break

        all_rewards.append(total_reward)

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

        # Print parameter values
        if not fixed_params:
            print(f"  Parameters: Rm={current_params['Rm']:.4f}, Km={current_params['Km']:.6f}, "
                  f"mL={current_params['mL']:.6f}, k={current_params['k']:.6f}, "
                  f"max_voltage={current_params['max_voltage']:.2f}V")

        if render:
            # Calculate evaluation metrics and create visualizations
            states_history = np.array(states_history)
            actions_history = np.array(actions_history)

            # Extract components
            thetas = states_history[:, 0]
            alphas = states_history[:, 1]
            theta_dots = states_history[:, 2]
            alpha_dots = states_history[:, 3]

            # Normalize angles
            alpha_normalized = np.array([normalize_angle(a + np.pi) for a in alphas])

            # Generate time array
            if variable_dt:
                t = np.zeros(len(env.time_history))
                t[0] = env.time_history[0]
                for i in range(1, len(env.time_history)):
                    t[i] = t[i - 1] + env.time_history[i]
            else:
                t = np.arange(len(states_history)) * env.dt

            # Calculate balance metrics
            balanced_time = 0.0
            num_upright_points = 0
            upright_threshold = 0.17  # about 10 degrees

            for i in range(len(alpha_normalized)):
                if abs(alpha_normalized[i]) < upright_threshold:
                    dt_value = env.time_history[i] if variable_dt and i < len(env.time_history) else env.dt
                    balanced_time += dt_value
                    num_upright_points += 1

            all_balance_times.append(balanced_time)

            # Calculate voltage values
            voltage_values = actions_history * current_params['max_voltage']

            # Create evaluation plot - same structure as training plots
            plt.figure(figsize=(12, 15))

            # 1. Plot arm angle
            plt.subplot(5, 1, 1)
            plt.plot(t, thetas, 'b-')
            plt.axhline(y=THETA_MAX, color='r', linestyle='--', alpha=0.7, label='Limits')
            plt.axhline(y=THETA_MIN, color='r', linestyle='--', alpha=0.7)
            plt.ylabel('Arm angle (rad)')

            title = f'Evaluation Episode {episode + 1}'
            if variable_dt:
                title += " (Variable dt)"
            if not fixed_params:
                title += f" (Param var: {param_variation * 100:.0f}%)"
            if voltage_range:
                title += f" (Voltage: {current_params['max_voltage']:.2f}V)"
            plt.title(title)
            plt.legend()
            plt.grid(True)

            # 2. Plot pendulum angle
            plt.subplot(5, 1, 2)
            plt.scatter(t, alpha_normalized, color='g', s=10)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Upright position
            plt.axhline(y=upright_threshold, color='r', linestyle=':', alpha=0.5)
            plt.axhline(y=-upright_threshold, color='r', linestyle=':', alpha=0.5)
            plt.ylabel('Pendulum angle (rad)')
            plt.grid(True)

            # 3. Plot motor voltage
            plt.subplot(5, 1, 3)
            plt.plot(t, voltage_values, 'r-')
            plt.ylabel('Control voltage (V)')
            plt.ylim([-current_params['max_voltage'] * 1.1, current_params['max_voltage'] * 1.1])
            plt.grid(True)

            # 4. Plot angular velocities
            plt.subplot(5, 1, 4)
            plt.plot(t, theta_dots, 'b-', label='Arm velocity')
            plt.plot(t, alpha_dots, 'g-', label='Pendulum velocity')
            plt.ylabel('Angular velocity (rad/s)')
            plt.legend()
            plt.grid(True)

            # 5. Plot rewards with dual y-axis
            ax1 = plt.subplot(5, 1, 5)
            # Step rewards on primary y-axis
            ax1.plot(t[:len(step_rewards)], step_rewards, 'purple', label='Step reward')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Step Reward', color='purple')
            ax1.tick_params(axis='y', labelcolor='purple')
            ax1.grid(True)

            # Cumulative rewards on secondary y-axis
            ax2 = ax1.twinx()
            cumulative_rewards = np.cumsum(step_rewards)
            if len(cumulative_rewards) > 0:
                ax2.plot(t[:len(step_rewards)], cumulative_rewards, 'g--', linewidth=1.5, alpha=0.8,
                         label='Cumulative reward')
                ax2.set_ylabel('Cumulative Reward', color='green')
                ax2.tick_params(axis='y', labelcolor='green')

                # Add cumulative episode reward to the last point
                if len(cumulative_rewards) > 0:
                    ax2.annotate(f'{cumulative_rewards[-1]:.1f}',
                                 xy=(t[len(step_rewards) - 1], cumulative_rewards[-1]),
                                 xytext=(0, 5), textcoords='offset points',
                                 ha='center', va='bottom', color='green')

            # Create a combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.tight_layout()
            plt.savefig(f"evaluation_episode_{episode + 1}.png")
            plt.close()

            # Print performance metrics
            print(f"  Time balanced: {balanced_time:.2f} seconds")
            print(f"  Upright points: {num_upright_points}")
            print(f"  Max arm angle: {np.max(np.abs(thetas)):.2f} rad")
            print(f"  Max pendulum velocity: {np.max(np.abs(alpha_dots)):.2f} rad/s")
            final_angle_deg = abs(alpha_normalized[-1]) * 180 / np.pi
            print(f"  Final angle from vertical: {abs(alpha_normalized[-1]):.2f} rad ({final_angle_deg:.1f}°)")
            print("-" * 50)

    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"Average reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    if all_balance_times:
        print(f"Average balanced time: {np.mean(all_balance_times):.2f} ± {np.std(all_balance_times):.2f} seconds")

    # Parameter analysis
    if not fixed_params and len(all_params) > 1:
        print("\n===== Parameter Variation Analysis =====")
        # Important parameters to analyze
        param_keys = ['Rm', 'Km', 'mL', 'LA', 'LL', 'JA', 'JL', 'k', 'max_voltage']  # Added max_voltage

        # Create correlation analysis
        if len(all_rewards) > 3:
            plt.figure(figsize=(15, 10))
            for i, param in enumerate(param_keys):
                values = [p[param] for p in all_params]

                # Only analyze if parameter actually varied
                if len(set(values)) > 1:
                    corr = np.corrcoef(values, all_rewards)[0, 1]
                    plt.subplot(len(param_keys), 1, i + 1)
                    plt.scatter(values, all_rewards)
                    plt.xlabel(f'{param} value')
                    plt.ylabel('Reward')
                    plt.title(f'Correlation between {param} and reward: {corr:.3f}')
                    plt.grid(True)

            plt.tight_layout()
            plt.savefig("parameter_reward_correlation.png")
            plt.close()


# ====== Main Execution ======
if __name__ == "__main__":
    print("Inverted Pendulum Control with Soft Actor-Critic")
    print("=" * 60)

    agent = train(
        # actor_path="actor_final_1743528264.pth",
        # critic_path="critic_final_1743528264.pth",
        variable_dt=True,
        param_variation=0.2,
        # fixed_params=True,
        voltage_range=(4.0, 18.0),
        max_episodes=1000,
        eval_interval=10
    )

    # Evaluate trained agent with different parameter variations
    print("\n--- Standard Evaluation ---")
    evaluate(agent, num_episodes=3, variable_dt=True, param_variation=0.0)

    print("\n--- Robustness Test (High Variation) ---")
    evaluate(agent, num_episodes=3, variable_dt=True, param_variation=0.25)

    print("\n--- Robustness Test (High Variation) ---")
    evaluate(agent, num_episodes=3, variable_dt=True, param_variation=0.35)

    print("=" * 60)
    print("PROGRAM EXECUTION COMPLETE")
    print("=" * 60)