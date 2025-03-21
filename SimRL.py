import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba as nb
from time import time
from dataclasses import dataclass
from collections import deque

# Import the necessary constants and functions from your existing code
# System Parameters (same as in your original code)
Rm = 8.94  # Motor resistance (Ohm)
Km = 0.0431  # Motor back-emf constant
Jm = 6e-5  # Total moment of inertia acting on motor shaft (kg·m^2)
bm = 3e-4  # Viscous damping coefficient (Nm/rad/s)
DA = 3e-4  # Damping coefficient of pendulum arm (Nm/rad/s)
DL = 5e-4  # Damping coefficient of pendulum link (Nm/rad/s)
mA = 0.053  # Weight of pendulum arm (kg)
mL = 0.024  # Weight of pendulum link (kg)
LA = 0.086  # Length of pendulum arm (m)
LL = 0.128  # Length of pendulum link (m)
JA = 5.72e-5  # Inertia moment of pendulum arm (kg·m^2)
JL = 1.31e-4  # Inertia moment of pendulum link (kg·m^2)
g = 9.81  # Gravity constant (m/s^2)

max_voltage = 10.0  # Maximum motor voltage
THETA_MIN = -2.2  # Minimum arm angle (radians)
THETA_MAX = 2.2  # Maximum arm angle (radians)

# Pre-compute constants for optimization (same as in your original code)
half_mL_LL_g = 0.5 * mL * LL * g
half_mL_LL_LA = 0.5 * mL * LL * LA
quarter_mL_LL_squared = 0.25 * mL * LL ** 2
Mp_g_Lp = mL * g * LL
Jp = (1 / 3) * mL * LL ** 2  # Pendulum moment of inertia (kg·m²)

batch_size = 256 * 32  # Batch size for training


# Reuse some of your helper functions
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
    """Ultra-optimized dynamics calculation with theta limits"""
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


@dataclass
class SystemParameters:
    """Dataclass to store the physical parameters of the system with randomization capabilities"""
    # Base values
    Rm: float = 8.94  # Motor resistance (Ohm)
    Km: float = 0.0431  # Motor back-emf constant
    Jm: float = 6e-5  # Total moment of inertia acting on motor shaft (kg·m^2)
    bm: float = 3e-4  # Viscous damping coefficient (Nm/rad/s)
    DA: float = 3e-4  # Damping coefficient of pendulum arm (Nm/rad/s)
    DL: float = 5e-4  # Damping coefficient of pendulum link (Nm/rad/s)
    mA: float = 0.053  # Weight of pendulum arm (kg)
    mL: float = 0.024  # Weight of pendulum link (kg)
    LA: float = 0.086  # Length of pendulum arm (m)
    LL: float = 0.128  # Length of pendulum link (m)
    JA: float = 5.72e-5  # Inertia moment of pendulum arm (kg·m^2)
    JL: float = 1.31e-4  # Inertia moment of pendulum link (kg·m^2)
    g: float = 9.81  # Gravity constant (m/s^2)

    # Derived parameters (will be computed when randomized)
    half_mL_LL_g: float = 0.0
    half_mL_LL_LA: float = 0.0
    quarter_mL_LL_squared: float = 0.0
    Mp_g_Lp: float = 0.0
    Jp: float = 0.0

    def randomize(self, randomization_factor=0.1):
        """
        Randomize the physical parameters within a given factor range.

        Args:
            randomization_factor (float): Maximum percentage variation (0.1 = ±10%)

        Returns:
            SystemParameters: Self with randomized values
        """

        # Function to randomize a parameter within the given factor
        def rand_param(value):
            return value * (1.0 + np.random.uniform(-randomization_factor, randomization_factor))

        # Randomize each parameter
        self.Rm = rand_param(8.94)
        self.Km = rand_param(0.0431)
        self.Jm = rand_param(6e-5)
        self.bm = rand_param(3e-4)
        self.DA = rand_param(3e-4)
        self.DL = rand_param(5e-4)
        self.mA = rand_param(0.053)
        self.mL = rand_param(0.024)
        self.LA = rand_param(0.086)
        self.LL = rand_param(0.128)
        self.JA = rand_param(5.72e-5)
        self.JL = rand_param(1.31e-4)

        # Optional: Randomize gravity slightly (simulates different mounting angles)
        self.g = rand_param(9.81)

        # Update derived parameters
        self.update_derived_parameters()

        return self

    def update_derived_parameters(self):
        """Update the derived parameters based on the current physical parameters"""
        self.half_mL_LL_g = 0.5 * self.mL * self.LL * self.g
        self.half_mL_LL_LA = 0.5 * self.mL * self.LL * self.LA
        self.quarter_mL_LL_squared = 0.25 * self.mL * self.LL ** 2
        self.Mp_g_Lp = self.mL * self.g * self.LL
        self.Jp = (1 / 3) * self.mL * self.LL ** 2  # Pendulum moment of inertia


# Modify the PendulumEnv class to use randomized parameters
class PendulumEnv:
    def __init__(self, dt=0.01, max_steps=1000, delay_steps=5, delay_range=None, randomization_factor=0.1):
        self.dt = dt
        self.max_steps = max_steps
        self.step_count = 0
        self.state = None

        # Delay mechanism with variable delay support
        self.min_delay = delay_steps
        self.max_delay = delay_steps

        # If delay_range is provided, use it to set min and max delay
        if delay_range is not None:
            if isinstance(delay_range, (list, tuple)) and len(delay_range) == 2:
                self.min_delay = max(0, delay_range[0])  # Ensure non-negative
                self.max_delay = max(self.min_delay, delay_range[1])  # Ensure max >= min

        # Current episode's delay will be set during reset
        self.delay_steps = delay_steps
        self.max_buffer_size = self.max_delay + 1  # Buffer size based on maximum possible delay
        self.observation_buffer = deque(maxlen=self.max_buffer_size)

        # System parameters with randomization
        self.randomization_factor = randomization_factor
        self.params = SystemParameters()

        # External limits
        self.max_voltage = 10.0
        self.THETA_MIN = -2.2
        self.THETA_MAX = 2.2

        # For tracking the current episode's delay
        self.current_delay = self.delay_steps

    def reset(self, random_init=True):
        # Randomize system parameters for this episode
        self.params.randomize(self.randomization_factor)

        # Randomize the delay if using variable delay
        if self.min_delay != self.max_delay:
            self.current_delay = np.random.randint(self.min_delay, self.max_delay + 1)
        else:
            self.current_delay = self.delay_steps

        # Debug info about current delay
        # print(f"Episode using delay of {self.current_delay} steps")

        # Initialize state with small random variations if requested
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

        # Clear and initialize the observation buffer
        observation = self._get_observation_from_state(self.state)
        self.observation_buffer.clear()
        for _ in range(self.max_buffer_size):
            self.observation_buffer.append(observation)

        # Return the delayed observation
        return self._get_observation()

    def _get_observation_from_state(self, state):
        # Same as before
        theta, alpha, theta_dot, alpha_dot = state
        alpha_norm = normalize_angle(alpha + np.pi)

        obs = np.array([
            np.sin(theta), np.cos(theta),
            np.sin(alpha_norm), np.cos(alpha_norm),
            theta_dot / 10.0,
            alpha_dot / 10.0
        ])

        return obs

    def _get_observation(self):
        # Return observation with the current episode's delay
        if len(self.observation_buffer) <= self.current_delay:
            return self.observation_buffer[0]  # Fallback if buffer isn't filled yet
        return self.observation_buffer[-(self.current_delay + 1)]  # Get appropriately delayed observation

    def step(self, action):
        # Convert normalized action [-1, 1] to voltage
        voltage = float(action) * self.max_voltage

        # RK4 integration step with randomized parameters
        self.state = self._rk4_step(self.state, voltage)

        # Enforce limits
        self.state = self._enforce_theta_limits(self.state)

        # Add current observation to buffer
        current_obs = self._get_observation_from_state(self.state)
        self.observation_buffer.append(current_obs)

        # Get reward based on true state
        reward = self._compute_reward()

        # Check if done
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_observation(), reward, done, {}

    def _enforce_theta_limits(self, state):
        """Enforce hard limits on theta angle and velocity"""
        theta, alpha, theta_dot, alpha_dot = state

        # Apply hard limit on theta
        if theta > self.THETA_MAX:
            theta = self.THETA_MAX
            # If hitting upper limit with positive velocity, stop the motion
            if theta_dot > 0:
                theta_dot = 0.0
        elif theta < self.THETA_MIN:
            theta = self.THETA_MIN
            # If hitting lower limit with negative velocity, stop the motion
            if theta_dot < 0:
                theta_dot = 0.0

        return np.array([theta, alpha, theta_dot, alpha_dot])

    def _dynamics_step(self, state, t, vm):
        """Ultra-optimized dynamics calculation with theta limits and randomized parameters"""
        theta_m, theta_L, theta_m_dot, theta_L_dot = state[0], state[1], state[2], state[3]
        p = self.params  # Get the current randomized parameters

        # Check theta limits - implement hard stops
        if (theta_m >= self.THETA_MAX and theta_m_dot > 0) or (theta_m <= self.THETA_MIN and theta_m_dot < 0):
            theta_m_dot = 0.0  # Stop the arm motion at the limits

        # Apply dead zone and calculate motor torque
        if -0.2 <= vm <= 0.2:
            vm = 0.0

        # Motor torque calculation with randomized params
        im = (vm - p.Km * theta_m_dot) / p.Rm
        Tm = p.Km * im

        # Equations of motion coefficients from Eq. (9) in paper
        # For theta_m equation with randomized parameters:
        M11 = p.mL * p.LA ** 2 + p.quarter_mL_LL_squared - p.quarter_mL_LL_squared * np.cos(theta_L) ** 2 + p.JA
        M12 = -p.half_mL_LL_LA * np.cos(theta_L)
        C1 = 0.5 * p.mL * p.LL ** 2 * np.sin(theta_L) * np.cos(theta_L) * theta_m_dot * theta_L_dot
        C2 = p.half_mL_LL_LA * np.sin(theta_L) * theta_L_dot ** 2

        # For theta_L equation with randomized parameters:
        M21 = p.half_mL_LL_LA * np.cos(theta_L)
        M22 = p.JL + p.quarter_mL_LL_squared
        C3 = -p.quarter_mL_LL_squared * np.cos(theta_L) * np.sin(theta_L) * theta_m_dot ** 2
        G = p.half_mL_LL_g * np.sin(theta_L)

        # Calculate determinant for matrix inversion
        det_M = M11 * M22 - M12 * M21

        # Handle near-singular matrix
        if abs(det_M) < 1e-10:
            theta_m_ddot = 0
            theta_L_ddot = 0
        else:
            # Right-hand side of equations
            RHS1 = Tm - C1 - C2 - p.DA * theta_m_dot
            RHS2 = -G - p.DL * theta_L_dot - C3

            # Solve for accelerations
            theta_m_ddot = (M22 * RHS1 - M12 * RHS2) / det_M
            theta_L_ddot = (-M21 * RHS1 + M11 * RHS2) / det_M

        return np.array([theta_m_dot, theta_L_dot, theta_m_ddot, theta_L_ddot])

    def _rk4_step(self, state, vm):
        """4th-order Runge-Kutta integrator step"""
        # Apply limits to initial state
        state = self._enforce_theta_limits(state)

        k1 = self._dynamics_step(state, 0, vm)
        state_k2 = self._enforce_theta_limits(state + 0.5 * self.dt * k1)
        k2 = self._dynamics_step(state_k2, 0, vm)

        state_k3 = self._enforce_theta_limits(state + 0.5 * self.dt * k2)
        k3 = self._dynamics_step(state_k3, 0, vm)

        state_k4 = self._enforce_theta_limits(state + self.dt * k3)
        k4 = self._dynamics_step(state_k4, 0, vm)

        new_state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self._enforce_theta_limits(new_state)

    def _compute_reward(self):
        theta, alpha, theta_dot, alpha_dot = self.state
        alpha_norm = normalize_angle(alpha + np.pi)  # Normalized for upright position
        p = self.params

        # Component 1: Reward for pendulum being close to upright
        upright_reward = np.cos(alpha_norm)  # 1 when upright, -1 when downward

        # Component 2: Penalty for high velocities
        velocity_penalty = -0.01 * (theta_dot ** 2 + alpha_dot ** 2)

        # Component 3: Penalty for arm position away from center
        pos_penalty = -0.01 * theta ** 2

        # Component 4: Extra reward for being very close to upright and stable
        bonus = 0.0
        if abs(alpha_norm) < 0.2 and abs(alpha_dot) < 1.0:
            bonus = 5.0

        # Component 5: Penalty for hitting limits
        limit_penalty = 0.0
        if (abs(theta - self.THETA_MAX) < 0.02 or abs(theta - self.THETA_MIN) < 0.02):
            limit_penalty = -10.0

        # Component 6: Energy management reward
        E = p.Mp_g_Lp * (1 - np.cos(alpha)) + 0.5 * p.Jp * alpha_dot ** 2  # Current energy
        E_ref = p.Mp_g_Lp  # Energy at upright position (target energy)
        E_diff = abs(E - E_ref)  # Difference from optimal energy
        # Reward for being close to the optimal energy (inverted Gaussian)
        energy_reward = 2.0 * np.exp(-0.5 * (E_diff / (0.2 * E_ref)) ** 2)

        return upright_reward + bonus + limit_penalty + energy_reward


# Actor (Policy) Network
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

    def sample(self, state):
        mean, log_std = self.forward(state)
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


# Critic (Value) Network
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


# Soft Actor-Critic (SAC) Agent
class SACAgent:
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
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
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)

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

    def update_parameters(self, memory, batch_size=batch_size):
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
            next_action, next_log_prob = self.actor.sample(next_state_batch)

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
        actions, log_probs = self.actor.sample(state_batch)
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


# Replay Buffer
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



def train(randomization_factor=0.1, delay_steps=5, delay_range=None):
    """
    Train a SAC agent for the pendulum control task.

    Args:
        randomization_factor (float): Amount of parameter randomization
        delay_steps (int): Base number of delay steps
        delay_range (tuple, optional): Range of possible delays (min, max)

    Returns:
        SACAgent: The trained agent
    """
    # Create delay string for display
    if delay_range:
        delay_str = f"{delay_range[0]}-{delay_range[1]}"
    else:
        delay_str = str(delay_steps)

    print(f"Starting SAC training with domain randomization (factor: {randomization_factor}) "
          f"and input delay ({delay_str} steps)...")

    # Environment setup with randomization and variable delay
    env = PendulumEnv(randomization_factor=randomization_factor,
                      delay_steps=delay_steps,
                      delay_range=delay_range)

    # Hyperparameters
    state_dim = 6  # Our observation space
    action_dim = 1  # Motor voltage (normalized)
    max_episodes = 1500  # Increase episodes for more robust learning with randomization
    max_steps = 1000
    batch_size = 256 * 32
    replay_buffer_size = 200000  # Larger buffer for more diverse experiences
    updates_per_step = 1

    # Initialize agent
    agent = SACAgent(state_dim, action_dim)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # Metrics
    episode_rewards = []
    avg_rewards = []

    # Training loop
    start_time = time()

    for episode in range(max_episodes):
        state = env.reset()  # This will randomize parameters for each episode
        episode_reward = 0

        # If we're going to plot this episode, prepare to collect state data
        plot_this_episode = (episode + 1) % 10 == 0
        if plot_this_episode:
            episode_states = []
            episode_actions = []
            # Store the randomized parameters for this episode
            episode_params = {
                'Rm': env.params.Rm,
                'Km': env.params.Km,
                'mA': env.params.mA,
                'mL': env.params.mL,
                'LA': env.params.LA,
                'LL': env.params.LL,
                'g': env.params.g,
            }

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
                    agent.update_parameters(replay_buffer, batch_size)

            if done:
                break

        # Log progress
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{max_episodes} | Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f}")

            # Plot simulation for visual progress tracking
            if plot_this_episode:
                plot_training_episode(episode, episode_states, episode_actions, env.dt,
                                      episode_reward, episode_params, delay_steps, env)

        # Early stopping if well trained
        if avg_reward > 5000 and episode > 200:  # Higher threshold for robust policy
            print(f"Environment solved in {episode + 1} episodes!")
            break

    training_time = time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")

    # Save trained model
    torch.save(agent.actor.state_dict(), f"sac_pendulum_actor_random{int(randomization_factor*100)}_delay{delay_steps}.pth")
    torch.save(agent.critic.state_dict(), f"sac_pendulum_critic_random{int(randomization_factor*100)}_delay{delay_steps}.pth")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='100-Episode Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'SAC Training with Domain Randomization ({randomization_factor*100}% variation) and {delay_steps*env.dt*1000}ms delay')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"sac_training_random{int(randomization_factor*100)}_delay{delay_steps}.png")
    #plt.show()

    return agent


# Update plot_training_episode to show randomized parameters
def plot_training_episode(episode, states_history, actions_history, dt, episode_reward, params=None, delay_steps=None,
                          env=None):
    """
    Plot the pendulum state evolution for a training episode with randomized parameters and delay info

    Args:
        episode (int): Episode number
        states_history (numpy.ndarray): Array of state histories
        actions_history (numpy.ndarray): Array of actions
        dt (float): Timestep size
        episode_reward (float): Total episode reward
        params (dict, optional): Randomized physical parameters if available
        delay_steps (int, optional): Fixed delay steps (if not using variable delay)
        env (PendulumEnv, optional): Environment object to extract current delay info
    """
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

    # Generate time array
    t = np.arange(len(states_history)) * dt

    # Count performance metrics
    balanced_time = 0.0
    num_upright_points = 0

    for i in range(len(alpha_normalized)):
        # Check if pendulum is close to upright
        if abs(alpha_normalized[i]) < 0.17:  # about 10 degrees
            balanced_time += dt
            num_upright_points += 1

    # Plot results
    plt.figure(figsize=(12, 10))  # Make it a bit taller for parameter display

    # Get delay information
    delay_info = ""
    if env is not None and hasattr(env, 'current_delay'):
        delay_info = f"Delay: {env.current_delay} steps ({env.current_delay * dt * 1000:.1f}ms), "
    elif delay_steps is not None:
        delay_info = f"Delay: {delay_steps} steps ({delay_steps * dt * 1000:.1f}ms), "

    # Plot arm angle
    plt.subplot(3, 1, 1)
    plt.plot(t, thetas, 'b-')
    plt.axhline(y=2.2, color='r', linestyle='--', alpha=0.7, label='Theta Limits')
    plt.axhline(y=-2.2, color='r', linestyle='--', alpha=0.7)
    plt.ylabel('Arm angle (rad)')

    # Add randomized parameters to title if provided
    if params:
        param_text = f"{delay_info}Rm={params['Rm']:.2f}, Km={params['Km']:.4f}, mL={params['mL']:.4f}, LL={params['LL']:.4f}"
        plt.title(
            f'Training Episode {episode} | Reward: {episode_reward:.2f} | Balanced: {balanced_time:.2f}s\n{param_text}')
    else:
        plt.title(
            f'Training Episode {episode} | Reward: {episode_reward:.2f} | Balanced: {balanced_time:.2f}s\n{delay_info}')

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
    plt.plot(t, actions_history * 10.0, 'r-')  # Scale back to actual voltage
    plt.xlabel('Time (s)')
    plt.ylabel('Control voltage (V)')
    plt.ylim([-11, 11])
    plt.grid(True)

    plt.tight_layout()

    # Include randomization info in filename if available
    if params:
        plt.savefig(f"training_episode_{episode}_random.png")
    else:
        plt.savefig(f"training_episode_{episode}.png")


# Function to evaluate across a range of randomized environments
def evaluate_robustness(agent, num_episodes=10, randomization_levels=[0.0, 0.05, 0.1, 0.2],
                        delay_steps=5, delay_range=None):
    """
    Evaluate the trained agent's performance across different randomization levels.

    Args:
        agent (SACAgent): The agent to evaluate
        num_episodes (int): Number of episodes to evaluate per condition
        randomization_levels (list): List of randomization factors to test
        delay_steps (int): Base number of delay steps
        delay_range (tuple, optional): Range of possible delays (min, max)

    Returns:
        dict: Results dictionary with performance metrics
    """
    # Create delay string for display
    if delay_range:
        delay_str = f"{delay_range[0]}-{delay_range[1]}"
    else:
        delay_str = str(delay_steps)

    results = {}

    for rand_level in randomization_levels:
        print(f"\nEvaluating with {rand_level * 100}% parameter randomization and {delay_str} step delay...")

        # Create environment with current randomization level and variable delay
        env = PendulumEnv(randomization_factor=rand_level,
                          delay_steps=delay_steps,
                          delay_range=delay_range)

        episode_rewards = []
        balanced_times = []

        for episode in range(num_episodes):
            state = env.reset(random_init=True)  # Random init to test robustness
            total_reward = 0
            balanced_time = 0

            states_history = []

            for step in range(env.max_steps):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = env.step(action)

                # Track balancing performance
                states_history.append(env.state.copy())

                # Count balanced frames
                alpha_norm = normalize_angle(env.state[1] + np.pi)
                if abs(alpha_norm) < 0.17:  # ~10 degrees
                    balanced_time += env.dt

                total_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(total_reward)
            balanced_times.append(balanced_time)

            print(f"  Episode {episode + 1}: Reward = {total_reward:.2f}, Balanced = {balanced_time:.2f}s")

            # Plot the last episode for each randomization level
            if episode == num_episodes - 1:
                states_history = np.array(states_history)
                alphas = states_history[:, 1]
                alpha_normalized = np.zeros(len(alphas))
                for i in range(len(alphas)):
                    alpha_normalized[i] = normalize_angle(alphas[i] + np.pi)

                plt.figure(figsize=(10, 6))
                t = np.arange(len(states_history)) * env.dt
                plt.plot(t, alpha_normalized, label=f'{rand_level * 100}% randomization')
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                plt.axhline(y=0.17, color='r', linestyle=':', alpha=0.3)
                plt.axhline(y=-0.17, color='r', linestyle=':', alpha=0.3)
                plt.title(f'Pendulum Angle with {rand_level * 100}% Parameter Randomization')
                plt.xlabel('Time (s)')
                plt.ylabel('Pendulum angle (rad)')
                plt.grid(True)
                plt.legend()
                plt.savefig(f"robustness_test_{int(rand_level * 100)}pct.png")
                plt.close()

        # Store results
        results[rand_level] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_balanced_time': np.mean(balanced_times),
            'std_balanced_time': np.std(balanced_times)
        }

    # Plot summary of robustness results
    plt.figure(figsize=(12, 6))

    # Convert randomization levels to percentages for plotting
    x_labels = [f"{r * 100}%" for r in randomization_levels]

    # Plot rewards
    plt.subplot(1, 2, 1)
    means = [results[r]['mean_reward'] for r in randomization_levels]
    stds = [results[r]['std_reward'] for r in randomization_levels]
    plt.bar(x_labels, means, yerr=stds, capsize=10)
    plt.title('Reward vs. Parameter Randomization')
    plt.xlabel('Randomization Level')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)

    # Plot balanced times
    plt.subplot(1, 2, 2)
    means = [results[r]['mean_balanced_time'] for r in randomization_levels]
    stds = [results[r]['std_balanced_time'] for r in randomization_levels]
    plt.bar(x_labels, means, yerr=stds, capsize=10)
    plt.title('Balancing Time vs. Parameter Randomization')
    plt.xlabel('Randomization Level')
    plt.ylabel('Average Time Balanced (s)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("robustness_summary.png")
    #plt.show()

    return results


def load_model(actor_path, critic_path=None, state_dim=6, action_dim=1, hidden_dim=256):
    """
    Load a pre-trained SAC model from saved state dictionaries.

    Args:
        actor_path (str): Path to the saved actor model state dict
        critic_path (str, optional): Path to the saved critic model state dict
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
        hidden_dim (int): Hidden dimension size of the networks

    Returns:
        SACAgent: Loaded agent with the pre-trained weights
    """
    print(f"Loading pre-trained model from {actor_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize a new agent
    agent = SACAgent(state_dim, action_dim, hidden_dim)

    # Load actor state dict
    agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
    agent.actor.eval()  # Set to evaluation mode

    # Load critic if provided
    if critic_path:
        agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
        agent.critic.eval()

        # Copy critic weights to target critic
        for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
            target_param.data.copy_(param.data)

    print("Model loaded successfully!")
    return agent


def run_loaded_model(model_path, randomization_factor=0.1, delay_steps=5,
                    delay_range=None, num_episodes=5):
    """
    Run a loaded model in the pendulum environment and evaluate its performance.
    """
    # Environment setup as before
    env = PendulumEnv(randomization_factor=randomization_factor,
                     delay_steps=delay_steps,
                     delay_range=delay_range)

    # Load the agent
    agent = load_model(model_path)

    # Performance metrics
    performance = {
        'rewards': [],
        'balanced_times': [],
        'params_used': []
    }

    # Run episodes
    for episode in range(num_episodes):
        # Reset environment (which randomizes parameters)
        state = env.reset(random_init=True)
        total_reward = 0

        # Store episode data
        states_history = []
        actions_history = []

        # Remember parameters for this episode
        performance['params_used'].append({
            'Rm': env.params.Rm,
            'Km': env.params.Km,
            'mA': env.params.mA,
            'mL': env.params.mL,
            'LA': env.params.LA,
            'LL': env.params.LL,
            'g': env.params.g
        })

        for step in range(env.max_steps):
            # Get action from model
            action = agent.select_action(state, evaluate=True)

            # Cast to float explicitly to avoid the deprecation warning
            if isinstance(action, np.ndarray):
                action_value = float(action.item())
            else:
                action_value = float(action)

            # Take step in environment
            next_state, reward, done, _ = env.step(action_value)

            # Record data
            states_history.append(env.state.copy())
            actions_history.append(action_value)

            # Update metrics
            total_reward += reward
            state = next_state

            if done:
                break

        # Calculate balancing performance
        balanced_time = 0
        for state in states_history:
            alpha_norm = normalize_angle(state[1] + np.pi)
            if abs(alpha_norm) < 0.17:  # ~10 degrees from vertical
                balanced_time += env.dt

        # Store performance metrics
        performance['rewards'].append(total_reward)
        performance['balanced_times'].append(balanced_time)

        # Print results
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Balanced = {balanced_time:.2f}s")

        # Plot the episode
        states_array = np.array(states_history)
        actions_array = np.array(actions_history)

        # Create visualization
        plot_episode(episode, states_array, actions_array, env.dt,
                     total_reward, balanced_time, env.params,
                     prefix="model_run", randomized=True,
                     delay_steps=delay_steps, env=env)

    # Plot summary statistics
    if num_episodes > 1:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, num_episodes + 1), performance['rewards'])
        plt.axhline(y=np.mean(performance['rewards']), color='r', linestyle='--',
                    label=f'Mean: {np.mean(performance["rewards"]):.2f}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.bar(range(1, num_episodes + 1), performance['balanced_times'])
        plt.axhline(y=np.mean(performance['balanced_times']), color='r', linestyle='--',
                    label=f'Mean: {np.mean(performance["balanced_times"]):.2f}s')
        plt.xlabel('Episode')
        plt.ylabel('Time Balanced (s)')
        plt.title('Balancing Performance')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig("model_performance_summary.png")
        #plt.show()

    return performance


def plot_episode(episode, states_history, actions_history, dt, episode_reward,
                 balanced_time, params=None, prefix="episode", randomized=False, delay_steps=None, env=None):
    """
    Plot the pendulum state evolution for an episode

    Args:
        episode (int): Episode number
        states_history (numpy.ndarray): Array of state histories
        actions_history (numpy.ndarray): Array of actions
        dt (float): Timestep size
        episode_reward (float): Total episode reward
        balanced_time (float): Time spent balanced
        params (SystemParameters, optional): Randomized parameters if available
        prefix (str): Prefix for the saved file
        randomized (bool): Whether parameters are randomized
        delay_steps (int, optional): Fixed delay steps (if not using variable delay)
        env (PendulumEnv, optional): Environment object to extract current delay info
    """
    # Extract components
    thetas = states_history[:, 0]
    alphas = states_history[:, 1]
    theta_dots = states_history[:, 2]
    alpha_dots = states_history[:, 3]

    # Normalize alpha for visualization
    alpha_normalized = np.zeros(len(alphas))
    for i in range(len(alphas)):
        alpha_normalized[i] = normalize_angle(alphas[i] + np.pi)

    # Generate time array
    t = np.arange(len(states_history)) * dt

    # Get delay information
    delay_info = ""
    if env is not None and hasattr(env, 'current_delay'):
        delay_info = f"Delay: {env.current_delay} steps ({env.current_delay * dt * 1000:.1f}ms), "
    elif delay_steps is not None:
        delay_info = f"Delay: {delay_steps} steps ({delay_steps * dt * 1000:.1f}ms), "

    # Plot results
    plt.figure(figsize=(12, 10))

    # Plot arm angle
    plt.subplot(4, 1, 1)
    plt.plot(t, thetas, 'b-')
    plt.axhline(y=2.2, color='r', linestyle='--', alpha=0.7, label='Theta Limits')
    plt.axhline(y=-2.2, color='r', linestyle='--', alpha=0.7)
    plt.ylabel('Arm angle (rad)')

    # Add randomized parameters to title if provided
    if randomized and params:
        param_text = f"{delay_info}Rm={params.Rm:.2f}, Km={params.Km:.4f}, mL={params.mL:.4f}, LL={params.LL:.4f}"
        plt.title(
            f'{prefix.capitalize()} {episode + 1} | Reward: {episode_reward:.2f} | Balanced: {balanced_time:.2f}s\n{param_text}')
    else:
        plt.title(
            f'{prefix.capitalize()} {episode + 1} | Reward: {episode_reward:.2f} | Balanced: {balanced_time:.2f}s\n{delay_info}')

    plt.legend()
    plt.grid(True)

    # Plot pendulum angle
    plt.subplot(4, 1, 2)
    plt.plot(t, alpha_normalized, 'g-')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)  # Line at upright position
    plt.axhline(y=0.17, color='r', linestyle=':', alpha=0.5)  # Upper balanced threshold
    plt.axhline(y=-0.17, color='r', linestyle=':', alpha=0.5)  # Lower balanced threshold
    plt.ylabel('Pendulum angle (rad)')
    plt.grid(True)

    # Plot pendulum angular velocity
    plt.subplot(4, 1, 3)
    plt.plot(t, alpha_dots, 'g-')
    plt.ylabel('Pendulum velocity (rad/s)')
    plt.grid(True)

    # Plot control actions
    plt.subplot(4, 1, 4)
    plt.plot(t, actions_history * 10.0, 'r-')  # Scale back to actual voltage
    plt.xlabel('Time (s)')
    plt.ylabel('Control voltage (V)')
    plt.ylim([-11, 11])
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{prefix}_{episode + 1}.png")
    plt.close()


def continue_training(actor_path, critic_path=None, randomization_factor=0.1,
                     delay_steps=5, delay_range=None, additional_episodes=500, lr=1e-4):
    """
    Continue training a pre-trained SAC model.

    Args:
        actor_path (str): Path to the saved actor model state dict
        critic_path (str, optional): Path to the saved critic model state dict
        randomization_factor (float): Amount of parameter randomization
        delay_steps (int): Base number of delay steps
        delay_range (tuple, optional): Range of possible delays (min, max)
        additional_episodes (int): Number of additional episodes to train
        lr (float): Learning rate for continued training

    Returns:
        SACAgent: The further trained agent
    """
    # Create delay string for display
    if delay_range:
        delay_str = f"{delay_range[0]}-{delay_range[1]}"
    else:
        delay_str = str(delay_steps)

    print(f"Continuing training from model: {actor_path}")
    print(f"Training parameters: randomization={randomization_factor}, "
          f"delay={delay_str}, episodes={additional_episodes}")

    # Environment setup with randomization and variable delay
    env = PendulumEnv(randomization_factor=randomization_factor,
                      delay_steps=delay_steps,
                      delay_range=delay_range)

    # Hyperparameters
    state_dim = 6  # Our observation space
    action_dim = 1  # Motor voltage (normalized)
    max_steps = 100
    batch_size = 256 * 32
    replay_buffer_size = 200000  # Larger buffer for diverse experiences
    updates_per_step = 1

    # Load the pre-trained agent (with a potentially lower learning rate for fine-tuning)
    agent = load_model(actor_path, critic_path, state_dim, action_dim)

    # Optionally reduce learning rate for fine-tuning
    for param_group in agent.actor_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in agent.critic_optimizer.param_groups:
        param_group['lr'] = lr
    if agent.automatic_entropy_tuning:
        for param_group in agent.alpha_optimizer.param_groups:
            param_group['lr'] = lr

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # Pre-fill buffer with some experiences to start learning immediately
    print("Pre-filling replay buffer with initial experiences...")
    while len(replay_buffer) < batch_size:
        state = env.reset()
        for step in range(100):  # Collect some steps
            action = agent.select_action(state)

            # Cast to float explicitly
            if isinstance(action, np.ndarray):
                action_value = float(action.item())
            else:
                action_value = float(action)

            next_state, reward, done, _ = env.step(action_value)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    # Metrics
    episode_rewards = []
    avg_rewards = []
    start_time = time()

    # Training loop
    for episode in range(additional_episodes):
        state = env.reset()  # Randomize parameters for this episode
        episode_reward = 0

        # If we're going to plot this episode, prepare to collect state data
        plot_this_episode = (episode + 1) % 10 == 0
        if plot_this_episode:
            episode_states = []
            episode_actions = []
            episode_params = {
                'Rm': env.params.Rm,
                'Km': env.params.Km,
                'mA': env.params.mA,
                'mL': env.params.mL,
                'LA': env.params.LA,
                'LL': env.params.LL,
                'g': env.params.g,
            }

        for step in range(max_steps):
            action = agent.select_action(state)

            # Cast to float explicitly
            if isinstance(action, np.ndarray):
                action_value = float(action.item())
            else:
                action_value = float(action)

            next_state, reward, done, _ = env.step(action_value)

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # If plotting, store the state and action
            if plot_this_episode:
                episode_states.append(env.state.copy())
                episode_actions.append(action_value)

            state = next_state
            episode_reward += reward

            # Update parameters - note we have enough samples from pre-filling
            for _ in range(updates_per_step):
                agent.update_parameters(replay_buffer, batch_size)

            if done:
                break

        # Log progress
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{additional_episodes} | Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f}")

            # Plot simulation for visual progress tracking
            if plot_this_episode:
                plot_training_episode(episode, episode_states, episode_actions,
                                      env.dt, episode_reward, episode_params,
                                      delay_steps, env)

        # Early stopping if well trained
        if avg_reward > 6000 and len(episode_rewards) >= 100:
            print(f"Environment solved with excellent performance after {episode + 1} episodes!")
            break

    training_time = time() - start_time
    print(f"Additional training completed in {training_time:.2f} seconds!")

    # Save continued training model with timestamp
    timestamp = int(time())
    torch.save(agent.actor.state_dict(), f"continued_sac_actor_{timestamp}.pth")
    torch.save(agent.critic.state_dict(), f"continued_sac_critic_{timestamp}.pth")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Moving Average')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title(f'Continued SAC Training Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"continued_training_{timestamp}.png")
    #plt.show()

    return agent


# Update the main function to properly handle command line arguments
if __name__ == "__main__":
    print("TorchRL Inverted Pendulum Training and Evaluation")
    print("=" * 50)

    # Check for command line arguments to determine mode
    import sys
    import argparse

    # Create an argument parser for better command-line handling
    parser = argparse.ArgumentParser(description='Pendulum RL Training and Evaluation')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train from scratch
    train_parser = subparsers.add_parser('train', help='Train a new model from scratch')
    train_parser.add_argument('--random', type=float, default=0.1, help='Randomization factor (0.0-1.0)')
    train_parser.add_argument('--delay', type=int, default=5, help='Fixed observation delay steps')
    train_parser.add_argument('--delay-range', type=str, help='Variable delay range, format: min-max (e.g., 0-5)')

    # Load and evaluate a model
    load_parser = subparsers.add_parser('load', help='Load and evaluate a pre-trained model')
    load_parser.add_argument('model_path', help='Path to the saved model file')
    load_parser.add_argument('--random', type=float, default=0.1, help='Randomization factor for evaluation')
    load_parser.add_argument('--delay', type=int, default=5, help='Fixed observation delay steps')
    load_parser.add_argument('--delay-range', type=str, help='Variable delay range, format: min-max (e.g., 0-5)')
    load_parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to evaluate')

    # Continue training a model
    continue_parser = subparsers.add_parser('continue', help='Continue training a pre-trained model')
    continue_parser.add_argument('model_path', help='Path to the saved actor model file')
    continue_parser.add_argument('--critic', help='Optional path to the saved critic model file')
    continue_parser.add_argument('--random', type=float, default=0.15, help='Randomization factor for training')
    continue_parser.add_argument('--delay', type=int, default=5, help='Fixed observation delay steps')
    continue_parser.add_argument('--delay-range', type=str, help='Variable delay range, format: min-max (e.g., 0-5)')
    continue_parser.add_argument('--episodes', type=int, default=500, help='Number of additional episodes to train')
    continue_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for continued training')
    continue_parser.add_argument('--eval', action='store_true', help='Evaluate the model after training')

    # Support for legacy command-line format
    if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
        # Handle the old format (--load, --continue)
        if sys.argv[1] == '--load':
            sys.argv[1] = 'load'
            if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
                # Ensure the model_path is properly positioned
                model_path = sys.argv[2]
                sys.argv.pop(2)
                sys.argv.insert(2, model_path)
        elif sys.argv[1] == '--continue':
            sys.argv[1] = 'continue'
            if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
                # Ensure the model_path is properly positioned
                model_path = sys.argv[2]
                sys.argv.pop(2)
                sys.argv.insert(2, model_path)

    # Parse arguments
    args = parser.parse_args()

    # Default to train if no command specified
    if not args.command:
        args.command = 'train'
        args.random = 0.1
        args.delay = 5
        args.delay_range = None

    # Parse delay range if provided
    delay_range = None
    if hasattr(args, 'delay_range') and args.delay_range:
        try:
            min_delay, max_delay = map(int, args.delay_range.split('-'))
            delay_range = (min_delay, max_delay)
            print(f"Using variable delay in range {delay_range}")
        except ValueError:
            print(f"Warning: Invalid delay range format '{args.delay_range}'. Using fixed delay of {args.delay}.")

    # Execute the appropriate command
    if args.command == 'train':
        print(f"\nTraining new model from scratch...")
        print(f"Parameters: randomization={args.random}, delay={args.delay if not delay_range else delay_range}")
        agent = train(
            randomization_factor=args.random,
            delay_steps=args.delay,
            delay_range=delay_range
        )

        # Evaluate trained agent
        print("\nEvaluating trained agent...")
        evaluate_robustness(
            agent,
            num_episodes=2,
            randomization_levels=[0.0, args.random, args.random * 2],
            delay_steps=args.delay,
            delay_range=delay_range
        )

    elif args.command == 'load':
        print(f"\nRunning pre-trained model from {args.model_path}...")
        performance = run_loaded_model(
            args.model_path,
            randomization_factor=args.random,
            delay_steps=args.delay,
            delay_range=delay_range,
            num_episodes=args.episodes
        )

        # Print performance statistics
        avg_reward = np.mean(performance['rewards'])
        avg_balanced = np.mean(performance['balanced_times'])
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average balancing time: {avg_balanced:.2f}s")

    elif args.command == 'continue':
        print(f"\nContinuing training from model: {args.model_path}")
        agent = continue_training(
            args.model_path,
            args.critic,
            randomization_factor=args.random,
            delay_steps=args.delay,
            delay_range=delay_range,
            additional_episodes=args.episodes,
            lr=args.lr
        )

        # Optionally evaluate the improved agent
        if args.eval:
            print("\nEvaluating improved agent...")
            evaluate_robustness(
                agent,
                num_episodes=2,
                randomization_levels=[0.0, args.random, args.random * 2],
                delay_steps=args.delay,
                delay_range=delay_range
            )

    print("=" * 50)
    print("PROGRAM EXECUTION COMPLETE")
    print("=" * 50)